import hmac
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_utils.compat import uuid7

from warden.core.hashing import (
    hash_token,
    password_hash,
    verify_and_update_password,
    verify_password,
)
from warden.core.jwt import (
    TokenPayload,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from warden.core.request_context import get_request_context
from warden.models import (
    AuditAction,
    AuditOutcome,
    AuditTarget,
    EmployeeModel,
    EmployeeSessionModel,
)
from warden.schemas.auth import LoginRequest, LoginResult
from warden.schemas.employee import EmployeeResponse
from warden.services import audit

# Checked against when the email doesn't exist, so a login for an unknown
# email takes as long as one with a wrong password and doesn't reveal which
# emails have accounts.
_DUMMY_PASSWORD_HASH = password_hash.hash("dummy-password-for-timing")


def invalid_credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
    )


def invalid_refresh_token_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
    )


async def login_user(db: AsyncSession, data: LoginRequest) -> LoginResult:
    email = data.email.strip().lower()
    employee = await db.scalar(select(EmployeeModel).where(EmployeeModel.email == email))

    if employee is None:
        await verify_password(data.password, _DUMMY_PASSWORD_HASH)
        await _record_failed_login(db, email, "unknown_email")
        raise invalid_credentials_error()

    is_valid, new_hash = await verify_and_update_password(data.password, employee.password_hash)
    # A deactivated employee gets the same error as a wrong password, so the
    # response doesn't reveal which accounts exist. Only the audit row says
    # which it was.
    if not is_valid:
        await _record_failed_login(db, email, "wrong_password", employee)
        raise invalid_credentials_error()
    if not employee.is_active:
        await _record_failed_login(db, email, "inactive", employee)
        raise invalid_credentials_error()

    if new_hash is not None:
        # Stored hash used older argon2 settings; save the upgraded one.
        employee.password_hash = new_hash

    session_id = uuid7()
    access_token = create_access_token(employee.id, session_id)
    refresh_token, refresh_expires_at = create_refresh_token(employee.id, session_id)

    context = get_request_context()
    db.add(
        EmployeeSessionModel(
            id=session_id,
            employee_id=employee.id,
            refresh_token_hash=hash_token(refresh_token),
            user_agent=context.user_agent,
            ip_address=context.ip,
            expires_at=refresh_expires_at,
        )
    )
    audit.record(
        db,
        action=AuditAction.AUTH_LOGIN,
        outcome=AuditOutcome.ALLOWED,
        actor_id=employee.id,
        target_type=AuditTarget.SESSION,
        target_id=session_id,
    )
    await db.commit()

    return LoginResult(
        employee=EmployeeResponse.model_validate(employee),
        access_token=access_token,
        refresh_token=refresh_token,
        refresh_expires_at=refresh_expires_at,
    )


async def refresh_session(db: AsyncSession, refresh_token: str | None) -> LoginResult:
    if not refresh_token:
        raise invalid_refresh_token_error()

    payload = decode_token(refresh_token, "refresh")
    now = datetime.now(UTC)

    # Lock the row so two refreshes with the same token can't both rotate it.
    session = await db.scalar(
        select(EmployeeSessionModel)
        .where(
            EmployeeSessionModel.id == payload.session_id,
            EmployeeSessionModel.employee_id == payload.employee_id,
        )
        .with_for_update()
    )
    if session is None or session.revoked_at is not None or session.expires_at <= now:
        raise invalid_refresh_token_error()

    if not hmac.compare_digest(session.refresh_token_hash, hash_token(refresh_token)):
        # A correctly signed token for this session that isn't the latest one
        # means an old token was used again. Either it was stolen or someone
        # is replaying it, so end every session for this employee.
        revoked = await db.scalars(
            update(EmployeeSessionModel)
            .where(
                EmployeeSessionModel.employee_id == session.employee_id,
                EmployeeSessionModel.revoked_at.is_(None),
            )
            .values(revoked_at=now)
            .returning(EmployeeSessionModel.id)
        )
        audit.record(
            db,
            action=AuditAction.AUTH_REFRESH_REUSE,
            outcome=AuditOutcome.DENIED,
            actor_id=session.employee_id,
            target_type=AuditTarget.SESSION,
            target_id=session.id,
            details={"revoked_sessions": len(revoked.all())},
        )
        await db.commit()
        raise invalid_refresh_token_error()

    employee = await db.get(EmployeeModel, session.employee_id)
    if employee is None or not employee.is_active:
        raise invalid_refresh_token_error()

    access_token = create_access_token(employee.id, session.id)
    new_refresh_token, refresh_expires_at = create_refresh_token(employee.id, session.id)

    session.refresh_token_hash = hash_token(new_refresh_token)
    session.expires_at = refresh_expires_at
    session.last_used_at = now
    await db.commit()

    return LoginResult(
        employee=EmployeeResponse.model_validate(employee),
        access_token=access_token,
        refresh_token=new_refresh_token,
        refresh_expires_at=refresh_expires_at,
    )


async def logout_user(
    db: AsyncSession, access_token: str | None, refresh_token: str | None
) -> None:
    # Prefer the refresh token: it outlives the access token, so logout still
    # finds the session after the access token has expired.
    payload = _decode_or_none(refresh_token, "refresh") or _decode_or_none(
        access_token, "access"
    )
    if payload is None:
        return

    session = await db.scalar(
        select(EmployeeSessionModel)
        .where(
            EmployeeSessionModel.id == payload.session_id,
            EmployeeSessionModel.employee_id == payload.employee_id,
        )
        .with_for_update()
    )
    # Sessions are only deleted along with their employee, so a missing row
    # means there's nobody to log out (and no actor to record).
    if session is None:
        return

    now = datetime.now(UTC)
    was_active = session.revoked_at is None and session.expires_at > now
    if session.revoked_at is None:
        session.revoked_at = now
    audit.record(
        db,
        action=AuditAction.AUTH_LOGOUT,
        outcome=AuditOutcome.ALLOWED,
        actor_id=session.employee_id,
        target_type=AuditTarget.SESSION,
        target_id=session.id,
        details={"revoked": was_active},
    )
    await db.commit()


async def _record_failed_login(
    db: AsyncSession, email: str, reason: str, employee: EmployeeModel | None = None
) -> None:
    """Record and commit a denied login. The actor is set for a known
    account, so HR can see repeated attempts against one employee."""
    audit.record(
        db,
        action=AuditAction.AUTH_LOGIN,
        outcome=AuditOutcome.DENIED,
        actor_id=employee.id if employee else None,
        details={"email": email, "reason": reason},
    )
    await db.commit()


def _decode_or_none(token: str | None, expected_type: TokenType) -> TokenPayload | None:
    if not token:
        return None
    try:
        return decode_token(token, expected_type)
    except HTTPException:
        return None
