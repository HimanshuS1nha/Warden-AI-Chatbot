import hmac
import ipaddress
import uuid
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.hashing import (
    hash_password,
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
from warden.models import UserModel, UserSessionModel
from warden.schemas.auth import LoginRequest, LoginResult, RegisterRequest
from warden.schemas.user import UserResponse

# Checked against when the email doesn't exist, so a login for an unknown
# email takes as long as one with a wrong password and doesn't reveal which
# emails have accounts.
_DUMMY_PASSWORD_HASH = password_hash.hash("dummy-password-for-timing")

_MAX_USER_AGENT_LENGTH = 512


def email_exists_error() -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")


def invalid_credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
    )


def invalid_refresh_token_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
    )


async def register_user(db: AsyncSession, data: RegisterRequest) -> UserResponse:
    email = data.email.strip().lower()

    existing = await db.scalar(select(UserModel.id).where(UserModel.email == email))
    if existing is not None:
        raise email_exists_error()

    user = UserModel(
        name=data.name,
        email=email,
        password_hash=await hash_password(data.password),
    )

    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        # Two signups with the same email at once: the unique index catches
        # the one that loses the race.
        await db.rollback()
        raise email_exists_error() from exc

    # TODO: generate an OTP for email verification, store it hashed with an
    # expiry, and add a verify endpoint that sets email_verified_at.
    # TODO: send the OTP to the user's email once an email provider is wired up.

    return UserResponse.model_validate(user)


async def login_user(
    db: AsyncSession,
    data: LoginRequest,
    user_agent: str | None,
    ip_address: str | None,
) -> LoginResult:
    email = data.email.strip().lower()
    user = await db.scalar(select(UserModel).where(UserModel.email == email))

    if user is None:
        await verify_password(data.password, _DUMMY_PASSWORD_HASH)
        raise invalid_credentials_error()

    is_valid, new_hash = await verify_and_update_password(data.password, user.password_hash)
    if not is_valid:
        raise invalid_credentials_error()

    if new_hash is not None:
        # Stored hash used older argon2 settings; save the upgraded one.
        user.password_hash = new_hash

    if user.email_verified_at is None:
        # TODO: create a new OTP for this user (store it hashed with an expiry)
        # and send it to their email, so they can verify and then log in.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email not verified")

    session_id = uuid.uuid4()
    access_token = create_access_token(user.id, session_id)
    refresh_token, refresh_expires_at = create_refresh_token(user.id, session_id)

    db.add(
        UserSessionModel(
            id=session_id,
            user_id=user.id,
            refresh_token_hash=hash_token(refresh_token),
            user_agent=user_agent[:_MAX_USER_AGENT_LENGTH] if user_agent else None,
            ip_address=_valid_ip_or_none(ip_address),
            expires_at=refresh_expires_at,
        )
    )
    await db.commit()

    return LoginResult(
        user=UserResponse.model_validate(user),
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
        select(UserSessionModel)
        .where(
            UserSessionModel.id == payload.session_id,
            UserSessionModel.user_id == payload.user_id,
        )
        .with_for_update()
    )
    if session is None or session.revoked_at is not None or session.expires_at <= now:
        raise invalid_refresh_token_error()

    if not hmac.compare_digest(session.refresh_token_hash, hash_token(refresh_token)):
        # A correctly signed token for this session that isn't the latest one
        # means an old token was used again. Either it was stolen or someone
        # is replaying it, so end every session for this user.
        await db.execute(
            update(UserSessionModel)
            .where(
                UserSessionModel.user_id == session.user_id,
                UserSessionModel.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        await db.commit()
        raise invalid_refresh_token_error()

    user = await db.get(UserModel, session.user_id)
    if user is None:
        raise invalid_refresh_token_error()

    access_token = create_access_token(user.id, session.id)
    new_refresh_token, refresh_expires_at = create_refresh_token(user.id, session.id)

    session.refresh_token_hash = hash_token(new_refresh_token)
    session.expires_at = refresh_expires_at
    session.last_used_at = now
    await db.commit()

    return LoginResult(
        user=UserResponse.model_validate(user),
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

    await db.execute(
        update(UserSessionModel)
        .where(
            UserSessionModel.id == payload.session_id,
            UserSessionModel.user_id == payload.user_id,
            UserSessionModel.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()


def _decode_or_none(token: str | None, expected_type: TokenType) -> TokenPayload | None:
    if not token:
        return None
    try:
        return decode_token(token, expected_type)
    except HTTPException:
        return None


def _valid_ip_or_none(value: str | None) -> str | None:
    # The INET column rejects anything that isn't an IP address.
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None
