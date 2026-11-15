import ipaddress
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.hashing import (
    hash_password,
    hash_token,
    password_hash,
    verify_and_update_password,
    verify_password,
)
from warden.core.jwt import create_access_token, create_refresh_token
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


def _valid_ip_or_none(value: str | None) -> str | None:
    # The INET column rejects anything that isn't an IP address.
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None
