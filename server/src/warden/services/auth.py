from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.hashing import hash_password
from warden.models import UserModel
from warden.schemas.auth import RegisterRequest


def email_exists_error() -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")


async def register_user(db: AsyncSession, data: RegisterRequest) -> UserModel:
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

    return user
