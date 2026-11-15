from datetime import UTC, datetime
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.cookies import ACCESS_TOKEN_COOKIE
from warden.core.database import get_db
from warden.core.jwt import decode_token
from warden.models import UserModel, UserSessionModel
from warden.schemas.user import UserResponse

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    db: DbSession,
    access_token: Annotated[str | None, Cookie(alias=ACCESS_TOKEN_COOKIE)] = None,
) -> UserResponse:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )
    if not access_token:
        raise unauthorized

    payload = decode_token(access_token, "access")

    # Load the user only if the session the token was issued for is still
    # active, so logout and revocation take effect right away instead of
    # when the access token expires.
    user = await db.scalar(
        select(UserModel)
        .join(UserSessionModel, UserSessionModel.user_id == UserModel.id)
        .where(
            UserModel.id == payload.user_id,
            UserSessionModel.id == payload.session_id,
            UserSessionModel.revoked_at.is_(None),
            UserSessionModel.expires_at > datetime.now(UTC),
        )
    )
    if user is None:
        raise unauthorized

    return UserResponse.model_validate(user)


CurrentUser = Annotated[UserResponse, Depends(get_current_user)]
