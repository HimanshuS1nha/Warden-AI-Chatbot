from typing import Annotated

from fastapi import APIRouter, Cookie, Request, Response, status

from warden.core.config import get_settings
from warden.core.cookies import (
    ACCESS_TOKEN_COOKIE,
    REFRESH_TOKEN_COOKIE,
    REFRESH_TOKEN_COOKIE_PATH,
)
from warden.core.dependencies import CurrentUser, DbSession
from warden.schemas.auth import LoginRequest, LoginResult, RegisterRequest
from warden.schemas.user import UserResponse
from warden.services.auth import login_user, refresh_session, register_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, db: DbSession) -> UserResponse:
    return await register_user(db, data)


@router.post("/login")
async def login(
    data: LoginRequest, request: Request, response: Response, db: DbSession
) -> UserResponse:
    result = await login_user(
        db,
        data,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    _set_auth_cookies(response, result)
    return result.user



@router.post("/refresh")
async def refresh(
    response: Response,
    db: DbSession,
    refresh_token: Annotated[str | None, Cookie(alias=REFRESH_TOKEN_COOKIE)] = None,
) -> UserResponse:
    result = await refresh_session(db, refresh_token)
    _set_auth_cookies(response, result)
    return result.user


@router.get("/me")
async def me(user: CurrentUser) -> UserResponse:
    return user


def _set_auth_cookies(response: Response, result: LoginResult) -> None:
    settings = get_settings()
    response.set_cookie(
        ACCESS_TOKEN_COOKIE,
        result.access_token,
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    response.set_cookie(
        REFRESH_TOKEN_COOKIE,
        result.refresh_token,
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        path=REFRESH_TOKEN_COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
