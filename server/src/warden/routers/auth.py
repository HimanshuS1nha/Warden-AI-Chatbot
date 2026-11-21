from typing import Annotated

from fastapi import APIRouter, Cookie, HTTPException, Response, status

from warden.core.config import get_settings
from warden.core.cookies import (
    ACCESS_TOKEN_COOKIE,
    REFRESH_TOKEN_COOKIE,
    REFRESH_TOKEN_COOKIE_PATH,
)
from warden.core.dependencies import CurrentUserDep, DbSession
from warden.core.rate_limit import RateLimiterDep
from warden.models import EmployeeModel
from warden.schemas.auth import LoginRequest, LoginResult
from warden.schemas.employee import EmployeeResponse
from warden.services.auth import (
    check_login_rate_limit,
    login_user,
    logout_user,
    refresh_session,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/login",
    responses={429: {"description": "Too many attempts from this IP or for this email."}},
)
async def login(
    data: LoginRequest, response: Response, db: DbSession, limiter: RateLimiterDep
) -> EmployeeResponse:
    # The client IP and user agent come from the request context.
    await check_login_rate_limit(db, limiter, data.email)
    result = await login_user(db, data)
    _set_auth_cookies(response, result)
    return result.employee


@router.post("/refresh")
async def refresh(
    response: Response,
    db: DbSession,
    refresh_token: Annotated[str | None, Cookie(alias=REFRESH_TOKEN_COOKIE)] = None,
) -> EmployeeResponse:
    result = await refresh_session(db, refresh_token)
    _set_auth_cookies(response, result)
    return result.employee


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    db: DbSession,
    access_token: Annotated[str | None, Cookie(alias=ACCESS_TOKEN_COOKIE)] = None,
    refresh_token: Annotated[str | None, Cookie(alias=REFRESH_TOKEN_COOKIE)] = None,
) -> None:
    # No CurrentUserDep dependency: logout must still work, and clear cookies,
    # when the access token has expired or the session is already revoked.
    await logout_user(db, access_token, refresh_token)
    _clear_auth_cookies(response)


@router.get("/me")
async def me(user: CurrentUserDep, db: DbSession) -> EmployeeResponse:
    # get_current_user already loaded this row in the same session, so this
    # comes from the identity map without another query.
    employee = await db.get(EmployeeModel, user.employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return EmployeeResponse.model_validate(employee)


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


def _clear_auth_cookies(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        ACCESS_TOKEN_COOKIE,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    response.delete_cookie(
        REFRESH_TOKEN_COOKIE,
        path=REFRESH_TOKEN_COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
