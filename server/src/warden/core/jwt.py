import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from fastapi import HTTPException, status

from warden.core.config import get_settings

TokenType = Literal["access", "refresh"]


@dataclass(frozen=True)
class TokenPayload:
    user_id: int
    session_id: uuid.UUID
    type: TokenType
    expires_at: datetime


def _create_token(
    user_id: int, session_id: uuid.UUID, token_type: TokenType, lifetime: timedelta
) -> tuple[str, datetime]:
    settings = get_settings()
    now = datetime.now(UTC)
    expires_at = now + lifetime
    claims = {
        "sub": str(user_id),
        "sid": str(session_id),
        "type": token_type,
        "iat": now,
        "exp": expires_at,
        # Unique per token, so two tokens issued in the same second still
        # differ and get different hashes in user_sessions.
        "jti": uuid.uuid4().hex,
    }
    token = jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, expires_at


def create_access_token(user_id: int, session_id: uuid.UUID) -> str:
    lifetime = timedelta(minutes=get_settings().access_token_expire_minutes)
    token, _ = _create_token(user_id, session_id, "access", lifetime)
    return token


def create_refresh_token(user_id: int, session_id: uuid.UUID) -> tuple[str, datetime]:
    """Return the token and its expiry. Store the expiry on the session row."""
    lifetime = timedelta(days=get_settings().refresh_token_expire_days)
    return _create_token(user_id, session_id, "refresh", lifetime)


def decode_token(token: str, expected_type: TokenType) -> TokenPayload:
    """Verify signature, expiry and token type. Raises 401 on any failure."""
    settings = get_settings()
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            # Fixed list: never trust the algorithm named in the token header.
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "sid", "type", "iat", "exp", "jti"]},
        )
    except jwt.InvalidTokenError:
        raise unauthorized from None

    # Stops a refresh token being used as an access token and vice versa.
    if claims["type"] != expected_type:
        raise unauthorized

    try:
        return TokenPayload(
            user_id=int(claims["sub"]),
            session_id=uuid.UUID(claims["sid"]),
            type=expected_type,
            expires_at=datetime.fromtimestamp(claims["exp"], UTC),
        )
    except (TypeError, ValueError):
        raise unauthorized from None
