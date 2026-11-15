import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi import HTTPException

from warden.core.config import get_settings
from warden.core.jwt import create_access_token, create_refresh_token, decode_token

EMPLOYEE_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()


def test_tokens_round_trip() -> None:
    access = decode_token(create_access_token(EMPLOYEE_ID, SESSION_ID), "access")
    refresh_token, expires_at = create_refresh_token(EMPLOYEE_ID, SESSION_ID)
    refresh = decode_token(refresh_token, "refresh")

    for payload in (access, refresh):
        assert payload.employee_id == EMPLOYEE_ID
        assert payload.session_id == SESSION_ID
    assert refresh.expires_at == expires_at.replace(microsecond=0)


def _claims(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "sub": str(EMPLOYEE_ID),
        "sid": str(SESSION_ID),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "jti": "x",
    } | overrides


def _sign(claims: dict[str, Any], key: str | None = None, algorithm: str = "HS256") -> str:
    if algorithm == "none":
        return jwt.encode(claims, None, algorithm="none")
    return jwt.encode(claims, key or get_settings().jwt_secret_key, algorithm=algorithm)


def _tampered() -> str:
    header, _, signature = create_access_token(EMPLOYEE_ID, SESSION_ID).split(".")
    other_payload = _sign(_claims(sub=str(uuid.uuid4()))).split(".")[1]
    return f"{header}.{other_payload}.{signature}"


@pytest.mark.parametrize(
    "make_token",
    [
        lambda: create_refresh_token(EMPLOYEE_ID, SESSION_ID)[0],
        lambda: _sign(_claims(exp=datetime.now(UTC) - timedelta(seconds=1))),
        lambda: _sign(_claims(), key="another-secret-key-that-is-32-characters-long"),
        lambda: _sign(_claims(), algorithm="none"),
        _tampered,
        lambda: _sign({k: v for k, v in _claims().items() if k != "sid"}),
        lambda: _sign(_claims(sub="not-a-uuid")),
        lambda: "not-a-jwt",
    ],
    ids=[
        "refresh-used-as-access",
        "expired",
        "wrong-secret",
        "alg-none",
        "tampered-payload",
        "missing-claim",
        "bad-subject",
        "garbage",
    ],
)
def test_invalid_access_tokens_are_rejected(make_token: Callable[[], str]) -> None:
    with pytest.raises(HTTPException) as error:
        decode_token(make_token(), "access")
    assert error.value.status_code == 401
