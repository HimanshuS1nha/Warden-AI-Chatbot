import uuid
from collections.abc import Iterator
from datetime import date
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.request_context import RequestContext, use_request_context, valid_ip_or_none
from warden.models import AuditAction, AuditLogModel, AuditOutcome, AuditTarget
from warden.services import audit


class FakeSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)


def record(**kwargs: Any) -> tuple[AuditLogModel, FakeSession]:
    db = FakeSession()
    entry = audit.record(
        cast(AsyncSession, db),
        action=kwargs.pop("action", AuditAction.LEAVE_APPLY),
        outcome=kwargs.pop("outcome", AuditOutcome.ALLOWED),
        **kwargs,
    )
    return entry, db


@pytest.fixture
def in_request() -> Iterator[None]:
    context = RequestContext(ip="10.0.0.7", user_agent="pytest", request_id="req-123")
    with use_request_context(context):
        yield


def test_adds_the_row_without_committing() -> None:
    actor, target = uuid.uuid4(), uuid.uuid4()

    entry, db = record(
        actor_id=actor,
        target_type=AuditTarget.LEAVE_REQUEST,
        target_id=target,
        details={"days": 2},
    )

    assert db.added == [entry]
    assert entry.action == "leave.apply"
    assert entry.target_type == "leave_request"
    assert (entry.actor_id, entry.target_id) == (actor, target)
    assert entry.details == {"days": 2}


def test_outside_a_request_there_is_no_ip_or_request_id() -> None:
    entry, _ = record(details={"days": 2})

    assert entry.ip is None
    assert entry.details == {"days": 2}


@pytest.mark.usefixtures("in_request")
def test_fills_ip_and_request_id_from_the_request_context() -> None:
    entry, _ = record(details={"days": 2})

    assert entry.ip == "10.0.0.7"
    assert entry.details == {"days": 2, "request_id": "req-123"}


def test_details_are_stored_as_plain_json() -> None:
    leave_id = uuid.uuid4()

    entry, _ = record(details={"id": leave_id, "start": date(2026, 3, 2)})

    assert entry.details == {"id": str(leave_id), "start": "2026-03-02"}


@pytest.mark.parametrize(
    "details",
    [
        {"password": "hunter2"},
        {"new_Password": "x"},
        {"refresh_token": "x"},
        {"client_secret": "x"},
        {"password_hash": "x"},
        {"request": {"body": {"password": "x"}}},
        {"attempts": [{"token": "x"}]},
    ],
)
def test_refuses_secret_keys(details: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="can't contain the key"):
        record(details=details)


def test_refuses_details_over_the_size_cap() -> None:
    record(details={"note": "x" * 4000})

    with pytest.raises(ValueError, match="bytes; the limit is 4096"):
        record(details={"note": "x" * audit.MAX_DETAILS_BYTES})


def test_nothing_is_added_when_details_are_refused() -> None:
    db = FakeSession()

    with pytest.raises(ValueError):
        audit.record(
            cast(AsyncSession, db),
            action=AuditAction.AUTH_LOGIN,
            outcome=AuditOutcome.DENIED,
            details={"password": "x"},
        )

    assert db.added == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
        ("0:0:0:0:0:0:0:1", "::1"),
        ("testclient", None),
        ("", None),
        (None, None),
    ],
)
def test_valid_ip_or_none(value: str | None, expected: str | None) -> None:
    assert valid_ip_or_none(value) == expected
