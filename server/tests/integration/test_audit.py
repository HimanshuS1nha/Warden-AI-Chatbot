from collections.abc import Iterator, Sequence

import pytest
from fastapi import Depends
from httpx import AsyncClient
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import PASSWORD, MakeEmployee
from tests.integration.test_auth import login, send_with_cookie
from warden.core.dependencies import require_role
from warden.main import app
from warden.models import (
    AuditAction,
    AuditLogModel,
    AuditOutcome,
    AuditTarget,
    EmployeeModel,
    EmployeeSessionModel,
    Role,
)
from warden.services import audit

# ASGITransport reports this as the client address.
CLIENT_IP = "127.0.0.1"


async def audit_rows(db: AsyncSession, action: AuditAction) -> Sequence[AuditLogModel]:
    # Rows may have been written or changed through other sessions on the same
    # connection, so reload them instead of trusting the identity map.
    result = await db.scalars(
        select(AuditLogModel)
        .where(AuditLogModel.action == action)
        .order_by(AuditLogModel.id)
        .execution_options(populate_existing=True)
    )
    return result.all()


def request_details(row: AuditLogModel) -> dict[str, object]:
    """`details` without the request ID, after checking it's there. Every row
    written during an HTTP request carries the ID from the X-Request-ID header."""
    details = dict(row.details)
    request_id = details.pop("request_id")
    assert isinstance(request_id, str) and len(request_id) == 32
    return details


async def session_ids(db: AsyncSession, employee: EmployeeModel) -> list[object]:
    result = await db.scalars(
        select(EmployeeSessionModel.id)
        .where(EmployeeSessionModel.employee_id == employee.id)
        .order_by(EmployeeSessionModel.id)
    )
    return list(result.all())


# Auth events


async def test_login_success_records_the_new_session(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()

    await login(client, employee)

    [row] = await audit_rows(db, AuditAction.AUTH_LOGIN)
    assert row.outcome == AuditOutcome.ALLOWED
    assert row.actor_id == employee.id
    assert row.target_type == AuditTarget.SESSION
    assert [row.target_id] == await session_ids(db, employee)
    assert request_details(row) == {}
    assert str(row.ip) == CLIENT_IP


@pytest.mark.parametrize(
    ("email", "password", "is_active", "reason", "actor_is_employee"),
    [
        ("employee1@warden.demo", "wrong-password", True, "wrong_password", True),
        ("nobody@warden.demo", PASSWORD, True, "unknown_email", False),
        ("employee1@warden.demo", PASSWORD, False, "inactive", True),
    ],
    ids=["wrong-password", "unknown-email", "inactive-employee"],
)
async def test_failed_login_records_the_reason_but_responds_the_same(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    email: str,
    password: str,
    is_active: bool,
    reason: str,
    actor_is_employee: bool,
) -> None:
    employee = await make_employee(email="employee1@warden.demo", is_active=is_active)

    response = await client.post(
        "/api/auth/login", json={"email": f"  {email.upper()} ", "password": password}
    )

    # Same body for every reason, as in test_auth.py; only the row differs.
    assert response.status_code == 401
    assert response.json() == {"message": "Invalid email or password"}
    [row] = await audit_rows(db, AuditAction.AUTH_LOGIN)
    assert row.outcome == AuditOutcome.DENIED
    assert row.actor_id == (employee.id if actor_is_employee else None)
    assert row.target_type is None and row.target_id is None
    assert request_details(row) == {"email": email, "reason": reason}
    assert str(row.ip) == CLIENT_IP


async def test_failed_login_row_never_contains_the_password(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    tried = "Sup3r-Secret-Guess!"

    await client.post("/api/auth/login", json={"email": employee.email, "password": tried})

    # The whole row as text, not only `details`.
    rows = (await db.execute(text("SELECT audit_logs::text FROM audit_logs"))).scalars().all()
    assert len(rows) == 1
    assert tried not in rows[0]
    assert tried.lower() not in rows[0].lower()


async def test_logout_records_whether_a_session_was_revoked(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    tokens = (await login(client, employee)).cookies
    [session_id] = await session_ids(db, employee)
    refresh_cookie = f"refresh_token={tokens['refresh_token']}"

    await send_with_cookie(client, "POST", "/api/auth/logout", refresh_cookie)
    # Logging out again with the same token finds the session already revoked.
    await send_with_cookie(client, "POST", "/api/auth/logout", refresh_cookie)

    first, second = await audit_rows(db, AuditAction.AUTH_LOGOUT)
    for row in (first, second):
        assert row.outcome == AuditOutcome.ALLOWED
        assert row.actor_id == employee.id
        assert (row.target_type, row.target_id) == (AuditTarget.SESSION, session_id)
    assert request_details(first) == {"revoked": True}
    assert request_details(second) == {"revoked": False}


async def test_logout_without_tokens_records_nothing(client: AsyncClient, db: AsyncSession) -> None:
    response = await client.post("/api/auth/logout")

    assert response.status_code == 204
    assert await audit_rows(db, AuditAction.AUTH_LOGOUT) == []


async def test_refresh_token_reuse_records_how_many_sessions_were_revoked(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    first = await login(client, employee)
    old_refresh = first.cookies["refresh_token"]
    await client.post("/api/auth/refresh")
    await login(client, employee)
    reused_session_id, _ = await session_ids(db, employee)

    await send_with_cookie(client, "POST", "/api/auth/refresh", f"refresh_token={old_refresh}")

    [row] = await audit_rows(db, AuditAction.AUTH_REFRESH_REUSE)
    assert row.outcome == AuditOutcome.DENIED
    assert row.actor_id == employee.id
    assert (row.target_type, row.target_id) == (AuditTarget.SESSION, reused_session_id)
    assert request_details(row) == {"revoked_sessions": 2}


async def test_routine_refresh_records_nothing(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    await login(client, await make_employee())

    assert (await client.post("/api/auth/refresh")).status_code == 200

    assert await audit_rows(db, AuditAction.AUTH_REFRESH_REUSE) == []
    assert len(await audit_rows(db, AuditAction.AUTH_LOGIN)) == 1


# require_role


@pytest.fixture
def hr_route() -> Iterator[None]:
    # Mounted on the real app, so the request context middleware runs too.
    path = "/api/test-hr-only"

    @app.get(path, dependencies=[Depends(require_role(Role.HR))])
    async def hr_only() -> None:
        pass

    yield
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != path]


@pytest.mark.usefixtures("hr_route")
async def test_forbidden_writes_one_row_and_unauthenticated_writes_none(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    assert (await client.get("/api/test-hr-only")).status_code == 401
    assert await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN) == []

    manager = await make_employee(role=Role.MANAGER)
    await login(client, manager)
    assert (await client.get("/api/test-hr-only")).status_code == 403

    [row] = await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)
    assert row.outcome == AuditOutcome.DENIED
    assert row.actor_id == manager.id
    assert request_details(row) == {
        "method": "GET",
        "path": "/api/test-hr-only",
        "role": "manager",
        "required": ["hr"],
    }
    assert str(row.ip) == CLIENT_IP


# Append-only trigger


async def add_row(db: AsyncSession, employee: EmployeeModel) -> AuditLogModel:
    row = audit.record(
        db, action=AuditAction.AUTH_LOGIN, outcome=AuditOutcome.ALLOWED, actor_id=employee.id
    )
    await db.flush()
    return row


@pytest.mark.parametrize(
    "values",
    [
        {"action": "auth.logout"},
        {"outcome": AuditOutcome.DENIED},
        {"details": {"edited": True}},
        # actor_id may only go to NULL, never to someone else.
        {"actor_id": None, "outcome": AuditOutcome.DENIED},
    ],
    ids=["action", "outcome", "details", "actor-and-outcome"],
)
async def test_updates_are_refused(
    db: AsyncSession, make_employee: MakeEmployee, values: dict[str, object]
) -> None:
    row = await add_row(db, await make_employee())

    with pytest.raises(DBAPIError, match="append-only: UPDATE"):
        async with db.begin_nested():
            await db.execute(
                update(AuditLogModel).where(AuditLogModel.id == row.id).values(**values)
            )


async def test_changing_the_actor_is_refused(db: AsyncSession, make_employee: MakeEmployee) -> None:
    row = await add_row(db, await make_employee())
    other = await make_employee()

    with pytest.raises(DBAPIError, match="append-only: UPDATE"):
        async with db.begin_nested():
            await db.execute(
                update(AuditLogModel).where(AuditLogModel.id == row.id).values(actor_id=other.id)
            )


async def test_delete_and_truncate_are_refused(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    row = await add_row(db, await make_employee())

    with pytest.raises(DBAPIError, match="append-only: DELETE"):
        async with db.begin_nested():
            await db.execute(delete(AuditLogModel).where(AuditLogModel.id == row.id))
    with pytest.raises(DBAPIError, match="append-only: TRUNCATE"):
        async with db.begin_nested():
            await db.execute(text("TRUNCATE audit_logs"))


async def test_deleting_an_employee_keeps_their_rows_with_no_actor(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    row = await add_row(db, employee)

    await db.execute(delete(EmployeeModel).where(EmployeeModel.id == employee.id))

    kept = await db.get(AuditLogModel, row.id, populate_existing=True)
    assert kept is not None
    assert kept.actor_id is None
    assert kept.action == AuditAction.AUTH_LOGIN
    count = await db.scalar(select(func.count()).select_from(AuditLogModel))
    assert count == 1
