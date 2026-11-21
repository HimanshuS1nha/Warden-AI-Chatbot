"""GET /api/admin/audit-logs and GET /api/admin/unanswered-queries."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee
from tests.integration.test_auth import login
from warden.models import (
    AuditAction,
    AuditOutcome,
    AuditTarget,
    EmployeeModel,
    Role,
    UnansweredQueryModel,
    UnansweredReason,
)
from warden.services import audit

# Times in UTC. The company is on IST (UTC+5:30), so 20:00 UTC on 1 March is
# already 2 March there.
MARCH_1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
MARCH_1_LATE = datetime(2026, 3, 1, 20, 0, tzinfo=UTC)
MARCH_5 = datetime(2026, 3, 5, 10, 0, tzinfo=UTC)


@pytest.fixture
async def hr(client: AsyncClient, make_employee: MakeEmployee) -> EmployeeModel:
    employee = await make_employee(role=Role.HR, name="Hana HR", email="hana@warden.demo")
    await login(client, employee)
    return employee


async def add_log(
    db: AsyncSession,
    action: AuditAction,
    *,
    at: datetime = MARCH_1,
    actor: EmployeeModel | None = None,
    outcome: AuditOutcome = AuditOutcome.ALLOWED,
    target_id: uuid.UUID | None = None,
    **details: Any,
) -> uuid.UUID:
    entry = audit.record(
        db,
        action=action,
        outcome=outcome,
        actor_id=actor.id if actor else None,
        target_type=AuditTarget.LEAVE_REQUEST if target_id else None,
        target_id=target_id,
        details=details,
    )
    entry.created_at = at
    await db.commit()
    return entry.id


async def logs(client: AsyncClient, **params: Any) -> dict[str, Any]:
    response = await client.get("/api/admin/audit-logs", params=params)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# Audit logs


async def test_lists_entries_newest_first_with_the_actor(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel, make_employee: MakeEmployee
) -> None:
    ravi = await make_employee(name="Ravi Report", email="ravi@warden.demo")
    request_id = uuid.uuid4()
    first = await add_log(
        db, AuditAction.LEAVE_APPLY, actor=ravi, target_id=request_id, leave_type="casual"
    )
    second = await add_log(db, AuditAction.AUTH_LOGIN, outcome=AuditOutcome.DENIED, email="x@y")

    entries = (await logs(client))["entries"]

    # The HR user's own login, made before these, is last.
    assert [e["id"] for e in entries[:2]] == [str(second), str(first)]
    assert entries[2]["action"] == "auth.login" and entries[2]["actor_id"] == str(hr.id)
    anonymous, applied = entries[0], entries[1]
    assert anonymous["actor_id"] is None and anonymous["actor_name"] is None
    assert anonymous["outcome"] == "denied"
    assert applied == applied | {
        "actor_id": str(ravi.id),
        "actor_name": "Ravi Report",
        "actor_email": "ravi@warden.demo",
        "action": "leave.apply",
        "outcome": "allowed",
        "target_type": "leave_request",
        "target_id": str(request_id),
    }
    assert applied["details"]["leave_type"] == "casual"


async def test_filters(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel, make_employee: MakeEmployee
) -> None:
    ravi = await make_employee()
    target = uuid.uuid4()
    applied = await add_log(db, AuditAction.LEAVE_APPLY, actor=ravi, target_id=target)
    denied = await add_log(
        db, AuditAction.LEAVE_APPLY, actor=ravi, outcome=AuditOutcome.DENIED, reason="overlap"
    )
    approved = await add_log(db, AuditAction.LEAVE_APPROVE, actor=hr, target_id=target)

    def ids(body: dict[str, Any]) -> list[str]:
        return [e["id"] for e in body["entries"]]

    assert ids(await logs(client, action="leave.apply")) == [str(denied), str(applied)]
    assert ids(await logs(client, outcome="denied")) == [str(denied)]
    assert ids(await logs(client, actor_id=str(ravi.id))) == [str(denied), str(applied)]
    assert ids(await logs(client, target_id=str(target))) == [str(approved), str(applied)]
    assert ids(await logs(client, action="leave.apply", outcome="allowed")) == [str(applied)]


async def test_date_range_is_whole_days_in_company_time(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel
) -> None:
    early = await add_log(db, AuditAction.LEAVE_APPLY, at=MARCH_1)
    late = await add_log(db, AuditAction.LEAVE_APPLY, at=MARCH_1_LATE)  # 2 March in IST
    later = await add_log(db, AuditAction.LEAVE_APPLY, at=MARCH_5)

    def ids(body: dict[str, Any]) -> set[str]:
        return {e["id"] for e in body["entries"]}

    action = {"action": "leave.apply"}
    assert ids(await logs(client, **action, from_date="2026-03-01", to_date="2026-03-01")) == {
        str(early)
    }
    assert ids(await logs(client, **action, from_date="2026-03-02", to_date="2026-03-05")) == {
        str(late),
        str(later),
    }
    assert ids(await logs(client, **action, to_date="2026-03-04")) == {str(early), str(late)}


async def test_pages_with_the_before_cursor(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel
) -> None:
    made = [await add_log(db, AuditAction.LEAVE_CANCEL) for _ in range(5)]
    newest_first = [str(i) for i in reversed(made)]

    seen: list[str] = []
    page = await logs(client, action="leave.cancel", limit=2)
    while True:
        seen += [e["id"] for e in page["entries"]]
        if page["next"] is None:
            break
        page = await logs(client, action="leave.cancel", limit=2, before=page["next"])

    assert seen == newest_first


@pytest.mark.parametrize(
    "params",
    [
        {"action": "leave.teleport"},
        {"outcome": "maybe"},
        {"actor_id": "not-a-uuid"},
        {"from_date": "yesterday"},
        {"limit": 0},
        {"limit": 101},
    ],
)
async def test_invalid_filters_are_refused(
    client: AsyncClient, hr: EmployeeModel, params: dict[str, Any]
) -> None:
    response = await client.get("/api/admin/audit-logs", params=params)

    assert response.status_code == 422


async def test_from_date_after_to_date_is_refused(client: AsyncClient, hr: EmployeeModel) -> None:
    response = await client.get(
        "/api/admin/audit-logs", params={"from_date": "2026-03-05", "to_date": "2026-03-01"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_input"


# Unanswered queries


async def add_query(
    db: AsyncSession,
    employee: EmployeeModel,
    query: str,
    reason: UnansweredReason = UnansweredReason.NO_RELEVANT_DOCS,
    at: datetime = MARCH_1,
) -> uuid.UUID:
    row = UnansweredQueryModel(employee_id=employee.id, query=query, reason=reason, created_at=at)
    db.add(row)
    await db.commit()
    return row.id


async def queries(client: AsyncClient, **params: Any) -> dict[str, Any]:
    response = await client.get("/api/admin/unanswered-queries", params=params)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_unanswered_queries_without_who_asked(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel, make_employee: MakeEmployee
) -> None:
    ravi = await make_employee(name="Ravi Report")
    await add_query(db, ravi, "Is there a gym allowance?")
    await add_query(db, ravi, "Can I bring my dog?", UnansweredReason.LOW_CONFIDENCE, MARCH_5)

    body = await queries(client)

    assert [(q["query"], q["reason"]) for q in body["queries"]] == [
        ("Can I bring my dog?", "low_confidence"),
        ("Is there a gym allowance?", "no_relevant_docs"),
    ]
    assert body["next"] is None
    assert set(body["queries"][0]) == {"id", "created_at", "query", "reason"}
    assert str(ravi.id) not in str(body)


async def test_unanswered_queries_filters_and_pages(
    client: AsyncClient, db: AsyncSession, hr: EmployeeModel
) -> None:
    gym = await add_query(db, hr, "Is there a gym allowance?")
    dog = await add_query(db, hr, "Can I bring my dog?", UnansweredReason.LOW_CONFIDENCE, MARCH_5)
    car = await add_query(db, hr, "Is there a car lease scheme?", at=MARCH_5)

    def ids(body: dict[str, Any]) -> list[str]:
        return [q["id"] for q in body["queries"]]

    assert ids(await queries(client, reason="low_confidence")) == [str(dog)]
    assert ids(await queries(client, from_date="2026-03-05")) == [str(car), str(dog)]
    first = await queries(client, limit=2)
    assert ids(first) == [str(car), str(dog)]
    second = await queries(client, limit=2, before=first["next"])
    assert ids(second) == [str(gym)]
    assert second["next"] is None


# Access


@pytest.mark.parametrize("role", [Role.EMPLOYEE, Role.MANAGER])
async def test_only_hr_can_read_the_logs(
    client: AsyncClient, make_employee: MakeEmployee, role: Role
) -> None:
    await login(client, await make_employee(role=role))

    assert (await client.get("/api/admin/audit-logs")).status_code == 403
    assert (await client.get("/api/admin/unanswered-queries")).status_code == 403
