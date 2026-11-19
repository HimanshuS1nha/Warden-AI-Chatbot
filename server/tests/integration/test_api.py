"""The /api/me, /api/team, /api/employees and /api/holidays routes, end to end:
log in for real, call the route, check the response and the database.

Leave dates are relative to a fixed TODAY, Monday 1 June 2026.
"""

import uuid
from collections.abc import Iterator
from datetime import date, timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee
from tests.integration.test_auth import login
from tests.integration.test_leave_service import add_balance, add_request, audit_rows, reload
from warden.core.dependencies import DbSession
from warden.main import app
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    AuditAction,
    DepartmentModel,
    EmployeeModel,
    HolidayModel,
    LeaveStatus,
    PayslipModel,
    Role,
)
from warden.routers.services import get_leave_service
from warden.services.leave import LeaveService

TODAY = date(2026, 6, 1)


def on(offset: int) -> str:
    return (TODAY + timedelta(days=offset)).isoformat()


@pytest.fixture(autouse=True)
def fixed_today(client: AsyncClient) -> Iterator[None]:
    # Depends on `client` so this override is set after, and removed before,
    # the client fixture's own.
    def leave_service(db: DbSession) -> LeaveService:
        return LeaveService(db, today=lambda: TODAY)

    app.dependency_overrides[get_leave_service] = leave_service
    yield
    app.dependency_overrides.pop(get_leave_service, None)


@pytest.fixture
async def org(make_employee: MakeEmployee) -> dict[str, EmployeeModel]:
    hr = await make_employee(role=Role.HR, name="Hana HR")
    manager = await make_employee(role=Role.MANAGER, name="Mona Manager", manager_id=hr.id)
    other_manager = await make_employee(role=Role.MANAGER, name="Omar Other", manager_id=hr.id)
    report = await make_employee(name="Ravi Report", manager_id=manager.id)
    outsider = await make_employee(name="Ola Outsider", manager_id=other_manager.id)
    return {
        "hr": hr,
        "manager": manager,
        "other_manager": other_manager,
        "report": report,
        "outsider": outsider,
    }


# /api/me


async def test_profile_and_manager(client: AsyncClient, org: dict[str, EmployeeModel]) -> None:
    await login(client, org["report"])

    profile = (await client.get("/api/me")).json()
    assert profile["id"] == str(org["report"].id)
    assert profile["department"]["name"] == "Engineering"
    assert "password_hash" not in profile
    manager = (await client.get("/api/me/manager")).json()
    assert manager["id"] == str(org["manager"].id)

    await login(client, org["hr"])
    response = await client.get("/api/me/manager")
    assert (response.status_code, response.json()) == (200, None)


async def test_routes_need_a_session(client: AsyncClient) -> None:
    for path in ("/api/me", "/api/me/payslips", "/api/team", "/api/employees", "/api/holidays"):
        assert (await client.get(path)).status_code == 401, path


async def test_apply_list_and_cancel_leave(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    await add_balance(db, org["report"], allotted=12, used=2)
    await login(client, org["report"])

    applied = await client.post(
        "/api/me/leave-requests",
        json={"leave_type": "casual", "start_date": on(7), "end_date": on(8), "reason": "Trip"},
    )
    assert applied.status_code == 201
    body = applied.json()
    assert (body["status"], body["days"], body["reason"]) == ("pending", 2, "Trip")

    balances = (await client.get("/api/me/leave-balances", params={"year": 2026})).json()
    assert balances == [
        {
            "leave_type": "casual",
            "year": 2026,
            "allotted": 12,
            "used": 2,
            "pending": 2,
            "available": 8,
        }
    ]
    listed = (await client.get("/api/me/leave-requests", params={"status": "pending"})).json()
    assert [r["id"] for r in listed] == [body["id"]]

    cancelled = await client.post(f"/api/me/leave-requests/{body['id']}/cancel")
    assert (cancelled.status_code, cancelled.json()["status"]) == (200, "cancelled")


async def test_rule_violations_are_422_with_a_code(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    await add_balance(db, org["report"], allotted=1)
    await login(client, org["report"])

    async def apply(start: str, end: str, leave_type: str = "casual") -> tuple[int, dict]:
        response = await client.post(
            "/api/me/leave-requests",
            json={"leave_type": leave_type, "start_date": start, "end_date": end},
        )
        return response.status_code, response.json()

    status, body = await apply(on(7), on(8))
    assert status == 422
    assert body == {
        "message": "This needs 2 working days of casual leave, but only 1 are available.",
        "code": "insufficient_balance",
        "details": {"requested": 2, "available": 1},
    }
    status, body = await apply(on(-1), on(0))
    assert (status, body["code"], body["details"]) == (
        422,
        "backdate_not_allowed",
        {"earliest": TODAY.isoformat()},
    )
    status, body = await apply(on(3), on(2))
    assert (status, body["code"]) == (422, "invalid_dates")
    assert "details" not in body


async def test_malformed_bodies_fail_validation_before_the_service(
    client: AsyncClient, org: dict[str, EmployeeModel]
) -> None:
    await login(client, org["report"])

    response = await client.post(
        "/api/me/leave-requests",
        json={"leave_type": "vacation", "start_date": "soon", "reason": "x" * 1001},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Validation failed"
    assert {e["field"] for e in body["errors"]} == {
        "leave_type",
        "start_date",
        "end_date",
        "reason",
    }


async def test_cancelling_someone_elses_request_is_404(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    request = await add_request(db, org["outsider"], TODAY + timedelta(7), TODAY + timedelta(7))
    await login(client, org["report"])

    response = await client.post(f"/api/me/leave-requests/{request.id}/cancel")

    assert response.status_code == 404
    assert response.json() == {"message": "Leave request not found.", "code": "not_found"}
    assert (await reload(db, request)).status == LeaveStatus.PENDING


async def test_attendance_defaults_to_this_month_and_checks_the_month(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    db.add(
        AttendanceModel(
            employee_id=org["report"].id, date=date(2026, 2, 2), status=AttendanceStatus.WFH
        )
    )
    await db.flush()
    await login(client, org["report"])

    rows = (await client.get("/api/me/attendance", params={"month": 2, "year": 2026})).json()
    assert rows == [{"date": "2026-02-02", "status": "wfh", "check_in": None, "check_out": None}]
    assert (await client.get("/api/me/attendance", params={"month": 13})).status_code == 422


async def test_nobody_sees_another_persons_payslips(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    for i, key in enumerate(("report", "manager", "hr")):
        db.add(
            PayslipModel(
                employee_id=org[key].id, year=2026, month=1, net_pay=Decimal(f"{i + 1}000.00")
            )
        )
    await db.flush()

    # HR included: there's no route that takes someone else's ID.
    for key, pay in (("report", "1000.00"), ("manager", "2000.00"), ("hr", "3000.00")):
        await login(client, org[key])
        payslips = (await client.get("/api/me/payslips")).json()
        assert [p["net_pay"] for p in payslips] == [pay], key
    await login(client, org["outsider"])
    assert (await client.get("/api/me/payslips")).json() == []


async def test_holidays_are_for_everyone(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    db.add(HolidayModel(date=date(2026, 8, 15), name="Independence Day"))
    await db.flush()
    await login(client, org["report"])

    response = await client.get("/api/holidays", params={"year": 2026})

    assert response.json() == [{"date": "2026-08-15", "name": "Independence Day"}]


# /api/team


@pytest.mark.parametrize("key", ["report", "hr"])
async def test_only_managers_reach_team_routes(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel], key: str
) -> None:
    request = await add_request(db, org["report"], TODAY + timedelta(7), TODAY + timedelta(7))
    await login(client, org[key])

    for method, path in (
        ("GET", "/api/team"),
        ("GET", "/api/team/leave-requests"),
        ("GET", "/api/team/attendance"),
        ("POST", f"/api/team/leave-requests/{request.id}/approve"),
    ):
        response = await client.request(method, path, json={})
        assert response.status_code == 403, path

    assert len(await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)) == 4
    assert (await reload(db, request)).status == LeaveStatus.PENDING


async def test_manager_sees_and_decides_for_own_team(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    balance = await add_balance(db, org["report"], allotted=10)
    to_approve = await add_request(db, org["report"], TODAY + timedelta(7), TODAY + timedelta(8))
    to_reject = await add_request(db, org["report"], TODAY + timedelta(14), TODAY + timedelta(14))
    await add_request(db, org["outsider"], TODAY + timedelta(7), TODAY + timedelta(7))
    await login(client, org["manager"])

    team = (await client.get("/api/team")).json()
    assert [e["id"] for e in team] == [str(org["report"].id)]
    pending = (await client.get("/api/team/leave-requests", params={"status": "pending"})).json()
    assert {r["id"] for r in pending} == {str(to_approve.id), str(to_reject.id)}
    assert {r["employee"]["name"] for r in pending} == {"Ravi Report"}

    approved = await client.post(
        f"/api/team/leave-requests/{to_approve.id}/approve", json={"comment": "Enjoy"}
    )
    assert (approved.status_code, approved.json()["status"]) == (200, "approved")
    assert (await reload(db, balance)).used == 2

    no_comment = await client.post(
        f"/api/team/leave-requests/{to_reject.id}/reject", json={"comment": ""}
    )
    assert no_comment.status_code == 422
    rejected = await client.post(
        f"/api/team/leave-requests/{to_reject.id}/reject", json={"comment": "Release week"}
    )
    assert (rejected.status_code, rejected.json()["decision_comment"]) == (200, "Release week")

    again = await client.post(f"/api/team/leave-requests/{to_approve.id}/approve", json={})
    assert (again.status_code, again.json()["code"]) == (422, "invalid_status")


async def test_manager_cant_approve_another_teams_request(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    await add_balance(db, org["outsider"])
    request = await add_request(db, org["outsider"], TODAY + timedelta(7), TODAY + timedelta(7))
    await login(client, org["manager"])

    response = await client.post(f"/api/team/leave-requests/{request.id}/approve", json={})

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert (await reload(db, request)).status == LeaveStatus.PENDING


async def test_team_attendance_for_a_date(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    db.add(
        AttendanceModel(
            employee_id=org["report"].id, date=date(2026, 3, 2), status=AttendanceStatus.PRESENT
        )
    )
    await db.flush()
    await login(client, org["manager"])

    rows = (await client.get("/api/team/attendance", params={"date": "2026-03-02"})).json()
    assert [(r["employee"]["name"], r["attendance"]["status"]) for r in rows] == [
        ("Ravi Report", "present")
    ]
    rows = (await client.get("/api/team/attendance", params={"date": "2026-03-03"})).json()
    assert [r["attendance"] for r in rows] == [None]


# /api/employees


async def test_hr_can_search_employees(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    sales = DepartmentModel(name="Sales")
    db.add(sales)
    await db.flush()
    org["outsider"].department_id = sales.id
    await db.flush()
    await login(client, org["hr"])

    by_name = (await client.get("/api/employees", params={"name": "ravi"})).json()
    assert [e["id"] for e in by_name] == [str(org["report"].id)]
    by_department = (await client.get("/api/employees", params={"department": "Sales"})).json()
    assert [(e["name"], e["department"]["name"]) for e in by_department] == [
        ("Ola Outsider", "Sales")
    ]
    assert len(await audit_rows(db, AuditAction.EMPLOYEE_VIEW)) == 2


@pytest.mark.parametrize("key", ["manager", "report"])
async def test_only_hr_can_search(
    client: AsyncClient, org: dict[str, EmployeeModel], key: str
) -> None:
    await login(client, org[key])

    assert (await client.get("/api/employees", params={"name": "a"})).status_code == 403


async def test_viewing_an_employee_is_scoped(
    client: AsyncClient, org: dict[str, EmployeeModel]
) -> None:
    await login(client, org["manager"])
    assert (await client.get(f"/api/employees/{org['report'].id}")).status_code == 200
    assert (await client.get(f"/api/employees/{org['outsider'].id}")).status_code == 404

    await login(client, org["hr"])
    assert (await client.get(f"/api/employees/{org['outsider'].id}")).status_code == 200

    await login(client, org["report"])
    assert (await client.get(f"/api/employees/{org['manager'].id}")).status_code == 403


async def test_hr_reads_an_employees_leave_summary(
    client: AsyncClient, db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    await add_balance(db, org["outsider"], allotted=12, used=1)
    request = await add_request(
        db, org["outsider"], TODAY + timedelta(days=7), TODAY + timedelta(days=7)
    )
    await login(client, org["hr"])

    response = await client.get(f"/api/employees/{org['outsider'].id}/leave")

    assert response.status_code == 200
    body = response.json()
    assert (body["employee"]["name"], body["year"]) == ("Ola Outsider", 2026)
    assert [(b["leave_type"], b["available"]) for b in body["balances"]] == [("casual", 10)]
    assert [r["id"] for r in body["recent_requests"]] == [str(request.id)]
    assert len(await audit_rows(db, AuditAction.EMPLOYEE_VIEW)) == 1
    missing = await client.get(f"/api/employees/{uuid.uuid4()}/leave")
    assert missing.status_code == 404


async def test_managers_cant_read_leave_summaries_even_for_their_reports(
    client: AsyncClient, org: dict[str, EmployeeModel]
) -> None:
    await login(client, org["manager"])

    response = await client.get(f"/api/employees/{org['report'].id}/leave")

    assert response.status_code == 403
