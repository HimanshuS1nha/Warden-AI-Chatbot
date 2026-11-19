"""Agent tools against the real services and database."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee, as_user
from tests.integration.test_leave_service import TODAY, add_balance, add_request
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    EmployeeModel,
    HolidayModel,
    LeaveStatus,
    LeaveType,
    PayslipModel,
    Role,
)
from warden.services.attendance import AttendanceService
from warden.services.employee import EmployeeService
from warden.services.holiday import HolidayService
from warden.services.leave import LeaveService
from warden.services.payslip import PayslipService


def tools_for(db: AsyncSession, employee: EmployeeModel) -> dict[str, BaseTool]:
    """The self-service tools for `employee`, with TODAY (1 June 2026) as the date."""
    services = ToolServices(
        employees=EmployeeService(db),
        leave=LeaveService(db, today=lambda: TODAY),
        attendance=AttendanceService(db),
        payslips=PayslipService(db),
        holidays=HolidayService(db, today=lambda: TODAY),
        today=lambda: TODAY,
    )
    return {t.name: t for t in build_self_service_tools(as_user(employee), services)}


async def run(tool: BaseTool, **args: Any) -> Any:
    return json.loads(await tool.ainvoke(args))


async def test_get_my_profile_returns_the_session_users_profile(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await make_employee()  # Someone else, who must not show up.

    profile = await run(tools_for(db, me)["get_my_profile"])

    assert profile["employee_code"] == me.employee_code
    assert profile["email"] == me.email
    assert isinstance(profile["department"], str)
    assert not {"id", "department_id", "manager_id"} & profile.keys()


async def test_get_my_manager(db: AsyncSession, make_employee: MakeEmployee) -> None:
    boss = await make_employee(role=Role.MANAGER, job_title="Engineering Manager")
    me = await make_employee(manager_id=boss.id)

    result = await run(tools_for(db, me)["get_my_manager"])
    top = await run(tools_for(db, boss)["get_my_manager"])

    assert result["manager"] == {
        "employee_code": boss.employee_code,
        "name": boss.name,
        "email": boss.email,
        "job_title": "Engineering Manager",
    }
    assert top["manager"] is None


async def test_get_my_leave_balance_counts_pending_days(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    other = await make_employee()
    await add_balance(db, me, allotted=12, used=3)
    await add_balance(db, me, allotted=8, leave_type=LeaveType.SICK)
    await add_balance(db, other, allotted=30)
    await add_request(db, me, date(2026, 6, 8), date(2026, 6, 9))  # 2 pending days

    result = await run(tools_for(db, me)["get_my_leave_balance"])

    assert result["year"] == 2026
    by_type = {b["leave_type"]: b for b in result["balances"]}
    assert by_type.keys() == {"casual", "sick"}
    assert by_type["casual"] == {
        "leave_type": "casual",
        "allotted": 12,
        "used": 3,
        "pending": 2,
        "available": 7,
    }


async def test_get_my_leave_balance_for_a_year_with_none(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()

    result = await run(tools_for(db, me)["get_my_leave_balance"], year=2030)

    assert result["year"] == 2030
    assert result["balances"] == []
    assert "note" in result


async def test_get_my_leave_history_is_mine_only_and_filters_by_status(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    other = await make_employee()
    pending = await add_request(db, me, date(2026, 6, 8), date(2026, 6, 8), reason="Errand")
    await add_request(db, me, date(2026, 5, 4), date(2026, 5, 5), status=LeaveStatus.APPROVED)
    await add_request(db, other, date(2026, 6, 10), date(2026, 6, 10))
    tools = tools_for(db, me)

    everything = await run(tools["get_my_leave_history"])
    only_pending = await run(tools["get_my_leave_history"], status="pending")

    assert len(everything["requests"]) == 2
    assert [r["id"] for r in only_pending["requests"]] == [str(pending.id)]
    assert only_pending["requests"][0]["reason"] == "Errand"
    assert "approver_id" not in only_pending["requests"][0]
    assert only_pending["requests"][0]["created_at"].endswith("+05:30")
    assert only_pending["requests"][0]["decided_at"] is None


async def test_get_my_attendance_defaults_to_this_month(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    other = await make_employee()
    check_in = datetime(2026, 6, 1, 3, 30, tzinfo=UTC)
    db.add_all(
        [
            AttendanceModel(
                employee_id=me.id,
                date=date(2026, 6, 1),
                status=AttendanceStatus.PRESENT,
                check_in=check_in,
            ),
            AttendanceModel(employee_id=me.id, date=date(2026, 6, 2), status=AttendanceStatus.WFH),
            AttendanceModel(employee_id=me.id, date=date(2026, 6, 3), status=AttendanceStatus.WFH),
            AttendanceModel(
                employee_id=me.id, date=date(2026, 5, 29), status=AttendanceStatus.ABSENT
            ),
            AttendanceModel(
                employee_id=other.id, date=date(2026, 6, 1), status=AttendanceStatus.ABSENT
            ),
        ]
    )
    await db.flush()
    tools = tools_for(db, me)

    june = await run(tools["get_my_attendance"])
    may = await run(tools["get_my_attendance"], month=5)

    assert (june["month"], june["year"]) == (6, 2026)
    assert june["days_by_status"] == {"present": 1, "wfh": 2}
    assert [d["date"] for d in june["days"]] == ["2026-06-01", "2026-06-02", "2026-06-03"]
    # 03:30 UTC is 09:00 in the company's timezone (Asia/Kolkata).
    assert june["days"][0]["check_in"] == "2026-06-01T09:00+05:30"
    assert june["days"][1]["check_in"] is None
    assert may["days_by_status"] == {"absent": 1}


async def test_get_my_attendance_for_a_month_with_nothing_recorded(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()

    result = await run(tools_for(db, me)["get_my_attendance"], month=1, year=2025)

    assert (result["month"], result["year"], result["days"]) == (1, 2025, [])
    assert "note" in result


async def test_get_my_payslips_newest_first_and_mine_only(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    other = await make_employee()
    db.add_all(
        [
            PayslipModel(employee_id=me.id, year=2025, month=12, net_pay=Decimal("81000.00")),
            PayslipModel(employee_id=me.id, year=2026, month=1, net_pay=Decimal("85500.50")),
            PayslipModel(employee_id=other.id, year=2026, month=1, net_pay=Decimal("99999.00")),
        ]
    )
    await db.flush()
    tools = tools_for(db, me)

    every_year = await run(tools["get_my_payslips"])
    in_2025 = await run(tools["get_my_payslips"], year=2025)
    in_2024 = await run(tools["get_my_payslips"], year=2024)

    assert every_year["payslips"] == [
        {"year": 2026, "month": 1, "net_pay": "85500.50"},
        {"year": 2025, "month": 12, "net_pay": "81000.00"},
    ]
    assert [p["month"] for p in in_2025["payslips"]] == [12]
    assert in_2024["payslips"] == [] and "note" in in_2024


async def test_get_company_holidays_for_this_year_and_another(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    db.add_all(
        [
            HolidayModel(date=date(2026, 8, 15), name="Independence Day"),
            HolidayModel(date=date(2026, 1, 26), name="Republic Day"),
            HolidayModel(date=date(2027, 1, 26), name="Republic Day"),
        ]
    )
    await db.flush()
    tools = tools_for(db, me)

    this_year = await run(tools["get_company_holidays"])
    next_year = await run(tools["get_company_holidays"], year=2027)

    assert this_year == {
        "year": 2026,
        "holidays": [
            {"date": "2026-01-26", "name": "Republic Day"},
            {"date": "2026-08-15", "name": "Independence Day"},
        ],
    }
    assert [h["date"] for h in next_year["holidays"]] == ["2027-01-26"]
