"""Agent tools against the real services and database."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_utils.compat import uuid7

from tests.agent_fakes import PROMPT, ScriptedChatModel, call
from tests.factories import MakeEmployee, as_user
from tests.integration.test_leave_service import TODAY, add_balance, add_request
from warden.graph.graph import build_graph
from warden.graph.tools.hr import SEARCH_LIMIT, build_hr_tools
from warden.graph.tools.manager import NOT_RECORDED, build_manager_tools
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    DepartmentModel,
    EmployeeModel,
    HolidayModel,
    LeaveStatus,
    LeaveType,
    PayslipModel,
    Role,
)
from warden.services.attendance import AttendanceService
from warden.services.employee import EmployeeService
from warden.services.errors import Forbidden, InvalidInput, NotFound
from warden.services.holiday import HolidayService
from warden.services.leave import LeaveService
from warden.services.payslip import PayslipService


def tools_for(db: AsyncSession, employee: EmployeeModel) -> dict[str, BaseTool]:
    """Every tool for `employee`, with TODAY (1 June 2026) as the date. The
    manager and HR tools are built for every role here, so tests can check
    the services still refuse the wrong role."""
    services = ToolServices(
        employees=EmployeeService(db),
        leave=LeaveService(db, today=lambda: TODAY),
        attendance=AttendanceService(db),
        payslips=PayslipService(db),
        holidays=HolidayService(db, today=lambda: TODAY),
        today=lambda: TODAY,
    )
    user = as_user(employee)
    tools = (
        build_self_service_tools(user, services)
        + build_manager_tools(user, services)
        + build_hr_tools(user, services)
    )
    return {t.name: t for t in tools}


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


async def test_get_my_attendance_for_a_range_of_months(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    db.add_all(
        [
            AttendanceModel(
                employee_id=me.id, date=date(2026, 1, 5), status=AttendanceStatus.PRESENT
            ),
            AttendanceModel(
                employee_id=me.id, date=date(2026, 1, 6), status=AttendanceStatus.ABSENT
            ),
            AttendanceModel(employee_id=me.id, date=date(2026, 3, 2), status=AttendanceStatus.WFH),
            AttendanceModel(
                employee_id=me.id, date=date(2026, 3, 3), status=AttendanceStatus.PRESENT
            ),
            # Outside the range.
            AttendanceModel(
                employee_id=me.id, date=date(2026, 4, 1), status=AttendanceStatus.ABSENT
            ),
        ]
    )
    await db.flush()
    tools = tools_for(db, me)

    result = await run(tools["get_my_attendance"], month=1, to_month=3)

    assert result == {
        "year": 2026,
        "from_month": 1,
        "to_month": 3,
        "months": [
            {"month": 1, "days_by_status": {"present": 1, "absent": 1}},
            {"month": 2, "days_by_status": {}},
            {"month": 3, "days_by_status": {"wfh": 1, "present": 1}},
        ],
        "days_not_present": [
            {"date": "2026-01-06", "status": "absent"},
            {"date": "2026-03-02", "status": "wfh"},
        ],
    }
    with pytest.raises(InvalidInput, match="can't be before"):
        await tools["get_my_attendance"].ainvoke({"month": 3, "to_month": 1})


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


async def test_list_my_team_is_my_active_reports_only(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    boss = await make_employee(role=Role.MANAGER)
    other_boss = await make_employee(role=Role.MANAGER)
    ana = await make_employee(manager_id=boss.id, name="Ana", job_title="Engineer")
    await make_employee(manager_id=boss.id, name="Ben", is_active=False)
    await make_employee(manager_id=other_boss.id, name="Cara")
    loner = await make_employee(role=Role.MANAGER)

    result = await run(tools_for(db, boss)["list_my_team"])
    empty = await run(tools_for(db, loner)["list_my_team"])

    assert result == {
        "team": [
            {
                "employee_code": ana.employee_code,
                "name": "Ana",
                "email": ana.email,
                "job_title": "Engineer",
            }
        ]
    }
    assert empty["team"] == [] and "note" in empty


async def test_get_team_leave_requests_covers_my_reports_and_filters_by_status(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    boss = await make_employee(role=Role.MANAGER)
    other_boss = await make_employee(role=Role.MANAGER)
    ana = await make_employee(manager_id=boss.id, name="Ana")
    outsider = await make_employee(manager_id=other_boss.id)
    pending = await add_request(db, ana, date(2026, 6, 8), date(2026, 6, 9), reason="Wedding")
    await add_request(db, ana, date(2026, 5, 4), date(2026, 5, 4), status=LeaveStatus.APPROVED)
    await add_request(db, outsider, date(2026, 6, 10), date(2026, 6, 10))
    tools = tools_for(db, boss)

    everything = await run(tools["get_team_leave_requests"])
    only_pending = await run(tools["get_team_leave_requests"], status="pending")
    rejected = await run(tools["get_team_leave_requests"], status="rejected")

    assert len(everything["requests"]) == 2
    [row] = only_pending["requests"]
    assert row["id"] == str(pending.id)
    assert row["employee"] == {"name": "Ana", "employee_code": ana.employee_code}
    assert (row["days"], row["reason"], row["status"]) == (2, "Wedding", "pending")
    assert "approver_id" not in row
    assert row["created_at"].endswith("+05:30")
    assert rejected["requests"] == [] and "note" in rejected


async def test_get_team_attendance_defaults_to_today_and_marks_missing_days(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    boss = await make_employee(role=Role.MANAGER)
    ana = await make_employee(manager_id=boss.id, name="Ana")
    ben = await make_employee(manager_id=boss.id, name="Ben")
    outsider = await make_employee()
    db.add_all(
        [
            AttendanceModel(
                employee_id=ana.id,
                date=TODAY,
                status=AttendanceStatus.PRESENT,
                check_in=datetime(2026, 6, 1, 3, 30, tzinfo=UTC),
            ),
            AttendanceModel(
                employee_id=ben.id, date=date(2026, 5, 29), status=AttendanceStatus.WFH
            ),
            AttendanceModel(employee_id=outsider.id, date=TODAY, status=AttendanceStatus.ABSENT),
        ]
    )
    await db.flush()
    tools = tools_for(db, boss)

    today = await run(tools["get_team_attendance"])
    friday = await run(tools["get_team_attendance"], date="2026-05-29")

    assert today["date"] == "2026-06-01"
    assert today["people_by_status"] == {"present": 1, NOT_RECORDED: 1}
    assert today["members"] == [
        {
            "name": "Ana",
            "employee_code": ana.employee_code,
            "status": "present",
            "check_in": "2026-06-01T09:00+05:30",
            "check_out": None,
        },
        {
            "name": "Ben",
            "employee_code": ben.employee_code,
            "status": NOT_RECORDED,
            "check_in": None,
            "check_out": None,
        },
    ]
    assert friday["people_by_status"] == {NOT_RECORDED: 1, "wfh": 1}


@pytest.mark.parametrize("name", ["list_my_team", "get_team_leave_requests", "get_team_attendance"])
@pytest.mark.parametrize("role", [Role.EMPLOYEE, Role.HR])
async def test_team_tools_refuse_non_managers_even_if_given_to_them(
    db: AsyncSession, make_employee: MakeEmployee, name: str, role: Role
) -> None:
    me = await make_employee(role=role)

    with pytest.raises(Forbidden):
        await tools_for(db, me)[name].ainvoke({})


async def test_search_employees_returns_ids_for_the_next_tools(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    hr = await make_employee(role=Role.HR)
    sales = DepartmentModel(name="Sales")
    db.add(sales)
    await db.flush()
    ravi = await make_employee(name="Ravi Kumar", department_id=sales.id, job_title="Rep")
    await make_employee(name="Ravi Left", is_active=False)
    await make_employee(name="Someone Else")
    tools = tools_for(db, hr)

    by_name = await run(tools["search_employees"], name="RAVI")
    by_department = await run(tools["search_employees"], department="sales")
    nobody = await run(tools["search_employees"], name="zzz")

    expected = {
        "id": str(ravi.id),
        "employee_code": ravi.employee_code,
        "name": "Ravi Kumar",
        "email": ravi.email,
        "job_title": "Rep",
        "role": "employee",
        "department": "Sales",
    }
    assert by_name == {"employees": [expected]}
    assert by_department == {"employees": [expected]}
    assert nobody["employees"] == [] and "note" in nobody


async def test_search_employees_says_when_results_are_cut_off(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    hr = await make_employee(role=Role.HR, name="HR Person")
    for i in range(SEARCH_LIMIT + 1):
        await make_employee(name=f"Clone {i:02}")

    result = await run(tools_for(db, hr)["search_employees"], name="clone")

    assert len(result["employees"]) == SEARCH_LIMIT
    assert "note" in result


async def test_get_employee_profile_includes_the_manager_id(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    hr = await make_employee(role=Role.HR)
    boss = await make_employee(role=Role.MANAGER)
    ravi = await make_employee(manager_id=boss.id, is_active=False)
    tools = tools_for(db, hr)

    profile = await run(tools["get_employee_profile"], employee_id=str(ravi.id))

    assert (profile["id"], profile["manager_id"]) == (str(ravi.id), str(boss.id))
    assert profile["is_active"] is False
    assert isinstance(profile["department"], str) and "department_id" not in profile
    with pytest.raises(NotFound):
        await tools["get_employee_profile"].ainvoke({"employee_id": str(uuid7())})


async def test_get_employee_leave_summary(db: AsyncSession, make_employee: MakeEmployee) -> None:
    hr = await make_employee(role=Role.HR)
    ravi = await make_employee(name="Ravi")
    await add_balance(db, ravi, allotted=12, used=3)
    request = await add_request(db, ravi, date(2026, 6, 8), date(2026, 6, 9), reason="Trip")

    result = await run(tools_for(db, hr)["get_employee_leave_summary"], employee_id=str(ravi.id))

    assert result["employee"] == {"name": "Ravi", "id": str(ravi.id)}
    assert result["year"] == 2026
    assert result["balances"] == [
        {"leave_type": "casual", "allotted": 12, "used": 3, "pending": 2, "available": 7}
    ]
    [row] = result["recent_requests"]
    assert (row["id"], row["reason"], row["status"]) == (str(request.id), "Trip", "pending")
    assert "approver_id" not in row and row["created_at"].endswith("+05:30")


@pytest.mark.parametrize(
    "name", ["search_employees", "get_employee_profile", "get_employee_leave_summary"]
)
@pytest.mark.parametrize("role", [Role.EMPLOYEE, Role.MANAGER])
async def test_hr_tools_refuse_everyone_else_even_if_given_to_them(
    db: AsyncSession, make_employee: MakeEmployee, name: str, role: Role
) -> None:
    me = await make_employee(role=role)
    stranger = await make_employee()
    args = {} if name == "search_employees" else {"employee_id": str(stranger.id)}

    # A manager may view their own reports' profiles (the REST route allows
    # it too), so for them a stranger's profile is "not found".
    with pytest.raises((Forbidden, NotFound)):
        await tools_for(db, me)[name].ainvoke(args)


async def test_parallel_tool_calls_share_the_request_session_safely(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    """Gemini often asks for several tools in one reply. They all use the
    request's session, which allows one operation at a time."""
    me = await make_employee()
    names = ["get_my_profile", "get_my_leave_balance", "get_my_payslips", "get_my_attendance"]
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[call(n, f"call_{n}") for n in names]),
                AIMessage(content="Done."),
            ]
        )
    )
    graph = build_graph(model, list(tools_for(db, me).values()), PROMPT)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Everything about me")]})

    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_results] == names
    assert all(m.status == "success" for m in tool_results)
