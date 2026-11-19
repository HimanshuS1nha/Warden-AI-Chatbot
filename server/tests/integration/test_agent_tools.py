"""Agent tools against the real services and database."""

import json
from datetime import date
from typing import Any

from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee, as_user
from tests.integration.test_leave_service import TODAY, add_balance, add_request
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import EmployeeModel, LeaveStatus, LeaveType, Role
from warden.services.employee import EmployeeService
from warden.services.leave import LeaveService


def tools_for(db: AsyncSession, employee: EmployeeModel) -> dict[str, BaseTool]:
    services = ToolServices(
        employees=EmployeeService(db), leave=LeaveService(db, today=lambda: TODAY)
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
