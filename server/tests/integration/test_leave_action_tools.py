"""Write tools: they check the rules and save a pending action, and change
nothing else. TODAY is Monday 1 June 2026."""

import json
import uuid
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee, as_user
from tests.integration.test_leave_service import TODAY, add_balance, add_request
from warden.graph.pending_actions import PendingActions
from warden.graph.tools.leave_actions import (
    AWAITING_CONFIRMATION,
    build_leave_action_tools,
    build_team_leave_action_tools,
    cancel_summary,
    leave_summary,
)
from warden.graph.tools.services import ToolServices
from warden.models import (
    ChatThreadModel,
    EmployeeModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
    PendingActionModel,
    PendingActionStatus,
    Role,
)
from warden.services.attendance import AttendanceService
from warden.services.employee import EmployeeService
from warden.services.errors import (
    Forbidden,
    InsufficientBalance,
    InvalidStatus,
    NotFound,
    Overlap,
    ServiceError,
)
from warden.services.holiday import HolidayService
from warden.services.leave import CancelQuote, LeaveQuote, LeaveService
from warden.services.payslip import PayslipService


async def tools_for(db: AsyncSession, employee: EmployeeModel) -> dict[str, BaseTool]:
    """The write tools for `employee` in a new thread of theirs. The
    manager's tools are built for every role, so tests can check the service
    still refuses the wrong role."""
    thread = ChatThreadModel(employee_id=employee.id)
    db.add(thread)
    await db.flush()
    services = ToolServices(
        employees=EmployeeService(db),
        leave=LeaveService(db, today=lambda: TODAY),
        attendance=AttendanceService(db),
        payslips=PayslipService(db),
        holidays=HolidayService(db, today=lambda: TODAY),
        pending_actions=PendingActions(db),
        today=lambda: TODAY,
    )
    user = as_user(employee)
    tools = build_leave_action_tools(user, services, thread.id) + build_team_leave_action_tools(
        user, services, thread.id
    )
    return {t.name: t for t in tools}


async def call(tool: BaseTool, **args: Any) -> ToolMessage:
    """Invoke the tool the way the tools node does, so the artifact comes back."""
    message = await tool.ainvoke({"name": tool.name, "args": args, "id": "1", "type": "tool_call"})
    assert isinstance(message, ToolMessage)
    return message


async def count(db: AsyncSession, model: type[PendingActionModel | LeaveRequestModel]) -> int:
    return await db.scalar(select(func.count()).select_from(model)) or 0


async def test_apply_for_leave_proposes_and_saves_nothing_else(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await add_balance(db, me, allotted=8)
    tools = await tools_for(db, me)

    message = await call(
        tools["apply_for_leave"],
        leave_type="casual",
        start_date="2026-06-08",
        end_date="2026-06-09",
        reason="  Family function  ",
    )

    summary = "Casual leave, Mon 8 Jun to Tue 9 Jun, 2 days. Balance after: 6"
    assert json.loads(str(message.content)) == {
        "status": "awaiting_confirmation",
        "summary": summary,
        "days": 2,
        "available_after": 6,
        "note": AWAITING_CONFIRMATION,
    }
    action = await db.get(PendingActionModel, uuid.UUID(message.artifact["pending_action_id"]))
    assert action is not None
    assert message.artifact == {
        "pending_action_id": str(action.id),
        "action_type": "apply_leave",
        "summary": summary,
    }
    assert action.employee_id == me.id
    assert action.status == PendingActionStatus.PENDING
    assert action.params == {
        "leave_type": "casual",
        "start": "2026-06-08",
        "end": "2026-06-09",
        "reason": "Family function",
    }
    assert await count(db, LeaveRequestModel) == 0


async def test_a_broken_rule_is_raised_and_nothing_is_proposed(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await add_balance(db, me, allotted=1)
    await add_request(db, me, date(2026, 6, 15), date(2026, 6, 15), status=LeaveStatus.APPROVED)
    tools = await tools_for(db, me)

    with pytest.raises(InsufficientBalance):
        await call(
            tools["apply_for_leave"],
            leave_type="casual",
            start_date="2026-06-08",
            end_date="2026-06-09",
        )
    with pytest.raises(Overlap):
        await call(
            tools["apply_for_leave"],
            leave_type="casual",
            start_date="2026-06-15",
            end_date="2026-06-15",
        )
    assert await count(db, PendingActionModel) == 0


@pytest.mark.parametrize(
    ("start", "end", "days", "expected"),
    [
        (date(2026, 6, 8), date(2026, 6, 8), 1, "Sick leave, Mon 8 Jun, 1 day. Balance after: 4"),
        (
            date(2027, 1, 4),
            date(2027, 1, 5),
            2,
            "Sick leave, Mon 4 Jan 2027 to Tue 5 Jan 2027, 2 days. Balance after: 4",
        ),
    ],
)
def test_leave_summary(start: date, end: date, days: int, expected: str) -> None:
    quote = LeaveQuote(
        LeaveType.SICK, start, end, days, available_before=4 + days, available_after=4
    )

    assert leave_summary(quote, TODAY) == expected


async def test_cancel_proposes_and_leaves_the_request_alone(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await add_balance(db, me, allotted=10, used=2)
    request = await add_request(
        db, me, date(2026, 6, 8), date(2026, 6, 9), status=LeaveStatus.APPROVED
    )
    tools = await tools_for(db, me)

    message = await call(tools["cancel_my_leave_request"], request_id=str(request.id))

    summary = "Cancel approved casual leave, Mon 8 Jun to Tue 9 Jun, 2 days. Balance after: 10"
    content = json.loads(str(message.content))
    assert (content["status"], content["summary"]) == ("awaiting_confirmation", summary)
    assert (content["days"], content["available_after"]) == (2, 10)
    action = await db.get(PendingActionModel, uuid.UUID(message.artifact["pending_action_id"]))
    assert action is not None
    assert action.action_type == "cancel_leave"
    assert action.params == {"request_id": str(request.id)}
    await db.refresh(request)
    assert request.status == LeaveStatus.APPROVED


@pytest.mark.parametrize(
    ("status", "start"),
    [
        (LeaveStatus.APPROVED, TODAY),
        (LeaveStatus.REJECTED, date(2026, 6, 8)),
        (LeaveStatus.CANCELLED, date(2026, 6, 8)),
    ],
    ids=["approved-starts-today", "rejected", "cancelled"],
)
async def test_cancel_refuses_requests_that_cant_be_cancelled(
    db: AsyncSession, make_employee: MakeEmployee, status: LeaveStatus, start: date
) -> None:
    me = await make_employee()
    request = await add_request(db, me, start, start, status=status)
    tools = await tools_for(db, me)

    with pytest.raises(InvalidStatus):
        await call(tools["cancel_my_leave_request"], request_id=str(request.id))
    assert await count(db, PendingActionModel) == 0


async def test_cancel_cant_reach_someone_elses_request(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee()
    report = await make_employee(manager_id=manager.id)
    request = await add_request(db, report, date(2026, 6, 8), date(2026, 6, 8))
    tools = await tools_for(db, manager)

    with pytest.raises(NotFound):
        await call(tools["cancel_my_leave_request"], request_id=str(request.id))
    assert await count(db, PendingActionModel) == 0


def test_cancel_summary() -> None:
    request = LeaveRequestModel(
        leave_type=LeaveType.EARNED,
        start_date=date(2026, 6, 8),
        end_date=date(2026, 6, 8),
        days=1,
        status=LeaveStatus.PENDING,
    )

    summary = cancel_summary(CancelQuote(request, available_after=12), TODAY)

    assert summary == "Cancel pending earned leave, Mon 8 Jun, 1 day. Balance after: 12"


async def test_approve_proposes_and_leaves_the_request_and_balance_alone(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee(role=Role.MANAGER)
    report = await make_employee(manager_id=manager.id, name="Asha Rao")
    balance = await add_balance(db, report, allotted=10, used=4)
    request = await add_request(db, report, date(2026, 6, 8), date(2026, 6, 9))
    tools = await tools_for(db, manager)

    message = await call(
        tools["approve_team_leave"], request_id=str(request.id), comment="  Enjoy  "
    )

    summary = (
        "Approve Asha Rao's casual leave, Mon 8 Jun to Tue 9 Jun, 2 days. "
        "Their balance after: 4. Comment: Enjoy"
    )
    content = json.loads(str(message.content))
    assert (content["status"], content["summary"]) == ("awaiting_confirmation", summary)
    assert (content["employee"], content["days"], content["available_after"]) == (
        "Asha Rao",
        2,
        4,
    )
    action = await db.get(PendingActionModel, uuid.UUID(message.artifact["pending_action_id"]))
    assert action is not None
    assert (action.employee_id, action.action_type) == (manager.id, "approve_leave")
    assert action.params == {"request_id": str(request.id), "comment": "Enjoy"}
    await db.refresh(request)
    await db.refresh(balance)
    assert (request.status, balance.used) == (LeaveStatus.PENDING, 4)


async def test_approve_refuses_what_approve_would_refuse(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    head = await make_employee(role=Role.MANAGER)
    manager = await make_employee(role=Role.MANAGER, manager_id=head.id)
    report = await make_employee(manager_id=manager.id)
    outsider = await make_employee(manager_id=head.id)
    await add_balance(db, report, allotted=3, used=2)
    june_8 = date(2026, 6, 8)
    over_balance = await add_request(db, report, june_8, date(2026, 6, 9))
    decided = await add_request(db, report, june_8, june_8, status=LeaveStatus.APPROVED)
    other_team = await add_request(db, outsider, june_8, june_8)
    own = await add_request(db, manager, june_8, june_8)
    tools = await tools_for(db, manager)

    cases: list[tuple[LeaveRequestModel, type[ServiceError]]] = [
        (over_balance, InsufficientBalance),
        (decided, InvalidStatus),
        (other_team, NotFound),
        (own, NotFound),
    ]
    for request, error in cases:
        with pytest.raises(error):
            await call(tools["approve_team_leave"], request_id=str(request.id))
    with pytest.raises(Forbidden):
        report_tools = await tools_for(db, report)
        await call(report_tools["approve_team_leave"], request_id=str(over_balance.id))

    assert await count(db, PendingActionModel) == 0
    used = await db.scalar(
        select(LeaveBalanceModel.used).where(LeaveBalanceModel.employee_id == report.id)
    )
    assert used == 2
