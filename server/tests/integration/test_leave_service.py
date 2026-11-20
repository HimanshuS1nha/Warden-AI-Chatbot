"""Every rule in plan.md's "Business rules for leave", against the database.

TODAY is Monday 1 June 2026. Wednesday 3 June is made a holiday where a
test needs one.
"""

import asyncio
import uuid
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from tests.factories import MakeEmployee, as_user
from tests.integration.conftest import wait_for_lock_waiters
from warden.models import (
    AuditAction,
    AuditLogModel,
    AuditOutcome,
    EmployeeModel,
    HolidayModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
    Role,
)
from warden.services.errors import (
    BackdateNotAllowed,
    Forbidden,
    InsufficientBalance,
    InvalidDates,
    InvalidInput,
    InvalidStatus,
    NotFound,
    Overlap,
)
from warden.services.leave import LeaveService

TODAY = date(2026, 6, 1)
HOLIDAY = date(2026, 6, 3)


def days(n: int) -> timedelta:
    return timedelta(days=n)


def service(db: AsyncSession) -> LeaveService:
    return LeaveService(db, today=lambda: TODAY)


async def add_balance(
    db: AsyncSession,
    employee: EmployeeModel,
    allotted: int = 10,
    used: int = 0,
    leave_type: LeaveType = LeaveType.CASUAL,
    year: int = 2026,
) -> LeaveBalanceModel:
    balance = LeaveBalanceModel(
        employee_id=employee.id, year=year, leave_type=leave_type, allotted=allotted, used=used
    )
    db.add(balance)
    await db.flush()
    return balance


async def add_request(
    db: AsyncSession, employee: EmployeeModel, start: date, end: date, **overrides: Any
) -> LeaveRequestModel:
    fields: dict[str, Any] = {
        "employee_id": employee.id,
        "leave_type": LeaveType.CASUAL,
        "start_date": start,
        "end_date": end,
        "days": (end - start).days + 1,
        "status": LeaveStatus.PENDING,
    } | overrides
    request = LeaveRequestModel(**fields)
    db.add(request)
    await db.flush()
    return request


async def audit_rows(db: AsyncSession, action: AuditAction) -> Sequence[AuditLogModel]:
    result = await db.scalars(
        select(AuditLogModel)
        .where(AuditLogModel.action == action)
        .order_by(AuditLogModel.id)
        .execution_options(populate_existing=True)
    )
    return result.all()


async def reload(db: AsyncSession, obj: Any) -> Any:
    await db.refresh(obj)
    return obj


@pytest.fixture
async def holiday(db: AsyncSession) -> None:
    db.add(HolidayModel(date=HOLIDAY, name="Test Holiday"))
    await db.flush()


@pytest.fixture
async def team(make_employee: MakeEmployee) -> tuple[EmployeeModel, EmployeeModel]:
    manager = await make_employee(role=Role.MANAGER, name="Manager")
    employee = await make_employee(manager_id=manager.id, name="Report")
    return manager, employee


# Validation: days, dates, overlap, balance


@pytest.mark.usefixtures("holiday")
async def test_quote_counts_working_days_only(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, allotted=12, used=2)

    # Mon 1 to Fri 12 June: 10 weekdays, minus the Wednesday holiday.
    quote = await service(db).validate_application(
        as_user(employee), LeaveType.CASUAL, TODAY, TODAY + days(11)
    )

    assert (quote.days, quote.available_before, quote.available_after) == (9, 10, 1)


@pytest.mark.usefixtures("holiday")
async def test_a_range_with_no_working_days_is_refused(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee)
    leave = service(db)

    with pytest.raises(InvalidDates, match="no working days"):
        await leave.validate_application(
            as_user(employee), LeaveType.CASUAL, date(2026, 6, 6), date(2026, 6, 7)
        )
    with pytest.raises(InvalidDates, match="no working days"):
        await leave.validate_application(as_user(employee), LeaveType.CASUAL, HOLIDAY, HOLIDAY)


async def test_date_rules_apply_before_anything_else(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee)
    leave = service(db)
    user = as_user(employee)

    with pytest.raises(InvalidDates):
        await leave.validate_application(user, LeaveType.CASUAL, TODAY + days(2), TODAY)
    with pytest.raises(BackdateNotAllowed):
        await leave.validate_application(user, LeaveType.CASUAL, TODAY - days(1), TODAY)
    with pytest.raises(BackdateNotAllowed):
        await leave.validate_application(user, LeaveType.SICK, TODAY - days(8), TODAY)


async def test_sick_leave_can_be_backdated_by_7_days(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, leave_type=LeaveType.SICK)

    # Mon 25 May to Fri 29 May.
    quote = await service(db).validate_application(
        as_user(employee), LeaveType.SICK, TODAY - days(7), TODAY - days(3)
    )

    assert quote.days == 5


@pytest.mark.parametrize("status", [LeaveStatus.PENDING, LeaveStatus.APPROVED])
async def test_overlapping_a_pending_or_approved_request_is_refused(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel], status: LeaveStatus
) -> None:
    _, employee = team
    await add_balance(db, employee)
    # Any type blocks: you can't be on sick and casual leave the same day.
    existing = await add_request(
        db, employee, TODAY + days(7), TODAY + days(9), status=status, leave_type=LeaveType.SICK
    )

    with pytest.raises(Overlap) as exc_info:
        await service(db).validate_application(
            as_user(employee), LeaveType.CASUAL, TODAY + days(9), TODAY + days(10)
        )
    assert exc_info.value.details == {"request_id": existing.id}


@pytest.mark.parametrize("status", [LeaveStatus.REJECTED, LeaveStatus.CANCELLED])
async def test_rejected_and_cancelled_requests_dont_block(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel], status: LeaveStatus
) -> None:
    _, employee = team
    await add_balance(db, employee)
    await add_request(db, employee, TODAY + days(7), TODAY + days(9), status=status)

    quote = await service(db).validate_application(
        as_user(employee), LeaveType.CASUAL, TODAY + days(7), TODAY + days(9)
    )

    assert quote.days == 3


async def test_available_is_allotted_minus_used_minus_pending(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, allotted=10, used=3)
    # 4 pending days (Mon to Thu). Approved ones are already in `used`, and
    # other types or years don't count.
    await add_request(db, employee, TODAY + days(14), TODAY + days(17), days=4)
    await add_request(db, employee, TODAY + days(21), TODAY + days(21), leave_type=LeaveType.SICK)
    await add_request(db, employee, date(2027, 1, 4), date(2027, 1, 4))
    leave = service(db)
    user = as_user(employee)

    # Mon 8 to Thu 11 June is 4 days; only 3 are available.
    with pytest.raises(InsufficientBalance) as exc_info:
        await leave.validate_application(user, LeaveType.CASUAL, TODAY + days(7), TODAY + days(10))
    assert exc_info.value.details == {"requested": 4, "available": 3}

    quote = await leave.validate_application(
        user, LeaveType.CASUAL, TODAY + days(7), TODAY + days(9)
    )
    assert (quote.available_before, quote.available_after) == (3, 0)


async def test_no_balance_row_means_nothing_available(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team

    with pytest.raises(InsufficientBalance) as exc_info:
        await service(db).validate_application(as_user(employee), LeaveType.EARNED, TODAY, TODAY)
    assert exc_info.value.details == {"requested": 1, "available": 0}


# Apply


async def test_apply_saves_a_pending_request_and_audits_it(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee)

    request = await service(db).apply(
        as_user(employee), LeaveType.CASUAL, TODAY + days(7), TODAY + days(8), "  Family visit "
    )

    request = await reload(db, request)
    assert request.status == LeaveStatus.PENDING
    assert (request.days, request.reason, request.approver_id) == (2, "Family visit", None)
    [row] = await audit_rows(db, AuditAction.LEAVE_APPLY)
    assert row.outcome == AuditOutcome.ALLOWED
    assert (row.actor_id, row.target_id) == (employee.id, request.id)
    assert row.details == {
        "leave_type": "casual",
        "start": "2026-06-08",
        "end": "2026-06-09",
        "days": 2,
    }


async def test_apply_refusal_is_audited_and_saves_nothing(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, allotted=1)

    with pytest.raises(InsufficientBalance):
        await service(db).apply(as_user(employee), LeaveType.CASUAL, TODAY, TODAY + days(1))

    assert await service(db).list_my_requests(as_user(employee)) == []
    [row] = await audit_rows(db, AuditAction.LEAVE_APPLY)
    assert row.outcome == AuditOutcome.DENIED
    assert row.target_id is None
    assert row.details == {
        "leave_type": "casual",
        "start": "2026-06-01",
        "end": "2026-06-02",
        "reason": "insufficient_balance",
    }


async def test_apply_refuses_an_overlong_reason(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee)

    with pytest.raises(InvalidInput, match="at most 1000"):
        await service(db).apply(as_user(employee), LeaveType.CASUAL, TODAY, TODAY, "x" * 1001)


async def test_second_application_sees_the_first_as_pending(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, allotted=3)
    leave = service(db)
    user = as_user(employee)

    await leave.apply(user, LeaveType.CASUAL, TODAY + days(7), TODAY + days(8))

    with pytest.raises(Overlap):
        await leave.apply(user, LeaveType.CASUAL, TODAY + days(8), TODAY + days(8))
    with pytest.raises(InsufficientBalance):
        await leave.apply(user, LeaveType.CASUAL, TODAY + days(14), TODAY + days(15))


# Cancel


async def test_cancelling_a_pending_request_leaves_the_balance_alone(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    balance = await add_balance(db, employee, used=2)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(8))

    cancelled = await service(db).cancel(as_user(employee), request.id)

    assert cancelled.status == LeaveStatus.CANCELLED
    assert (await reload(db, balance)).used == 2
    [row] = await audit_rows(db, AuditAction.LEAVE_CANCEL)
    assert row.outcome == AuditOutcome.ALLOWED
    assert row.details["previous_status"] == "pending"
    assert "balance_before" not in row.details


async def test_cancelling_approved_future_leave_gives_the_days_back(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    balance = await add_balance(db, employee, allotted=10, used=5)
    request = await add_request(
        db, employee, TODAY + days(1), TODAY + days(2), status=LeaveStatus.APPROVED
    )

    await service(db).cancel(as_user(employee), request.id)

    assert (await reload(db, balance)).used == 3
    [row] = await audit_rows(db, AuditAction.LEAVE_CANCEL)
    assert row.details == {
        "leave_type": "casual",
        "start": "2026-06-02",
        "end": "2026-06-03",
        "days": 2,
        "balance_before": 5,
        "balance_after": 7,
        "previous_status": "approved",
    }


@pytest.mark.parametrize(
    ("status", "start_offset", "message"),
    [
        (LeaveStatus.APPROVED, 0, "before it starts"),
        (LeaveStatus.APPROVED, -3, "before it starts"),
        (LeaveStatus.REJECTED, 7, "already rejected"),
        (LeaveStatus.CANCELLED, 7, "already cancelled"),
    ],
    ids=["approved-starts-today", "approved-started", "rejected", "cancelled"],
)
async def test_cancel_refuses_requests_that_cant_be_cancelled(
    db: AsyncSession,
    team: tuple[EmployeeModel, EmployeeModel],
    status: LeaveStatus,
    start_offset: int,
    message: str,
) -> None:
    _, employee = team
    balance = await add_balance(db, employee, used=5)
    start = TODAY + days(start_offset)
    request = await add_request(db, employee, start, start, status=status)

    with pytest.raises(InvalidStatus, match=message):
        await service(db).cancel(as_user(employee), request.id)

    assert (await reload(db, request)).status == status
    assert (await reload(db, balance)).used == 5
    [row] = await audit_rows(db, AuditAction.LEAVE_CANCEL)
    assert (row.outcome, row.details["reason"]) == (AuditOutcome.DENIED, "invalid_status")


async def test_nobody_can_cancel_someone_elses_request(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    manager, employee = team
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(7))

    # Not even their manager: cancelling is the employee's own action.
    with pytest.raises(NotFound):
        await service(db).cancel(as_user(manager), request.id)

    assert (await reload(db, request)).status == LeaveStatus.PENDING
    [row] = await audit_rows(db, AuditAction.LEAVE_CANCEL)
    assert (row.outcome, row.actor_id, row.target_id) == (
        AuditOutcome.DENIED,
        manager.id,
        request.id,
    )
    assert row.details == {"reason": "not_found"}


# Approve and reject


async def test_manager_approves_a_report_and_the_days_are_used(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    manager, employee = team
    balance = await add_balance(db, employee, allotted=10, used=1)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(9))

    approved = await service(db).approve(as_user(manager), request.id, "  Enjoy  ")

    approved = await reload(db, approved)
    assert approved.status == LeaveStatus.APPROVED
    assert (approved.approver_id, approved.decision_comment) == (manager.id, "Enjoy")
    assert approved.decided_at is not None
    assert (await reload(db, balance)).used == 4
    [row] = await audit_rows(db, AuditAction.LEAVE_APPROVE)
    assert (row.outcome, row.actor_id, row.target_id) == (
        AuditOutcome.ALLOWED,
        manager.id,
        request.id,
    )
    assert row.details == {
        "leave_type": "casual",
        "start": "2026-06-08",
        "end": "2026-06-10",
        "days": 3,
        "balance_before": 9,
        "balance_after": 6,
    }


async def test_manager_rejects_with_a_comment_and_the_balance_is_untouched(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    manager, employee = team
    balance = await add_balance(db, employee, used=1)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(9))
    leave = service(db)

    with pytest.raises(InvalidInput, match="Say why"):
        await leave.reject(as_user(manager), request.id, "   ")
    rejected = await leave.reject(as_user(manager), request.id, "Release week")

    rejected = await reload(db, rejected)
    assert (rejected.status, rejected.decision_comment) == (LeaveStatus.REJECTED, "Release week")
    assert (await reload(db, balance)).used == 1
    denied, allowed = await audit_rows(db, AuditAction.LEAVE_REJECT)
    assert (denied.outcome, denied.details["reason"]) == (AuditOutcome.DENIED, "invalid_input")
    assert allowed.outcome == AuditOutcome.ALLOWED


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_another_teams_manager_gets_not_found(
    db: AsyncSession,
    make_employee: MakeEmployee,
    team: tuple[EmployeeModel, EmployeeModel],
    decision: str,
) -> None:
    _, employee = team
    other_manager = await make_employee(role=Role.MANAGER)
    await add_balance(db, employee)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(7))

    with pytest.raises(NotFound):
        await getattr(service(db), decision)(as_user(other_manager), request.id, "No")

    assert (await reload(db, request)).status == LeaveStatus.PENDING
    action = AuditAction.LEAVE_APPROVE if decision == "approve" else AuditAction.LEAVE_REJECT
    [row] = await audit_rows(db, action)
    assert (row.outcome, row.actor_id, row.target_id) == (
        AuditOutcome.DENIED,
        other_manager.id,
        request.id,
    )
    assert row.details == {"reason": "not_found"}


async def test_a_manager_cant_approve_their_own_request(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    head = await make_employee(role=Role.MANAGER)
    manager = await make_employee(role=Role.MANAGER, manager_id=head.id)
    await add_balance(db, manager)
    own = await add_request(db, manager, TODAY + days(7), TODAY + days(7))

    with pytest.raises(NotFound):
        await service(db).approve(as_user(manager), own.id)

    # Their own manager can.
    approved = await service(db).approve(as_user(head), own.id)
    assert approved.status == LeaveStatus.APPROVED


@pytest.mark.parametrize("role", [Role.EMPLOYEE, Role.HR])
async def test_only_managers_decide(
    db: AsyncSession,
    make_employee: MakeEmployee,
    team: tuple[EmployeeModel, EmployeeModel],
    role: Role,
) -> None:
    _, employee = team
    # Even with the request's employee reporting to them.
    other = await make_employee(role=role)
    employee.manager_id = other.id
    await db.flush()
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(7))

    with pytest.raises(Forbidden):
        await service(db).approve(as_user(other), request.id)

    [row] = await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)
    assert row.details == {
        "operation": "leave.approve",
        "role": role.value,
        "required": ["manager"],
    }
    assert await audit_rows(db, AuditAction.LEAVE_APPROVE) == []


@pytest.mark.parametrize(
    "status", [LeaveStatus.APPROVED, LeaveStatus.REJECTED, LeaveStatus.CANCELLED]
)
async def test_only_pending_requests_can_be_decided(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel], status: LeaveStatus
) -> None:
    manager, employee = team
    balance = await add_balance(db, employee, used=1)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(7), status=status)

    with pytest.raises(InvalidStatus, match=f"already {status.value}"):
        await service(db).approve(as_user(manager), request.id)

    assert (await reload(db, balance)).used == 1


async def test_approval_refuses_to_overdraw_the_balance(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    manager, employee = team
    # Valid when applied for, but HR has since lowered the allotment.
    await add_balance(db, employee, allotted=2)
    request = await add_request(db, employee, TODAY + days(7), TODAY + days(9))

    with pytest.raises(InsufficientBalance) as exc_info:
        await service(db).approve(as_user(manager), request.id)

    assert exc_info.value.details == {"requested": 3, "available": 2}
    assert (await reload(db, request)).status == LeaveStatus.PENDING


# Reads


async def test_balances_summarise_each_type(
    db: AsyncSession, team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, employee = team
    await add_balance(db, employee, allotted=12, used=2)
    await add_balance(db, employee, allotted=10, leave_type=LeaveType.SICK)
    await add_request(db, employee, TODAY + days(7), TODAY + days(8), days=2)

    summaries = await service(db).get_balances(as_user(employee))

    assert [(s.leave_type, s.allotted, s.used, s.pending, s.available) for s in summaries] == [
        (LeaveType.CASUAL, 12, 2, 2, 8),
        (LeaveType.SICK, 10, 0, 0, 10),
    ]


async def test_team_requests_are_the_managers_reports_only(
    db: AsyncSession,
    make_employee: MakeEmployee,
    team: tuple[EmployeeModel, EmployeeModel],
) -> None:
    manager, employee = team
    outsider = await make_employee()
    pending = await add_request(db, employee, TODAY + days(7), TODAY + days(7))
    approved = await add_request(
        db, employee, TODAY + days(1), TODAY + days(1), status=LeaveStatus.APPROVED
    )
    await add_request(db, outsider, TODAY + days(7), TODAY + days(7))
    await add_request(db, manager, TODAY + days(7), TODAY + days(7))
    leave = service(db)

    everything = await leave.list_team_requests(as_user(manager))
    assert [r.id for r in everything] == [pending.id, approved.id]
    assert everything[0].employee.name == "Report"
    only_pending = await leave.list_team_requests(as_user(manager), LeaveStatus.PENDING)
    assert [r.id for r in only_pending] == [pending.id]

    with pytest.raises(Forbidden):
        await leave.list_team_requests(as_user(employee))


async def test_hr_reads_any_employees_leave_summary_and_it_is_audited(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    hr = await make_employee(role=Role.HR)
    ravi = await make_employee(is_active=False)  # People who have left too.
    await add_balance(db, ravi, allotted=12, used=2)
    await add_balance(db, ravi, allotted=10, year=2025)
    older = await add_request(db, ravi, date(2025, 12, 1), date(2025, 12, 1))
    newer = await add_request(db, ravi, TODAY + days(7), TODAY + days(8))

    summary = await service(db).get_employee_summary(as_user(hr), ravi.id)
    last_year = await service(db).get_employee_summary(as_user(hr), ravi.id, 2025, limit=1)

    assert (summary.employee.id, summary.year) == (ravi.id, 2026)
    [casual] = summary.balances
    assert (casual.allotted, casual.used, casual.pending, casual.available) == (12, 2, 2, 8)
    assert [r.id for r in summary.recent_requests] == [newer.id, older.id]
    assert last_year.year == 2025 and [b.allotted for b in last_year.balances] == [10]
    assert [r.id for r in last_year.recent_requests] == [newer.id]
    rows = await audit_rows(db, AuditAction.EMPLOYEE_VIEW)
    assert [(r.outcome, r.actor_id, r.target_id, r.details) for r in rows] == [
        (AuditOutcome.ALLOWED, hr.id, ravi.id, {"view": "leave_summary", "year": 2026}),
        (AuditOutcome.ALLOWED, hr.id, ravi.id, {"view": "leave_summary", "year": 2025}),
    ]


@pytest.mark.parametrize("role", [Role.MANAGER, Role.EMPLOYEE])
async def test_only_hr_reads_leave_summaries(
    db: AsyncSession, make_employee: MakeEmployee, role: Role
) -> None:
    caller = await make_employee(role=role)
    report = await make_employee(manager_id=caller.id)  # Even the caller's own report.

    with pytest.raises(Forbidden):
        await service(db).get_employee_summary(as_user(caller), report.id)

    assert await audit_rows(db, AuditAction.EMPLOYEE_VIEW) == []
    assert len(await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)) == 1


async def test_leave_summary_of_an_unknown_employee_is_not_found(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    hr = await make_employee(role=Role.HR)

    with pytest.raises(NotFound, match="Employee not found"):
        await service(db).get_employee_summary(as_user(hr), uuid.uuid4())

    assert await audit_rows(db, AuditAction.EMPLOYEE_VIEW) == []


# Concurrency


async def test_concurrent_approvals_cant_overdraw_a_balance(
    engine: AsyncEngine, committed_team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    manager, report = committed_team
    async with AsyncSession(engine, expire_on_commit=False) as db:
        await add_balance(db, report, allotted=5)
        # 3 + 3 days: each fits alone, both together don't.
        first = await add_request(db, report, TODAY + days(7), TODAY + days(9))
        second = await add_request(db, report, TODAY + days(14), TODAY + days(16))
        await db.commit()

    async def approve(request_id: uuid.UUID) -> str:
        async with AsyncSession(engine, expire_on_commit=False) as db:
            try:
                await service(db).approve(as_user(manager), request_id)
            except InsufficientBalance:
                return "refused"
            return "approved"

    # Hold the balance row so both approvals queue up behind it before either
    # reads `used`. Releasing it lets them race for real.
    async with engine.connect() as blocker:
        await blocker.begin()
        await blocker.execute(
            select(LeaveBalanceModel)
            .where(LeaveBalanceModel.employee_id == report.id)
            .with_for_update()
        )
        tasks = [asyncio.create_task(approve(r.id)) for r in (first, second)]
        await wait_for_lock_waiters(engine, 2)
        await blocker.rollback()
    results = await asyncio.gather(*tasks)

    assert sorted(results) == ["approved", "refused"]
    async with AsyncSession(engine) as db:
        assert (await db.get(LeaveBalanceModel, (report.id, 2026, LeaveType.CASUAL))).used == 3
        statuses = await db.scalars(
            select(LeaveRequestModel.status).where(LeaveRequestModel.employee_id == report.id)
        )
        assert sorted(statuses.all()) == [LeaveStatus.APPROVED, LeaveStatus.PENDING]
