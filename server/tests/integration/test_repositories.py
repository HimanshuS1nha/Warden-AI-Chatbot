from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_utils.compat import uuid7

from tests.factories import MakeEmployee
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    DepartmentModel,
    EmployeeModel,
    HolidayModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
    PayslipModel,
    Role,
)
from warden.repositories import (
    AttendanceRepository,
    EmployeeRepository,
    HolidayRepository,
    LeaveRepository,
    PayslipRepository,
)
from warden.repositories.employee import escape_like


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


# Employees


async def test_get_loads_department_and_returns_none_for_unknown_id(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    repo = EmployeeRepository(db)
    db.expunge_all()

    found = await repo.get(employee.id)
    assert found is not None
    assert found.department.name == "Engineering"
    assert await repo.get(uuid7()) is None


async def test_get_manager(db: AsyncSession, make_employee: MakeEmployee) -> None:
    manager = await make_employee(role=Role.MANAGER, name="Meera")
    report = await make_employee(manager_id=manager.id)
    repo = EmployeeRepository(db)

    found = await repo.get_manager(report.id)
    assert found is not None
    assert found.id == manager.id
    assert await repo.get_manager(manager.id) is None


async def test_list_reports_returns_active_direct_reports_only(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee(role=Role.MANAGER)
    other_manager = await make_employee(role=Role.MANAGER)
    bea = await make_employee(manager_id=manager.id, name="Bea")
    ana = await make_employee(manager_id=manager.id, name="Ana")
    await make_employee(manager_id=manager.id, name="Cal", is_active=False)
    await make_employee(manager_id=other_manager.id)
    # A report's report isn't a direct report.
    await make_employee(manager_id=ana.id)

    reports = await EmployeeRepository(db).list_reports(manager.id)
    assert [e.id for e in reports] == [ana.id, bea.id]


async def test_search_by_name_and_department(db: AsyncSession, make_employee: MakeEmployee) -> None:
    sales = DepartmentModel(name="Sales")
    db.add(sales)
    await db.flush()
    ravi = await make_employee(name="Ravi Kumar")
    ravi_sales = await make_employee(name="Ravina Shah", department_id=sales.id)
    await make_employee(name="Ravi Inactive", is_active=False)
    await make_employee(name="Someone Else")
    repo = EmployeeRepository(db)

    assert {e.id for e in await repo.search(name="ravi")} == {ravi.id, ravi_sales.id}
    assert [e.id for e in await repo.search(name="ravi", department="sales")] == [ravi_sales.id]
    assert [e.id for e in await repo.search(name="  RAVI KUMAR ")] == [ravi.id]
    # The department must match the whole name, not part of it.
    assert await repo.search(department="sal") == []
    assert len(await repo.search(limit=2)) == 2


async def test_search_treats_like_wildcards_literally(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    await make_employee(name="Ann")
    repo = EmployeeRepository(db)

    assert await repo.search(name="%") == []
    assert await repo.search(name="A_n") == []
    assert escape_like("50%_a\\b") == "50\\%\\_a\\\\b"


# Leave


async def test_get_balances_for_one_year(db: AsyncSession, make_employee: MakeEmployee) -> None:
    employee = await make_employee()
    other = await make_employee()
    db.add_all(
        [
            LeaveBalanceModel(employee_id=employee.id, year=2026, leave_type=t, allotted=10)
            for t in LeaveType
        ]
        + [
            LeaveBalanceModel(
                employee_id=employee.id, year=2025, leave_type=LeaveType.CASUAL, allotted=10
            ),
            LeaveBalanceModel(
                employee_id=other.id, year=2026, leave_type=LeaveType.CASUAL, allotted=10
            ),
        ]
    )
    await db.flush()
    repo = LeaveRepository(db)

    balances = await repo.get_balances(employee.id, 2026)
    assert [b.leave_type for b in balances] == list(LeaveType)
    assert all(b.employee_id == employee.id and b.year == 2026 for b in balances)

    locked = await repo.get_balance_for_update(employee.id, 2026, LeaveType.SICK)
    assert locked is not None
    assert locked.leave_type == LeaveType.SICK
    assert await repo.get_balance_for_update(employee.id, 2024, LeaveType.SICK) is None


async def test_list_requests_filters_orders_and_limits(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    other = await make_employee()
    march = await add_request(db, employee, date(2026, 3, 2), date(2026, 3, 3))
    june = await add_request(
        db, employee, date(2026, 6, 1), date(2026, 6, 1), status=LeaveStatus.APPROVED
    )
    may = await add_request(db, employee, date(2026, 5, 4), date(2026, 5, 4))
    await add_request(db, other, date(2026, 7, 1), date(2026, 7, 1))
    repo = LeaveRepository(db)

    assert [r.id for r in await repo.list_requests(employee.id)] == [june.id, may.id, march.id]
    pending = await repo.list_requests(employee.id, LeaveStatus.PENDING)
    assert [r.id for r in pending] == [may.id, march.id]
    assert [r.id for r in await repo.list_requests(employee.id, limit=1)] == [june.id]


async def test_get_request_for_employee_is_scoped(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    owner = await make_employee()
    other = await make_employee()
    request = await add_request(db, owner, date(2026, 3, 2), date(2026, 3, 2))
    repo = LeaveRepository(db)

    found = await repo.get_request_for_employee(request.id, owner.id)
    assert found is not None
    assert found.id == request.id
    assert await repo.get_request_for_employee(request.id, owner.id, for_update=True) is not None
    assert await repo.get_request_for_employee(request.id, other.id) is None


@pytest.mark.parametrize("for_update", [False, True])
async def test_get_request_for_manager_is_scoped_to_direct_reports(
    db: AsyncSession, make_employee: MakeEmployee, for_update: bool
) -> None:
    head = await make_employee(role=Role.MANAGER)
    manager = await make_employee(role=Role.MANAGER, manager_id=head.id)
    other_manager = await make_employee(role=Role.MANAGER)
    report = await make_employee(manager_id=manager.id)
    report_request = await add_request(db, report, date(2026, 3, 2), date(2026, 3, 2))
    own_request = await add_request(db, manager, date(2026, 3, 9), date(2026, 3, 9))
    repo = LeaveRepository(db)

    found = await repo.get_request_for_manager(report_request.id, manager.id, for_update=for_update)
    assert found is not None
    assert found.id == report_request.id
    # Another team's manager, and the manager's own request, are out of scope.
    assert await repo.get_request_for_manager(report_request.id, other_manager.id) is None
    assert await repo.get_request_for_manager(own_request.id, manager.id) is None
    # Only direct reports: the head doesn't manage the manager's team.
    assert await repo.get_request_for_manager(report_request.id, head.id) is None


async def test_find_overlapping(db: AsyncSession, make_employee: MakeEmployee) -> None:
    employee = await make_employee()
    other = await make_employee()
    pending = await add_request(db, employee, date(2026, 3, 2), date(2026, 3, 4))
    approved = await add_request(
        db, employee, date(2026, 3, 9), date(2026, 3, 10), status=LeaveStatus.APPROVED
    )
    for status in (LeaveStatus.REJECTED, LeaveStatus.CANCELLED):
        await add_request(db, employee, date(2026, 3, 5), date(2026, 3, 6), status=status)
    await add_request(db, other, date(2026, 3, 1), date(2026, 3, 31))
    repo = LeaveRepository(db)

    async def overlapping(start: date, end: date) -> list[Any]:
        return [r.id for r in await repo.find_overlapping(employee.id, start, end)]

    assert await overlapping(date(2026, 3, 4), date(2026, 3, 4)) == [pending.id]
    assert await overlapping(date(2026, 3, 10), date(2026, 3, 12)) == [approved.id]
    assert await overlapping(date(2026, 3, 1), date(2026, 3, 31)) == [pending.id, approved.id]
    # Rejected and cancelled requests on these days don't count.
    assert await overlapping(date(2026, 3, 5), date(2026, 3, 6)) == []
    assert await overlapping(date(2026, 2, 27), date(2026, 3, 1)) == []


async def test_pending_days_counts_pending_requests_of_the_type_and_year(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    other = await make_employee()
    repo = LeaveRepository(db)
    assert await repo.pending_days(employee.id, 2026, LeaveType.CASUAL) == 0

    await add_request(db, employee, date(2026, 3, 2), date(2026, 3, 4), days=3)
    await add_request(db, employee, date(2026, 12, 30), date(2026, 12, 31), days=2)
    # None of these count:
    approved_day, sick_day, next_year_day = date(2026, 4, 6), date(2026, 5, 4), date(2027, 1, 4)
    await add_request(db, employee, approved_day, approved_day, status=LeaveStatus.APPROVED)
    await add_request(db, employee, sick_day, sick_day, leave_type=LeaveType.SICK)
    await add_request(db, employee, next_year_day, next_year_day)
    await add_request(db, other, date(2026, 3, 2), date(2026, 3, 2), days=1)

    assert await repo.pending_days(employee.id, 2026, LeaveType.CASUAL) == 5


# Attendance, payslips, holidays


async def test_attendance_for_employee_in_range(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    other = await make_employee()
    db.add_all(
        [
            AttendanceModel(employee_id=employee.id, date=d, status=AttendanceStatus.PRESENT)
            for d in (date(2026, 2, 27), date(2026, 3, 3), date(2026, 3, 2), date(2026, 4, 1))
        ]
        + [
            AttendanceModel(
                employee_id=other.id, date=date(2026, 3, 2), status=AttendanceStatus.WFH
            )
        ]
    )
    await db.flush()

    rows = await AttendanceRepository(db).list_for_employee(
        employee.id, date(2026, 3, 1), date(2026, 3, 31)
    )
    assert [r.date for r in rows] == [date(2026, 3, 2), date(2026, 3, 3)]


async def test_team_attendance_on_date_includes_reports_without_a_record(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee(role=Role.MANAGER)
    present = await make_employee(manager_id=manager.id, name="Ana")
    missing = await make_employee(manager_id=manager.id, name="Bea")
    await make_employee(manager_id=manager.id, name="Cal", is_active=False)
    outsider = await make_employee()
    day = date(2026, 3, 2)
    db.add_all(
        [
            AttendanceModel(
                employee_id=present.id,
                date=day,
                status=AttendanceStatus.PRESENT,
                check_in=datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
            ),
            AttendanceModel(
                employee_id=present.id, date=date(2026, 3, 3), status=AttendanceStatus.ABSENT
            ),
            AttendanceModel(employee_id=outsider.id, date=day, status=AttendanceStatus.PRESENT),
        ]
    )
    await db.flush()

    rows = await AttendanceRepository(db).list_for_team_on_date(manager.id, day)
    assert [(e.id, a.status if a else None) for e, a in rows] == [
        (present.id, AttendanceStatus.PRESENT),
        (missing.id, None),
    ]


async def test_payslips_newest_first_and_filtered_by_year(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    other = await make_employee()
    db.add_all(
        [
            PayslipModel(employee_id=employee.id, year=y, month=m, net_pay=Decimal("1000.00"))
            for y, m in ((2025, 12), (2026, 1), (2026, 2))
        ]
        + [PayslipModel(employee_id=other.id, year=2026, month=1, net_pay=Decimal("1.00"))]
    )
    await db.flush()
    repo = PayslipRepository(db)

    all_years = await repo.list_for_employee(employee.id)
    assert [(p.year, p.month) for p in all_years] == [(2026, 2), (2026, 1), (2025, 12)]
    only_2025 = await repo.list_for_employee(employee.id, 2025)
    assert [(p.year, p.month) for p in only_2025] == [(2025, 12)]


async def test_holidays_for_year(db: AsyncSession) -> None:
    db.add_all(
        [
            HolidayModel(date=date(2026, 12, 25), name="Christmas"),
            HolidayModel(date=date(2026, 1, 1), name="New Year"),
            HolidayModel(date=date(2027, 1, 1), name="New Year"),
            HolidayModel(date=date(2025, 12, 31), name="Year End"),
        ]
    )
    await db.flush()

    holidays = await HolidayRepository(db).list_for_year(2026)
    assert [h.date for h in holidays] == [date(2026, 1, 1), date(2026, 12, 25)]
