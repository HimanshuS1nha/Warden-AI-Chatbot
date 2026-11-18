from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_utils.compat import uuid7

from tests.factories import MakeEmployee, as_user
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    AuditAction,
    AuditLogModel,
    AuditOutcome,
    AuditTarget,
    DepartmentModel,
    EmployeeModel,
    HolidayModel,
    PayslipModel,
    Role,
)
from warden.services.attendance import AttendanceService
from warden.services.employee import EmployeeService
from warden.services.errors import Forbidden, InvalidInput, NotFound
from warden.services.holiday import HolidayService
from warden.services.payslip import PayslipService


async def audit_rows(db: AsyncSession, action: AuditAction) -> Sequence[AuditLogModel]:
    result = await db.scalars(
        select(AuditLogModel)
        .where(AuditLogModel.action == action)
        .order_by(AuditLogModel.id)
        .execution_options(populate_existing=True)
    )
    return result.all()


@pytest.fixture
async def org(make_employee: MakeEmployee) -> dict[str, EmployeeModel]:
    hr = await make_employee(role=Role.HR, name="Hana HR")
    manager = await make_employee(role=Role.MANAGER, name="Mona Manager", manager_id=hr.id)
    other_manager = await make_employee(role=Role.MANAGER, name="Omar Other")
    report = await make_employee(name="Ravi Report", manager_id=manager.id)
    outsider = await make_employee(name="Ola Outsider", manager_id=other_manager.id)
    return {
        "hr": hr,
        "manager": manager,
        "other_manager": other_manager,
        "report": report,
        "outsider": outsider,
    }


# Employees


async def test_profile_and_manager_are_your_own(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    employees = EmployeeService(db)
    user = as_user(org["report"])

    profile = await employees.get_profile(user)
    assert profile.id == org["report"].id
    assert profile.department.name == "Engineering"
    manager = await employees.get_manager(user)
    assert manager is not None and manager.id == org["manager"].id
    assert await employees.get_manager(as_user(org["other_manager"])) is None
    # Reading your own data isn't audited.
    assert await audit_rows(db, AuditAction.EMPLOYEE_VIEW) == []


async def test_list_team_is_for_managers(db: AsyncSession, org: dict[str, EmployeeModel]) -> None:
    employees = EmployeeService(db)

    team = await employees.list_team(as_user(org["manager"]))
    assert [e.id for e in team] == [org["report"].id]

    with pytest.raises(Forbidden):
        await employees.list_team(as_user(org["report"]))
    [row] = await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)
    assert row.details == {
        "operation": "employee.list_team",
        "role": "employee",
        "required": ["manager"],
    }


async def test_hr_search_is_audited(db: AsyncSession, org: dict[str, EmployeeModel]) -> None:
    sales = DepartmentModel(name="Sales")
    db.add(sales)
    await db.flush()
    org["outsider"].department_id = sales.id
    await db.flush()
    employees = EmployeeService(db)

    found = await employees.search(as_user(org["hr"]), name="o", department="sales")

    assert [e.id for e in found] == [org["outsider"].id]
    [row] = await audit_rows(db, AuditAction.EMPLOYEE_VIEW)
    assert (row.outcome, row.actor_id, row.target_id) == (
        AuditOutcome.ALLOWED,
        org["hr"].id,
        None,
    )
    assert row.details == {"search": {"name": "o", "department": "sales"}, "results": 1}


@pytest.mark.parametrize("role", ["manager", "report"])
async def test_only_hr_can_search(
    db: AsyncSession, org: dict[str, EmployeeModel], role: str
) -> None:
    with pytest.raises(Forbidden):
        await EmployeeService(db).search(as_user(org[role]), name="a")

    assert await audit_rows(db, AuditAction.EMPLOYEE_VIEW) == []
    assert len(await audit_rows(db, AuditAction.AUTHZ_FORBIDDEN)) == 1


async def test_hr_can_view_anyone_and_each_view_is_audited(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    employees = EmployeeService(db)
    hr = as_user(org["hr"])

    viewed = await employees.get_employee(hr, org["outsider"].id)
    assert viewed.id == org["outsider"].id
    with pytest.raises(NotFound):
        await employees.get_employee(hr, uuid7())

    [row] = await audit_rows(db, AuditAction.EMPLOYEE_VIEW)
    assert (row.outcome, row.target_type, row.target_id) == (
        AuditOutcome.ALLOWED,
        AuditTarget.EMPLOYEE,
        org["outsider"].id,
    )


async def test_manager_sees_own_reports_and_probing_is_audited(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    employees = EmployeeService(db)
    manager = as_user(org["manager"])

    viewed = await employees.get_employee(manager, org["report"].id)
    assert viewed.id == org["report"].id
    assert await audit_rows(db, AuditAction.EMPLOYEE_VIEW) == []

    # Someone else's report and an ID that doesn't exist look the same.
    missing = uuid7()
    for employee_id in (org["outsider"].id, missing):
        with pytest.raises(NotFound, match="Employee not found"):
            await employees.get_employee(manager, employee_id)

    rows = await audit_rows(db, AuditAction.EMPLOYEE_VIEW)
    assert [(r.outcome, r.target_id, r.details) for r in rows] == [
        (AuditOutcome.DENIED, org["outsider"].id, {"reason": "not_found"}),
        (AuditOutcome.DENIED, missing, {"reason": "not_found"}),
    ]


async def test_employees_can_only_view_themselves(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    employees = EmployeeService(db)
    report = as_user(org["report"])

    assert (await employees.get_employee(report, org["report"].id)).id == org["report"].id
    with pytest.raises(Forbidden):
        await employees.get_employee(report, org["manager"].id)


# Attendance


async def test_my_attendance_is_one_calendar_month(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    other = await make_employee()
    days = (date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 28), date(2026, 3, 1))
    db.add_all(
        [
            AttendanceModel(employee_id=employee.id, date=d, status=AttendanceStatus.WFH)
            for d in days
        ]
        + [
            AttendanceModel(
                employee_id=other.id, date=date(2026, 2, 2), status=AttendanceStatus.PRESENT
            )
        ]
    )
    await db.flush()
    attendance = AttendanceService(db)

    rows = await attendance.get_mine(as_user(employee), month=2, year=2026)

    assert [r.date for r in rows] == [date(2026, 2, 1), date(2026, 2, 28)]
    for month in (0, 13):
        with pytest.raises(InvalidInput):
            await attendance.get_mine(as_user(employee), month=month, year=2026)


async def test_team_attendance_is_for_managers(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    day = date(2026, 3, 2)
    db.add(AttendanceModel(employee_id=org["report"].id, date=day, status=AttendanceStatus.ABSENT))
    await db.flush()
    attendance = AttendanceService(db)

    team = await attendance.get_team_for_date(as_user(org["manager"]), day)

    assert [(m.employee.id, m.attendance.status if m.attendance else None) for m in team] == [
        (org["report"].id, AttendanceStatus.ABSENT)
    ]
    with pytest.raises(Forbidden):
        await attendance.get_team_for_date(as_user(org["hr"]), day)


# Payslips and holidays


async def test_payslips_are_only_ever_your_own(
    db: AsyncSession, org: dict[str, EmployeeModel]
) -> None:
    for employee in (org["report"], org["hr"]):
        db.add(PayslipModel(employee_id=employee.id, year=2026, month=1, net_pay=Decimal("100.00")))
    await db.flush()
    payslips = PayslipService(db)

    for key in ("report", "hr"):
        mine = await payslips.get_mine(as_user(org[key]))
        assert [p.employee_id for p in mine] == [org[key].id]
    assert await payslips.get_mine(as_user(org["manager"])) == []


async def test_holidays_default_to_this_year(db: AsyncSession) -> None:
    db.add_all(
        [
            HolidayModel(date=date(2026, 8, 15), name="Independence Day"),
            HolidayModel(date=date(2027, 1, 26), name="Republic Day"),
        ]
    )
    await db.flush()
    holidays = HolidayService(db, today=lambda: date(2026, 6, 1))

    assert [h.name for h in await holidays.list()] == ["Independence Day"]
    assert [h.name for h in await holidays.list(2027)] == ["Republic Day"]
