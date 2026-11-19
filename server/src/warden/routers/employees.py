"""Looking up other employees. HR can search and view anyone; a manager can
view their own direct reports. Every HR view is audited by the service."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from warden.core.dependencies import CurrentUser, require_role
from warden.models import Role
from warden.routers.services import EmployeeServiceDep, LeaveServiceDep
from warden.schemas.employee import ProfileResponse
from warden.schemas.leave import EmployeeLeaveSummaryResponse

router = APIRouter(prefix="/api/employees", tags=["employees"])

HRUser = Annotated[CurrentUser, Depends(require_role(Role.HR))]
HROrManager = Annotated[CurrentUser, Depends(require_role(Role.HR, Role.MANAGER))]


@router.get("")
async def search_employees(
    user: HRUser,
    employees: EmployeeServiceDep,
    name: Annotated[str | None, Query(max_length=100)] = None,
    department: Annotated[str | None, Query(max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 50,
) -> list[ProfileResponse]:
    """Active employees whose name contains `name` and whose department is
    `department`, both ignoring case."""
    results = await employees.search(user, name, department, limit)
    return [ProfileResponse.model_validate(e) for e in results]


@router.get("/{employee_id}")
async def get_employee(
    employee_id: uuid.UUID, user: HROrManager, employees: EmployeeServiceDep
) -> ProfileResponse:
    """404 for an ID outside the caller's scope, the same as for one that
    doesn't exist."""
    return ProfileResponse.model_validate(await employees.get_employee(user, employee_id))


@router.get("/{employee_id}/leave")
async def get_employee_leave_summary(
    employee_id: uuid.UUID,
    user: HRUser,
    leave: LeaveServiceDep,
    year: Annotated[int | None, Query(ge=2000, le=2100)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> EmployeeLeaveSummaryResponse:
    """HR only: the employee's balances for `year` (default: this year) and
    their most recent requests. Audited as `employee.view`."""
    summary = await leave.get_employee_summary(user, employee_id, year, limit)
    return EmployeeLeaveSummaryResponse.model_validate(summary)
