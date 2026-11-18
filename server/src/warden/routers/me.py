"""The caller's own data. Every route here is scoped to the session's
employee; none takes an employee ID."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from warden.core.dependencies import CurrentUserDep
from warden.models import LeaveStatus
from warden.routers.services import (
    AttendanceServiceDep,
    EmployeeServiceDep,
    LeaveServiceDep,
    PayslipServiceDep,
)
from warden.schemas.attendance import AttendanceResponse
from warden.schemas.employee import EmployeeSummaryResponse, ProfileResponse
from warden.schemas.leave import LeaveApplication, LeaveBalanceResponse, LeaveRequestResponse
from warden.schemas.payslip import PayslipResponse
from warden.services.leave import company_today

router = APIRouter(prefix="/api/me", tags=["me"])

Year = Annotated[int | None, Query(ge=2000, le=2100)]


@router.get("")
async def get_profile(user: CurrentUserDep, employees: EmployeeServiceDep) -> ProfileResponse:
    return ProfileResponse.model_validate(await employees.get_profile(user))


@router.get("/manager")
async def get_manager(
    user: CurrentUserDep, employees: EmployeeServiceDep
) -> EmployeeSummaryResponse | None:
    """Null for someone with no manager, such as the HR head."""
    manager = await employees.get_manager(user)
    return EmployeeSummaryResponse.model_validate(manager) if manager else None


@router.get("/leave-balances")
async def get_leave_balances(
    user: CurrentUserDep, leave: LeaveServiceDep, year: Year = None
) -> list[LeaveBalanceResponse]:
    balances = await leave.get_balances(user, year)
    return [LeaveBalanceResponse.model_validate(b) for b in balances]


@router.get("/leave-requests")
async def list_leave_requests(
    user: CurrentUserDep,
    leave: LeaveServiceDep,
    status: LeaveStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[LeaveRequestResponse]:
    requests = await leave.list_my_requests(user, status, limit)
    return [LeaveRequestResponse.model_validate(r) for r in requests]


@router.post("/leave-requests", status_code=status.HTTP_201_CREATED)
async def apply_for_leave(
    data: LeaveApplication, user: CurrentUserDep, leave: LeaveServiceDep
) -> LeaveRequestResponse:
    request = await leave.apply(user, data.leave_type, data.start_date, data.end_date, data.reason)
    return LeaveRequestResponse.model_validate(request)


@router.post("/leave-requests/{request_id}/cancel")
async def cancel_leave_request(
    request_id: uuid.UUID, user: CurrentUserDep, leave: LeaveServiceDep
) -> LeaveRequestResponse:
    return LeaveRequestResponse.model_validate(await leave.cancel(user, request_id))


@router.get("/attendance")
async def get_attendance(
    user: CurrentUserDep,
    attendance: AttendanceServiceDep,
    month: Annotated[int | None, Query(ge=1, le=12)] = None,
    year: Year = None,
) -> list[AttendanceResponse]:
    """One calendar month; this month unless `month` and `year` are given."""
    today = company_today()
    rows = await attendance.get_mine(user, month or today.month, year or today.year)
    return [AttendanceResponse.model_validate(r) for r in rows]


@router.get("/payslips")
async def get_payslips(
    user: CurrentUserDep, payslips: PayslipServiceDep, year: Year = None
) -> list[PayslipResponse]:
    return [PayslipResponse.model_validate(p) for p in await payslips.get_mine(user, year)]
