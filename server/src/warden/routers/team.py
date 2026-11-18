"""A manager's direct reports. The router only lets managers in; the
services scope every query to the caller's own reports."""

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from warden.core.dependencies import CurrentUser, require_role
from warden.models import LeaveStatus, Role
from warden.routers.services import AttendanceServiceDep, EmployeeServiceDep, LeaveServiceDep
from warden.schemas.attendance import TeamAttendanceResponse
from warden.schemas.employee import EmployeeSummaryResponse
from warden.schemas.leave import (
    ApproveLeave,
    LeaveRequestResponse,
    RejectLeave,
    TeamLeaveRequestResponse,
)
from warden.services.leave import company_today

router = APIRouter(prefix="/api/team", tags=["team"])

ManagerUser = Annotated[CurrentUser, Depends(require_role(Role.MANAGER))]


@router.get("")
async def list_team(
    user: ManagerUser, employees: EmployeeServiceDep
) -> list[EmployeeSummaryResponse]:
    return [EmployeeSummaryResponse.model_validate(e) for e in await employees.list_team(user)]


@router.get("/leave-requests")
async def list_team_leave_requests(
    user: ManagerUser,
    leave: LeaveServiceDep,
    status: LeaveStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[TeamLeaveRequestResponse]:
    requests = await leave.list_team_requests(user, status, limit)
    return [TeamLeaveRequestResponse.model_validate(r) for r in requests]


@router.post("/leave-requests/{request_id}/approve")
async def approve_leave_request(
    request_id: uuid.UUID, data: ApproveLeave, user: ManagerUser, leave: LeaveServiceDep
) -> LeaveRequestResponse:
    return LeaveRequestResponse.model_validate(await leave.approve(user, request_id, data.comment))


@router.post("/leave-requests/{request_id}/reject")
async def reject_leave_request(
    request_id: uuid.UUID, data: RejectLeave, user: ManagerUser, leave: LeaveServiceDep
) -> LeaveRequestResponse:
    return LeaveRequestResponse.model_validate(await leave.reject(user, request_id, data.comment))


@router.get("/attendance")
async def get_team_attendance(
    user: ManagerUser,
    attendance: AttendanceServiceDep,
    day: Annotated[date | None, Query(alias="date")] = None,
) -> list[TeamAttendanceResponse]:
    """Each report's attendance on `day` (default: today)."""
    rows = await attendance.get_team_for_date(user, day or company_today())
    return [TeamAttendanceResponse.model_validate(r) for r in rows]
