import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from warden.models.enums import LeaveStatus, LeaveType
from warden.schemas.employee import EmployeeSummaryResponse
from warden.services.leave import MAX_COMMENT_LENGTH, MAX_REASON_LENGTH


class LeaveBalanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    leave_type: LeaveType
    year: int
    allotted: int
    used: int
    # Days in requests still waiting for a decision.
    pending: int
    # allotted - used - pending: what a new request can draw on.
    available: int


class LeaveRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    leave_type: LeaveType
    start_date: date
    end_date: date
    days: int
    reason: str | None
    status: LeaveStatus
    approver_id: uuid.UUID | None
    decision_comment: str | None
    decided_at: datetime | None
    created_at: datetime


class TeamLeaveRequestResponse(LeaveRequestResponse):
    employee: EmployeeSummaryResponse


class EmployeeLeaveSummaryResponse(BaseModel):
    """One employee's balances for a year and their recent requests, for HR."""

    model_config = ConfigDict(from_attributes=True)

    employee: EmployeeSummaryResponse
    year: int
    balances: list[LeaveBalanceResponse]
    recent_requests: list[LeaveRequestResponse]


class LeaveApplication(BaseModel):
    leave_type: LeaveType
    start_date: date
    end_date: date
    reason: str | None = Field(default=None, max_length=MAX_REASON_LENGTH)


class ApproveLeave(BaseModel):
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_LENGTH)


class RejectLeave(BaseModel):
    comment: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)
