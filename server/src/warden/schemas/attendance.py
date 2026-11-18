from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from warden.models.enums import AttendanceStatus
from warden.schemas.employee import EmployeeSummaryResponse


class AttendanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    status: AttendanceStatus
    check_in: datetime | None
    check_out: datetime | None


class TeamAttendanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    employee: EmployeeSummaryResponse
    # Null when nothing was recorded for the day.
    attendance: AttendanceResponse | None
