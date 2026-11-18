import uuid
from datetime import date

from pydantic import BaseModel, ConfigDict

from warden.models.enums import Role


class EmployeeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_code: str
    name: str
    email: str
    role: Role
    department_id: uuid.UUID
    manager_id: uuid.UUID | None
    job_title: str
    joining_date: date


class DepartmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


class ProfileResponse(EmployeeResponse):
    """A full profile. Needs `department` loaded; the repositories load it."""

    department: DepartmentResponse
    is_active: bool


class EmployeeSummaryResponse(BaseModel):
    """Enough to recognise and contact someone: a manager, a teammate."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_code: str
    name: str
    email: str
    job_title: str
