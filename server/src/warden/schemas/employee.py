from datetime import date

from pydantic import BaseModel, ConfigDict

from warden.models.enums import Role


class EmployeeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_code: str
    name: str
    email: str
    role: Role
    department_id: int
    manager_id: int | None
    job_title: str
    joining_date: date
