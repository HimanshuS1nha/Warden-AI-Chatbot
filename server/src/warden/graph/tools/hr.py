"""Tools for HR to look up other employees. Only HR gets them.

Unlike the other tools, these take an `employee_id`: the person HR is asking
about, never the caller. The model finds it with `search_employees` first.
The services check the HR role again and audit every read as
`employee.view`, so the audit log shows whose data HR looked at through the
agent just as through the REST routes.

Results are JSON strings. Employee and manager IDs are kept, because the
model passes them to the next tool.
"""

import json
import uuid
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from warden.core.dependencies import CurrentUser
from warden.graph.tools.common import Year, company_time
from warden.graph.tools.services import ToolServices
from warden.models import EmployeeModel
from warden.schemas.employee import ProfileResponse
from warden.schemas.leave import LeaveBalanceResponse, LeaveRequestResponse

SEARCH_LIMIT = 20

EmployeeId = Annotated[
    uuid.UUID,
    Field(description="The employee's `id` from search_employees, never a name or code."),
]


def _profile(employee: EmployeeModel) -> dict[str, Any]:
    """A profile with the department as its name instead of an ID."""
    profile = ProfileResponse.model_validate(employee)
    data = profile.model_dump(mode="json", exclude={"department_id"})
    data["department"] = profile.department.name
    return data


def build_hr_tools(user: CurrentUser, services: ToolServices) -> list[BaseTool]:
    @tool
    async def search_employees(
        name: Annotated[
            str | None,
            Field(max_length=100, description='Part of the name, e.g. "ravi". Any case.'),
        ] = None,
        department: Annotated[
            str | None,
            Field(
                max_length=100,
                description='The exact department name, e.g. "Engineering". Any case.',
            ),
        ] = None,
    ) -> str:
        """Find active employees by part of their name, by department, or both,
        sorted by name. Returns each one's `id`, which get_employee_profile and
        get_employee_leave_summary need, with employee code, email, job title,
        role and department. Use it first whenever HR asks about a person by
        name. If several match, ask which one the user means."""
        found = await services.employees.search(user, name, department, SEARCH_LIMIT)
        rows = [
            {
                "id": str(e.id),
                "employee_code": e.employee_code,
                "name": e.name,
                "email": e.email,
                "job_title": e.job_title,
                "role": e.role.value,
                "department": e.department.name,
            }
            for e in found
        ]
        result: dict[str, object] = {"employees": rows}
        if not rows:
            result["note"] = (
                "No active employees matched. The department must be its exact name; "
                "try part of the person's name instead."
            )
        elif len(rows) == SEARCH_LIMIT:
            result["note"] = f"Showing the first {SEARCH_LIMIT}. Narrow the search for more."
        return json.dumps(result)

    @tool
    async def get_employee_profile(employee_id: EmployeeId) -> str:
        """Get one employee's full profile: name, employee code, email, job
        title, role, department, joining date, whether they're still active,
        and their `manager_id` (pass it to this tool again to see the
        manager). Works for employees who have left, too."""
        return json.dumps(_profile(await services.employees.get_employee(user, employee_id)))

    @tool
    async def get_employee_leave_summary(
        employee_id: EmployeeId,
        year: Year = None,
    ) -> str:
        """Get one employee's leave balances for a year (allotted, used,
        pending approval and available, per leave type) and their 10 most
        recent leave requests of any year, latest start date first, with
        status and decision comment (times in company local time)."""
        summary = await services.leave.get_employee_summary(user, employee_id, year)
        requests = []
        for request in summary.recent_requests:
            row = LeaveRequestResponse.model_validate(request).model_dump(
                mode="json", exclude={"approver_id"}
            )
            row["created_at"] = company_time(request.created_at)
            row["decided_at"] = company_time(request.decided_at)
            requests.append(row)
        result: dict[str, object] = {
            "employee": {"name": summary.employee.name, "id": str(summary.employee.id)},
            "year": summary.year,
            "balances": [
                LeaveBalanceResponse.model_validate(b).model_dump(mode="json", exclude={"year"})
                for b in summary.balances
            ],
            "recent_requests": requests,
        }
        if not summary.balances:
            result["note"] = "No leave balances on record for this year."
        return json.dumps(result)

    return [search_employees, get_employee_profile, get_employee_leave_summary]
