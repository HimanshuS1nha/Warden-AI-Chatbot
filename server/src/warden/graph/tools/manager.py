"""Tools that read a manager's direct reports. Only managers get them.

The services check the role again and scope every query to the session
user's own reports, so even a tool given to the wrong role can't read
another team.

Results are JSON strings. Reports are shown by name and employee code, not
by UUID. Leave request IDs are kept: approving or rejecting a request needs
them.
"""

import datetime
import json
from collections import Counter
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from warden.core.dependencies import CurrentUser
from warden.graph.tools.common import company_time
from warden.graph.tools.services import ToolServices
from warden.models import EmployeeModel, LeaveStatus
from warden.schemas.employee import EmployeeSummaryResponse
from warden.schemas.leave import LeaveRequestResponse

NOT_RECORDED = "not_recorded"
NO_REPORTS = "The user has no active direct reports."


def _who(employee: EmployeeModel) -> dict[str, str | None]:
    """How a report is named in team results."""
    return {"name": employee.name, "employee_code": employee.employee_code}


def build_manager_tools(user: CurrentUser, services: ToolServices) -> list[BaseTool]:
    @tool
    async def list_my_team() -> str:
        """List the signed-in manager's active direct reports, by name: employee
        code, email and job title. Use it for questions like "who is on my
        team" or "how many people report to me"."""
        team = [
            EmployeeSummaryResponse.model_validate(e).model_dump(mode="json", exclude={"id"})
            for e in await services.employees.list_team(user)
        ]
        result: dict[str, object] = {"team": team}
        if not team:
            result["note"] = NO_REPORTS
        return json.dumps(result)

    @tool
    async def get_team_leave_requests(
        status: Annotated[
            LeaveStatus | None,
            Field(
                description='Only requests with this status, e.g. "pending" for the ones '
                "waiting on the manager. Leave out for all."
            ),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=50, description="How many, latest start date first.")
        ] = 20,
    ) -> str:
        """List leave requests from the signed-in manager's direct reports,
        latest start date first: who asked, type, dates, working days, reason,
        status, the decision comment, and when it was made and decided (company
        local time). Each has an `id`, needed to approve or reject that
        request."""
        rows = []
        for request in await services.leave.list_team_requests(user, status, limit):
            row: dict[str, Any] = {"employee": _who(request.employee)}
            row |= LeaveRequestResponse.model_validate(request).model_dump(
                mode="json", exclude={"approver_id"}
            )
            row["created_at"] = company_time(request.created_at)
            row["decided_at"] = company_time(request.decided_at)
            rows.append(row)
        result: dict[str, object] = {"requests": rows}
        if not rows:
            result["note"] = "No leave requests from the team" + (
                f" with status {status.value}." if status else "."
            )
        return json.dumps(result)

    @tool
    async def get_team_attendance(
        date: Annotated[
            datetime.date | None,
            Field(description="The day, as YYYY-MM-DD. Leave out for today."),
        ] = None,
    ) -> str:
        """Get each of the signed-in manager's direct reports' attendance on
        one day: status (present, absent, wfh, on_leave, holiday, or
        not_recorded) with check-in and check-out times in company local time,
        plus a count of people per status. Use it for questions like "who is
        working from home today". For several days, call it once per day."""
        day = date or services.today()
        members = []
        for row in await services.attendance.get_team_for_date(user, day):
            record = row.attendance
            members.append(
                _who(row.employee)
                | {
                    "status": record.status.value if record else NOT_RECORDED,
                    "check_in": company_time(record.check_in) if record else None,
                    "check_out": company_time(record.check_out) if record else None,
                }
            )
        result: dict[str, object] = {
            "date": day.isoformat(),
            "people_by_status": Counter(m["status"] for m in members),
            "members": members,
        }
        if not members:
            result["note"] = NO_REPORTS
        return json.dumps(result)

    return [list_my_team, get_team_leave_requests, get_team_attendance]
