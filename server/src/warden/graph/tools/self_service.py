"""Tools that read the signed-in employee's own data. Every role gets them.

Results are JSON strings. Internal UUIDs are left out because they mean
nothing to the user, except leave request IDs, which the model needs to
cancel a request later.
"""

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from warden.core.dependencies import CurrentUser
from warden.graph.tools.services import ToolServices
from warden.models import LeaveStatus
from warden.schemas.employee import EmployeeSummaryResponse, ProfileResponse
from warden.schemas.leave import LeaveBalanceResponse, LeaveRequestResponse

Year = Annotated[
    int | None,
    Field(ge=2000, le=2100, description="Calendar year, e.g. 2026. Leave out for this year."),
]


def build_self_service_tools(user: CurrentUser, services: ToolServices) -> list[BaseTool]:
    @tool
    async def get_my_profile() -> str:
        """Get the signed-in employee's own profile: name, employee code, email,
        job title, role, department, joining date and whether they're active.
        Use it for any question about who the user is or their job details."""
        profile = ProfileResponse.model_validate(await services.employees.get_profile(user))
        data = profile.model_dump(mode="json", exclude={"id", "department_id", "manager_id"})
        data["department"] = profile.department.name
        return json.dumps(data)

    @tool
    async def get_my_manager() -> str:
        """Get the signed-in employee's manager: name, employee code, email and
        job title. Use it for questions like "who is my manager" or "who
        approves my leave"."""
        manager = await services.employees.get_manager(user)
        if manager is None:
            return json.dumps({"manager": None, "note": "The user has no manager."})
        summary = EmployeeSummaryResponse.model_validate(manager)
        return json.dumps({"manager": summary.model_dump(mode="json", exclude={"id"})})

    @tool
    async def get_my_leave_balance(year: Year = None) -> str:
        """Get the signed-in employee's leave balances for a year, one per leave
        type (casual, sick, earned): days allotted, used, pending approval and
        available. Available is what a new request can use."""
        year = year or services.leave.today().year
        balances = await services.leave.get_balances(user, year)
        rows = [
            LeaveBalanceResponse.model_validate(b).model_dump(mode="json", exclude={"year"})
            for b in balances
        ]
        result: dict[str, object] = {"year": year, "balances": rows}
        if not rows:
            result["note"] = "No leave balances on record for this year."
        return json.dumps(result)

    @tool
    async def get_my_leave_history(
        status: Annotated[
            LeaveStatus | None,
            Field(description="Only requests with this status. Leave out for all."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="How many, newest first.")] = 10,
    ) -> str:
        """List the signed-in employee's own leave requests, newest first: type,
        dates, working days, reason, status, and the manager's decision comment.
        Each has an `id`, needed to cancel that request."""
        requests = await services.leave.list_my_requests(user, status, limit)
        rows = [
            LeaveRequestResponse.model_validate(r).model_dump(mode="json", exclude={"approver_id"})
            for r in requests
        ]
        return json.dumps({"requests": rows})

    return [get_my_profile, get_my_manager, get_my_leave_balance, get_my_leave_history]
