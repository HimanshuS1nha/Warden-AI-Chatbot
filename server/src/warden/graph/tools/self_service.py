"""Tools that read the signed-in employee's own data. Every role gets them."""

import json

from langchain_core.tools import BaseTool, tool

from warden.core.dependencies import CurrentUser
from warden.schemas.employee import ProfileResponse
from warden.services.employee import EmployeeService

# Internal UUIDs mean nothing to the user and the model would only repeat them.
_INTERNAL_IDS = {"id", "department_id", "manager_id"}


def build_self_service_tools(user: CurrentUser, employees: EmployeeService) -> list[BaseTool]:
    @tool
    async def get_my_profile() -> str:
        """Get the signed-in employee's own profile: name, employee code, email,
        job title, role, department, joining date and whether they're active.
        Use it for any question about who the user is or their job details."""
        profile = ProfileResponse.model_validate(await employees.get_profile(user))
        data = profile.model_dump(mode="json", exclude=_INTERNAL_IDS)
        data["department"] = profile.department.name
        return json.dumps(data)

    return [get_my_profile]
