"""Tools the agent can call, built per request for the signed-in user.

Every factory takes the CurrentUser from the session and returns tools with
no identity arguments: the model can choose which tool to call, never whose
data it reads. The one exception is HR's lookup tools, whose `employee_id`
is the person HR asks about.

A role's tools are its whole permission set in the chat: a tool the role
doesn't get is one the model never sees. The services still check the role
on every call, so a wrong list here is refused rather than obeyed.
"""

from langchain_core.tools import BaseTool

from warden.core.dependencies import CurrentUser
from warden.graph.tools.hr import build_hr_tools
from warden.graph.tools.manager import build_manager_tools
from warden.graph.tools.policy import build_policy_tools
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import Role
from warden.rag.retrievers import Retriever

__all__ = ["ToolServices", "build_tools"]


def build_tools(
    user: CurrentUser, services: ToolServices, retriever: Retriever | None
) -> list[BaseTool]:
    """The tools for the user's role (plan.md, "Tools"):

    - everyone: policy search and their own data
    - managers: plus their team
    - HR: plus looking up any employee

    Without a retriever (no chunk store yet) there is no policy search; the
    rest works as usual.
    """
    tools = build_policy_tools(user, retriever) if retriever else []
    tools += build_self_service_tools(user, services)
    if user.role == Role.MANAGER:
        tools += build_manager_tools(user, services)
    elif user.role == Role.HR:
        tools += build_hr_tools(user, services)
    return tools
