import json
import uuid
from datetime import date
from types import SimpleNamespace
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.agent_fakes import call, script
from warden.core.dependencies import CurrentUser
from warden.graph.graph import build_graph
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import Role

USER = CurrentUser(
    employee_id=uuid.uuid4(),
    role=Role.EMPLOYEE,
    manager_id=uuid.uuid4(),
    department_id=uuid.uuid4(),
    name="Asha Rao",
)


class FakeEmployeeService:
    def __init__(self) -> None:
        self.asked_for: list[CurrentUser] = []

    async def get_profile(self, user: CurrentUser) -> SimpleNamespace:
        self.asked_for.append(user)
        return SimpleNamespace(
            id=user.employee_id,
            employee_code="EMP042",
            name=user.name,
            email="asha.rao@example.com",
            role=user.role,
            department_id=user.department_id,
            manager_id=user.manager_id,
            job_title="Software Engineer",
            joining_date=date(2023, 4, 3),
            department=SimpleNamespace(id=user.department_id, name="Engineering"),
            is_active=True,
        )


def profile_tools(service: FakeEmployeeService):  # type: ignore[no-untyped-def]
    services = SimpleNamespace(employees=service, leave=None)
    return build_self_service_tools(USER, cast(ToolServices, services))


async def test_answers_without_tools_when_none_are_needed() -> None:
    graph = build_graph(script("Hello! How can I help?"), profile_tools(FakeEmployeeService()))

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})

    assert [m.content for m in result["messages"]] == ["Hi", "Hello! How can I help?"]


async def test_runs_the_requested_tool_and_answers_from_its_result() -> None:
    service = FakeEmployeeService()
    model = script(call("get_my_profile"), "You're a Software Engineer in Engineering.")
    graph = build_graph(model, profile_tools(service))

    result = await graph.ainvoke({"messages": [HumanMessage(content="What's my job title?")]})

    question, request, tool_result, answer = result["messages"]
    assert isinstance(request, AIMessage) and request.tool_calls[0]["name"] == "get_my_profile"
    assert isinstance(tool_result, ToolMessage)
    assert json.loads(str(tool_result.content)) == {
        "employee_code": "EMP042",
        "name": "Asha Rao",
        "email": "asha.rao@example.com",
        "role": "employee",
        "job_title": "Software Engineer",
        "joining_date": "2023-04-03",
        "department": "Engineering",
        "is_active": True,
    }
    assert answer.content == "You're a Software Engineer in Engineering."
    # The tool read the session's user, not anyone the model named.
    assert service.asked_for == [USER]
    assert "get_my_profile" in model.bound_tools


# Field names that would let the model pick whose data a tool reads.
IDENTITY_ARGUMENTS = {"user_id", "employee_id", "role", "manager_id"}


def test_no_self_service_tool_takes_an_identity_argument() -> None:
    for t in profile_tools(FakeEmployeeService()):
        assert not IDENTITY_ARGUMENTS & t.args.keys(), t.name


async def test_invalid_arguments_go_back_to_the_model_as_an_error() -> None:
    model = script(call("get_my_leave_balance", year=1900), "Sorry, that year isn't valid.")
    graph = build_graph(model, profile_tools(FakeEmployeeService()))

    result = await graph.ainvoke({"messages": [HumanMessage(content="Balance for 1900?")]})

    tool_result = result["messages"][2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert "year" in str(tool_result.content)
