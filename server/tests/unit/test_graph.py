import json
import uuid
from datetime import date
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from tests.agent_fakes import PROMPT, ScriptedChatModel, call, script
from warden.core.dependencies import CurrentUser
from warden.graph.agent import GAVE_UP, MAX_TOOL_CALLS, tool_calls_this_turn
from warden.graph.graph import OVER_LIMIT, build_graph
from warden.graph.prompts import TOOL_LIMIT_NOTE
from warden.graph.tools.self_service import build_self_service_tools
from warden.graph.tools.services import ToolServices
from warden.models import Role
from warden.services.errors import InsufficientBalance, NotFound

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
    graph = build_graph(
        script("Hello! How can I help?"), profile_tools(FakeEmployeeService()), PROMPT
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})

    assert [m.content for m in result["messages"]] == ["Hi", "Hello! How can I help?"]


async def test_runs_the_requested_tool_and_answers_from_its_result() -> None:
    service = FakeEmployeeService()
    model = script(call("get_my_profile"), "You're a Software Engineer in Engineering.")
    graph = build_graph(model, profile_tools(service), PROMPT)

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
async def test_invalid_arguments_go_back_to_the_model_as_an_error() -> None:
    model = script(call("get_my_leave_balance", year=1900), "Sorry, that year isn't valid.")
    graph = build_graph(model, profile_tools(FakeEmployeeService()), PROMPT)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Balance for 1900?")]})

    tool_result = result["messages"][2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert "year" in str(tool_result.content)


@tool
async def look_up(kind: str) -> str:
    """Fails the way services do."""
    if kind == "missing":
        raise NotFound("Employee not found.")
    if kind == "balance":
        raise InsufficientBalance("Not enough casual leave.", available=2, asked=date(2026, 6, 8))
    raise RuntimeError("a bug")


async def test_service_errors_go_back_to_the_model_with_their_code() -> None:
    model = script(
        call("look_up", "call_1", kind="missing"),
        call("look_up", "call_2", kind="balance"),
        "Done.",
    )
    graph = build_graph(model, [look_up], PROMPT)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Look them up")]})

    missing, balance = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert missing.status == "error"
    assert json.loads(str(missing.content)) == {
        "error": "not_found",
        "message": "Employee not found.",
    }
    assert json.loads(str(balance.content)) == {
        "error": "insufficient_balance",
        "message": "Not enough casual leave.",
        "available": 2,
        "asked": "2026-06-08",
    }
    assert result["messages"][-1].content == "Done."


async def test_other_exceptions_are_bugs_and_still_raise() -> None:
    graph = build_graph(script(call("look_up", kind="other"), "unreachable"), [look_up], PROMPT)

    with pytest.raises(RuntimeError, match="a bug"):
        await graph.ainvoke({"messages": [HumanMessage(content="Look it up")]})


# Tool call limit

pings: list[int] = []


@tool
async def ping(n: int) -> str:
    """Records that it ran."""
    pings.append(n)
    return "pong"


@pytest.fixture(autouse=True)
def clear_pings() -> None:
    pings.clear()


def ask(*calls: int) -> AIMessage:
    """A model reply asking for one ping per number, all at once."""
    return AIMessage(content="", tool_calls=[call("ping", f"call_{n}", n=n) for n in calls])


async def test_at_the_limit_the_model_must_answer_without_tools() -> None:
    replies = [ask(n) for n in range(MAX_TOOL_CALLS)] + [AIMessage(content="Here's what I found.")]
    model = ScriptedChatModel(messages=iter(replies))

    result = await build_graph(model, [ping], PROMPT).ainvoke(
        {"messages": [HumanMessage(content="Keep pinging")]}
    )

    assert pings == list(range(MAX_TOOL_CALLS))
    assert [c.tool_choice for c in model.calls] == [None] * MAX_TOOL_CALLS + ["none"]
    assert TOOL_LIMIT_NOTE in model.calls[-1].system_prompt
    assert TOOL_LIMIT_NOTE not in model.calls[0].system_prompt
    assert result["messages"][-1].content == "Here's what I found."


async def test_calls_past_the_limit_in_one_reply_are_refused() -> None:
    too_many = ask(*range(MAX_TOOL_CALLS + 2))
    model = ScriptedChatModel(messages=iter([too_many, AIMessage(content="Done.")]))

    result = await build_graph(model, [ping], PROMPT).ainvoke(
        {"messages": [HumanMessage(content="Ping a lot")]}
    )

    assert sorted(pings) == list(range(MAX_TOOL_CALLS))
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    # Every call gets an answer, so the model's history stays valid.
    assert len(tool_results) == MAX_TOOL_CALLS + 2
    refused = [m for m in tool_results if m.content == OVER_LIMIT]
    assert [m.tool_call_id for m in refused] == [
        f"call_{MAX_TOOL_CALLS}",
        f"call_{MAX_TOOL_CALLS + 1}",
    ]
    assert model.calls[-1].tool_choice == "none"


async def test_a_model_that_ignores_the_limit_gets_a_fixed_answer() -> None:
    replies = [ask(n) for n in range(MAX_TOOL_CALLS + 1)]
    model = ScriptedChatModel(messages=iter(replies))

    result = await build_graph(model, [ping], PROMPT).ainvoke(
        {"messages": [HumanMessage(content="Ping forever")]}
    )

    assert len(pings) == MAX_TOOL_CALLS
    assert result["messages"][-1].content == GAVE_UP


def test_only_this_turns_tool_calls_count() -> None:
    earlier_turn = [HumanMessage(content="First"), ask(1, 2, 3), AIMessage(content="Ok.")]
    this_turn = [HumanMessage(content="Second"), ask(4), AIMessage(content=""), ask(5, 6)]

    assert tool_calls_this_turn(earlier_turn + this_turn) == 3
    assert tool_calls_this_turn(earlier_turn + [HumanMessage(content="Third")]) == 0
