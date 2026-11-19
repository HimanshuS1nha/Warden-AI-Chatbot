import uuid
from datetime import date

import pytest
from langchain_core.messages import HumanMessage

from tests.agent_fakes import script
from warden.core.dependencies import CurrentUser
from warden.graph.graph import build_graph
from warden.graph.prompts import build_system_prompt
from warden.models import Role

THURSDAY = date(2026, 11, 19)


def user(role: Role) -> CurrentUser:
    return CurrentUser(
        employee_id=uuid.uuid4(),
        role=role,
        manager_id=None,
        department_id=uuid.uuid4(),
        name="Asha Rao",
    )


def test_prompt_has_the_date_with_weekday_and_the_user() -> None:
    prompt = build_system_prompt(user(Role.EMPLOYEE), THURSDAY)

    # The weekday lets the model work out "last Friday" without a tool call.
    assert "Today is Thursday, 19 November 2026 (timezone Asia/Kolkata)." in prompt
    assert "You are talking to Asha Rao." in prompt


@pytest.mark.parametrize(
    ("role", "says"),
    [
        (Role.EMPLOYEE, "their own data only"),
        (Role.MANAGER, "direct reports"),
        (Role.HR, "search_employees"),
    ],
)
def test_prompt_describes_the_users_role(role: Role, says: str) -> None:
    assert says in build_system_prompt(user(role), THURSDAY)


def test_prompt_has_the_rules() -> None:
    prompt = build_system_prompt(user(Role.EMPLOYEE), THURSDAY)

    for rule in ("cite every policy claim", "<untrusted_document>", "never guess", "confirmed"):
        assert rule in prompt


async def test_the_model_gets_the_prompt_first_on_every_call() -> None:
    model = script("Hi Asha.")
    prompt = build_system_prompt(user(Role.MANAGER), THURSDAY)

    await build_graph(model, [], prompt).ainvoke({"messages": [HumanMessage(content="Hi")]})

    [model_call] = model.calls
    assert model_call.system_prompt == prompt
    assert model_call.messages[1].content == "Hi"
