"""Which tools each role gets, and that none lets the model pick whose data it
reads. The expected lists are the tools in plan.md's "Tools" table."""

import re
import uuid
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.tools import BaseTool

from warden.core.dependencies import CurrentUser
from warden.graph.tools import ToolServices, build_tools
from warden.models import Role
from warden.rag.retrievers import Retriever

EVERYONE = {
    "search_policies",
    "get_my_profile",
    "get_my_manager",
    "get_my_leave_balance",
    "get_my_leave_history",
    "get_my_attendance",
    "get_my_payslips",
    "get_company_holidays",
    "apply_for_leave",
}
MANAGER_ONLY = {"list_my_team", "get_team_leave_requests", "get_team_attendance"}
HR_ONLY = {"search_employees", "get_employee_profile", "get_employee_leave_summary"}

EXPECTED = {
    Role.EMPLOYEE: EVERYONE,
    Role.MANAGER: EVERYONE | MANAGER_ONLY,
    Role.HR: EVERYONE | HR_ONLY,
}

# Argument names that would pick a person or a permission: user_id,
# employeeId, manager_code, staff_email, role, as_role and so on.
IDENTITY_ARGUMENT = re.compile(
    r"(user|employee|manager|person|staff|member)_?(id|code|email|name)$|(^|_)role$",
    re.IGNORECASE,
)
# HR's lookup tools take the ID of the person HR asks about. The service
# checks the caller is HR and audits every read.
ALLOWED_IDENTITY_ARGUMENTS = {
    "get_employee_profile": {"employee_id"},
    "get_employee_leave_summary": {"employee_id"},
}

# Building tools touches neither, so stand-ins are enough.
SERVICES = cast(ToolServices, SimpleNamespace())
RETRIEVER = cast(Retriever, object())


def user(role: Role) -> CurrentUser:
    return CurrentUser(
        employee_id=uuid.uuid4(),
        role=role,
        manager_id=None,
        department_id=uuid.uuid4(),
        name="Test User",
    )


def tools_for(role: Role, retriever: Retriever | None = RETRIEVER) -> list[BaseTool]:
    return build_tools(user(role), SERVICES, retriever, uuid.uuid4())


@pytest.mark.parametrize("role", list(Role))
def test_each_role_gets_the_tools_in_the_plan(role: Role) -> None:
    names = [t.name for t in tools_for(role)]

    assert set(names) == EXPECTED[role]
    assert len(names) == len(set(names)), "a tool is listed twice"


@pytest.mark.parametrize("role", list(Role))
def test_no_tool_takes_an_identity_argument(role: Role) -> None:
    for t in tools_for(role):
        allowed = ALLOWED_IDENTITY_ARGUMENTS.get(t.name, set())
        identity = {arg for arg in t.args if IDENTITY_ARGUMENT.search(arg)}
        assert identity <= allowed, f"{t.name} takes {identity - allowed}"


def test_the_pattern_catches_identity_arguments() -> None:
    caught = ["user_id", "employeeId", "employee_code", "manager_id", "role", "as_role"]
    let_through = ["query", "year", "status", "request_id", "name", "department"]

    assert all(IDENTITY_ARGUMENT.search(arg) for arg in caught)
    assert not any(IDENTITY_ARGUMENT.search(arg) for arg in let_through)


@pytest.mark.parametrize("role", list(Role))
def test_without_a_retriever_there_is_no_policy_search(role: Role) -> None:
    names = {t.name for t in tools_for(role, retriever=None)}

    assert names == EXPECTED[role] - {"search_policies"}
