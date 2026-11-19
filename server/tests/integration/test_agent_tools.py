"""Agent tools against the real services and database."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee, as_user
from warden.graph.tools.self_service import build_self_service_tools
from warden.services.employee import EmployeeService


async def test_get_my_profile_returns_the_session_users_profile(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await make_employee()  # Someone else, who must not show up.
    [get_my_profile] = build_self_service_tools(as_user(me), EmployeeService(db))

    profile = json.loads(await get_my_profile.ainvoke({}))

    assert profile["employee_code"] == me.employee_code
    assert profile["email"] == me.email
    assert isinstance(profile["department"], str)
    assert not {"id", "department_id", "manager_id"} & profile.keys()
