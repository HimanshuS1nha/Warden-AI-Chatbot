import uuid
from collections.abc import Awaitable, Callable

from warden.core.dependencies import CurrentUser
from warden.models import EmployeeModel, Role

# Every employee made by the `make_employee` fixture has this password.
PASSWORD = "correct-horse-battery-staple"

MakeEmployee = Callable[..., Awaitable[EmployeeModel]]


def as_user(employee: EmployeeModel) -> CurrentUser:
    """The CurrentUser that get_current_user would build for this employee."""
    return CurrentUser(
        employee_id=employee.id,
        role=employee.role,
        manager_id=employee.manager_id,
        department_id=employee.department_id,
        name=employee.name,
    )


def user_with_role(role: Role) -> CurrentUser:
    """A signed-in user with this role and no database row, for code that
    only looks at the role."""
    return CurrentUser(
        employee_id=uuid.uuid4(),
        role=role,
        manager_id=None,
        department_id=uuid.uuid4(),
        name=f"Test {role.value}",
    )
