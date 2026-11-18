from collections.abc import Awaitable, Callable

from warden.core.dependencies import CurrentUser
from warden.models import EmployeeModel

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
