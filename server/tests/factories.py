from collections.abc import Awaitable, Callable

from warden.models import EmployeeModel

# Every employee made by the `make_employee` fixture has this password.
PASSWORD = "correct-horse-battery-staple"

MakeEmployee = Callable[..., Awaitable[EmployeeModel]]
