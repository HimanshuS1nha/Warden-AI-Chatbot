import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.cookies import ACCESS_TOKEN_COOKIE
from warden.core.database import get_db
from warden.core.jwt import decode_token
from warden.models import EmployeeModel, EmployeeSessionModel, Role

DbSession = Annotated[AsyncSession, Depends(get_db)]


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated employee for this request.

    Built only from the session cookie and the database. Services take it as
    their first argument and scope every query by it.
    """

    employee_id: uuid.UUID
    role: Role
    manager_id: uuid.UUID | None
    department_id: uuid.UUID
    name: str


async def get_current_user(
    db: DbSession,
    access_token: Annotated[str | None, Cookie(alias=ACCESS_TOKEN_COOKIE)] = None,
) -> CurrentUser:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )
    if not access_token:
        raise unauthorized

    payload = decode_token(access_token, "access")

    # Load the employee only if the session the token was issued for is still
    # active, so logout and revocation take effect right away instead of
    # when the access token expires. Role and manager come from this row, not
    # the token, so a role change or deactivation applies on the next request.
    employee = await db.scalar(
        select(EmployeeModel)
        .join(EmployeeSessionModel, EmployeeSessionModel.employee_id == EmployeeModel.id)
        .where(
            EmployeeModel.id == payload.employee_id,
            EmployeeModel.is_active.is_(True),
            EmployeeSessionModel.id == payload.session_id,
            EmployeeSessionModel.revoked_at.is_(None),
            EmployeeSessionModel.expires_at > datetime.now(UTC),
        )
    )
    if employee is None:
        raise unauthorized

    return CurrentUser(
        employee_id=employee.id,
        role=employee.role,
        manager_id=employee.manager_id,
        department_id=employee.department_id,
        name=employee.name,
    )


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]


def require_role(*roles: Role) -> Callable[[CurrentUser], Awaitable[CurrentUser]]:
    """Dependency that lets the request through only for the given roles.

    Returns the CurrentUser, so it can replace CurrentUserDep:

        HRUser = Annotated[CurrentUser, Depends(require_role(Role.HR))]

    or guard a whole router without using the user:

        APIRouter(dependencies=[Depends(require_role(Role.HR))])

    No session gives 401 (from get_current_user); the wrong role gives 403.
    This only checks the role. Whether a manager may touch a given employee
    or request is a scoping rule and belongs in the service.
    """
    if not roles:
        raise ValueError("require_role needs at least one role")
    allowed = frozenset(Role(role) for role in roles)

    async def check_role(user: CurrentUserDep) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to do this",
            )
        return user

    return check_role
