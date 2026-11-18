import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from warden.core.database import get_db
from warden.core.dependencies import CurrentUser, get_current_user, require_role
from warden.core.exceptions import register_exception_handlers
from warden.models import AuditLogModel, AuditOutcome, Role

app = FastAPI()
register_exception_handlers(app)

HRUser = Annotated[CurrentUser, Depends(require_role(Role.HR))]


@app.get("/hr")
async def hr_only(user: HRUser) -> dict[str, str]:
    return {"employee_id": str(user.employee_id)}


@app.get("/managers-and-hr", dependencies=[Depends(require_role(Role.MANAGER, Role.HR))])
async def managers_and_hr() -> None:
    pass


guarded_router = APIRouter(prefix="/admin", dependencies=[Depends(require_role(Role.HR))])


@guarded_router.get("/ping")
async def admin_ping() -> None:
    pass


app.include_router(guarded_router)


def make_user(role: Role) -> CurrentUser:
    return CurrentUser(
        employee_id=uuid.uuid4(),
        role=role,
        manager_id=None,
        department_id=uuid.uuid4(),
        name="Test User",
    )


class FakeSession:
    """Stands in for the database. get_current_user is overridden or rejects
    a missing cookie first, so only the audit row on a 403 reaches it."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commits = 0

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
async def client(session: FakeSession) -> AsyncIterator[AsyncClient]:
    async def fake_db() -> AsyncIterator[FakeSession]:
        yield session

    app.dependency_overrides[get_db] = fake_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def log_in_as(user: CurrentUser) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


@pytest.mark.parametrize(
    ("path", "role", "expected"),
    [
        ("/hr", Role.HR, 200),
        ("/hr", Role.MANAGER, 403),
        ("/hr", Role.EMPLOYEE, 403),
        ("/managers-and-hr", Role.HR, 200),
        ("/managers-and-hr", Role.MANAGER, 200),
        ("/managers-and-hr", Role.EMPLOYEE, 403),
        ("/admin/ping", Role.HR, 200),
        ("/admin/ping", Role.MANAGER, 403),
        ("/admin/ping", Role.EMPLOYEE, 403),
    ],
)
async def test_only_listed_roles_get_through(
    client: AsyncClient, session: FakeSession, path: str, role: Role, expected: int
) -> None:
    log_in_as(make_user(role))

    response = await client.get(path)

    assert response.status_code == expected
    if expected == 403:
        assert response.json() == {"message": "You don't have permission to do this"}
    # Only refusals are audited, and each one is committed before the 403.
    assert len(session.added) == session.commits == (1 if expected == 403 else 0)


async def test_refusal_is_audited_with_the_request_and_roles(
    client: AsyncClient, session: FakeSession
) -> None:
    user = make_user(Role.EMPLOYEE)
    log_in_as(user)

    await client.get("/managers-and-hr")

    [entry] = session.added
    assert isinstance(entry, AuditLogModel)
    assert entry.action == "authz.forbidden"
    assert entry.outcome == AuditOutcome.DENIED
    assert entry.actor_id == user.employee_id
    assert entry.details == {
        "method": "GET",
        "path": "/managers-and-hr",
        "role": "employee",
        "required": ["hr", "manager"],
    }


async def test_returns_the_current_user(client: AsyncClient) -> None:
    user = make_user(Role.HR)
    log_in_as(user)

    response = await client.get("/hr")

    assert response.json() == {"employee_id": str(user.employee_id)}


@pytest.mark.parametrize("path", ["/hr", "/managers-and-hr", "/admin/ping"])
async def test_no_session_is_401_not_403(
    client: AsyncClient, session: FakeSession, path: str
) -> None:
    response = await client.get(path)

    assert response.status_code == 401
    assert session.added == []


def test_needs_at_least_one_role() -> None:
    with pytest.raises(ValueError, match="at least one role"):
        require_role()


def test_rejects_unknown_role_names() -> None:
    # A typo like "admin" fails at import time instead of locking everyone out.
    with pytest.raises(ValueError):
        require_role("admin")  # type: ignore[arg-type]
