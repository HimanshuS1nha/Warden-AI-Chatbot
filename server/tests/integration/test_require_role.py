from collections.abc import AsyncIterator

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from tests.factories import PASSWORD, MakeEmployee
from warden.core.database import get_db
from warden.core.dependencies import require_role
from warden.core.exceptions import register_exception_handlers
from warden.models import Role
from warden.routers import auth

# The real app has no role-guarded routes yet, so these tests mount the auth
# router next to one guarded route and log in for real.
app = FastAPI()
register_exception_handlers(app)
app.include_router(auth.router)


@app.get("/managers-only", dependencies=[Depends(require_role(Role.MANAGER))])
async def managers_only() -> None:
    pass


@pytest.fixture
async def guarded_client(connection: AsyncConnection) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with AsyncSession(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_role_change_applies_on_the_next_request(
    guarded_client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee(role=Role.MANAGER)
    login = await guarded_client.post(
        "/api/auth/login", json={"email": manager.email, "password": PASSWORD}
    )
    assert login.status_code == 200
    assert (await guarded_client.get("/managers-only")).status_code == 200

    # The role is read from the employee row on every request, not from the
    # token, so a demotion takes effect without logging out.
    manager.role = Role.EMPLOYEE
    await db.commit()

    assert (await guarded_client.get("/managers-only")).status_code == 403
