"""Fixtures for tests that need Postgres.

The schema is built once per run with `alembic upgrade head`, so tests run
against the real migrations. Each test runs inside one transaction that is
rolled back at the end. The app's own `commit()` calls become savepoint
releases (join_transaction_mode="create_savepoint"), so nothing is ever
written for real and tests don't see each other's rows.
"""

import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command
from tests.factories import PASSWORD, MakeEmployee
from warden.core.database import get_db
from warden.core.hashing import password_hash
from warden.main import app
from warden.models import DepartmentModel, EmployeeModel, Role

SERVER_DIR = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(item.path).is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.integration)


async def _create_database_if_missing(url: str) -> None:
    target = make_url(url)
    dsn = target.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    conn = await asyncpg.connect(dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", target.database
        )
        if exists:
            return
        try:
            await conn.execute(f'CREATE DATABASE "{target.database}"')
        except asyncpg.InsufficientPrivilegeError:
            raise RuntimeError(
                f"Test database {target.database!r} doesn't exist and user "
                f"{target.username!r} can't create it. Create it once with:\n"
                f"  sudo -u postgres createdb -O {target.username} {target.database}\n"
                "or point TEST_DATABASE_URL at a database you can use."
            ) from None
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ["TEST_DATABASE_URL"]
    asyncio.run(_create_database_if_missing(url))
    # Alembic's env.py reads DATABASE_URL, which tests/conftest.py set to the
    # test database.
    command.upgrade(Config(str(SERVER_DIR / "alembic.ini")), "head")
    return url


@pytest.fixture(scope="session")
def engine(database_url: str) -> AsyncEngine:
    # NullPool: each test gets its own event loop, and pooled asyncpg
    # connections can't move between loops.
    return create_async_engine(database_url, poolclass=NullPool)


@pytest.fixture
async def connection(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    async with engine.connect() as conn:
        transaction = await conn.begin()
        yield conn
        await transaction.rollback()


def _session(connection: AsyncConnection) -> AsyncSession:
    return AsyncSession(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )


@pytest.fixture
async def db(connection: AsyncConnection) -> AsyncIterator[AsyncSession]:
    async with _session(connection) as session:
        yield session


@pytest.fixture
async def client(connection: AsyncConnection) -> AsyncIterator[AsyncClient]:
    # A new session per request, like production, all on the test's connection.
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with _session(connection) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def hashed_password() -> str:
    # Argon2 is slow on purpose; hash once for the whole run.
    return password_hash.hash(PASSWORD)


@pytest.fixture
def make_employee(db: AsyncSession, hashed_password: str) -> MakeEmployee:
    counter = itertools.count(1)
    department: DepartmentModel | None = None

    async def _make(**overrides: Any) -> EmployeeModel:
        nonlocal department
        if department is None:
            department = DepartmentModel(name="Engineering")
            db.add(department)
            await db.flush()

        n = next(counter)
        fields: dict[str, Any] = {
            "employee_code": f"WRD{n:04}",
            "name": f"Employee {n}",
            "email": f"employee{n}@warden.demo",
            "password_hash": hashed_password,
            "role": Role.EMPLOYEE,
            "department_id": department.id,
            "job_title": "Engineer",
            "joining_date": date(2024, 1, 1),
        } | overrides
        employee = EmployeeModel(**fields)
        db.add(employee)
        await db.commit()
        return employee

    return _make
