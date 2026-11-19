from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection

from warden import models  # noqa: F401  registers every table on Base.metadata
from warden.core.database import Base
from warden.graph.checkpointer import alembic_manages


async def test_migrations_match_the_models(connection: AsyncConnection) -> None:
    """Fails when a model changed without a migration, or the other way round."""

    def diff(sync_connection: Connection) -> list[object]:
        context = MigrationContext.configure(
            sync_connection, opts={"include_name": alembic_manages}
        )
        return compare_metadata(context, Base.metadata)

    assert await connection.run_sync(diff) == []
