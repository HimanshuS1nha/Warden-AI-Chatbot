"""The agent's memory: LangGraph checkpoints stored in Postgres.

Each thread's full state (messages, tool calls and their results) is saved
after every step, so a conversation continues where it stopped, also after a
restart or on another worker.

The checkpointer uses psycopg, not the app's SQLAlchemy engine (asyncpg), so
it has its own pool and its own URL: CHECKPOINTER_DATABASE_URL.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

# Created and migrated by AsyncPostgresSaver.setup(), not by Alembic. Alembic
# skips them (see `alembic_manages`), so autogenerate never drops them.
CHECKPOINT_TABLES = frozenset(
    {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
)

# Any constant works, as long as nothing else uses the same advisory lock.
_SETUP_LOCK_ID = 0x5741_5244  # "WARD"
_SETUP_LOCK_POLL_SECONDS = 0.2


def alembic_manages(name: str | None, type_: str, _parent_names: object) -> bool:
    """Alembic `include_name` hook that leaves the checkpoint tables alone."""
    return not (type_ == "table" and name in CHECKPOINT_TABLES)


@asynccontextmanager
async def open_checkpointer(url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """A checkpointer with its own connection pool, its tables created or
    migrated. The pool closes when the block exits."""
    pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        url,
        open=False,
        # What AsyncPostgresSaver expects from every connection: autocommit so
        # each checkpoint write commits on its own, dict rows, and no prepared
        # statements, which break behind PgBouncer in transaction mode.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    # Fail at startup when the database is unreachable, instead of on the
    # first chat message.
    await pool.open(wait=True, timeout=10)
    try:
        checkpointer = AsyncPostgresSaver(pool)
        # Each worker runs this at startup. The lock stops two workers from
        # running the migrations at the same time on an empty database.
        async with pool.connection() as conn:
            await _wait_for_setup_lock(conn)
            try:
                await checkpointer.setup()
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK_ID,))
        yield checkpointer
    finally:
        await pool.close()


async def _wait_for_setup_lock(conn: AsyncConnection[DictRow]) -> None:
    # Polls instead of a blocking pg_advisory_lock. The migrations include
    # CREATE INDEX CONCURRENTLY, which waits for every running query to end,
    # including a worker blocked in pg_advisory_lock waiting for this same
    # lock: both would wait forever. Between polls this connection runs nothing.
    while True:
        cursor = await conn.execute("SELECT pg_try_advisory_lock(%s) AS locked", (_SETUP_LOCK_ID,))
        row = await cursor.fetchone()
        if row is not None and row["locked"]:
            return
        await asyncio.sleep(_SETUP_LOCK_POLL_SECONDS)
