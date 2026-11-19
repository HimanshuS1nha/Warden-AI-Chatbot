import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import ChatMessageModel, ChatThreadModel


class ChatRepository:
    """Chat threads and their visible messages. IDs are UUIDv7, so ordering
    by id is ordering by creation time, and `before` is a cursor: the id of
    the last row of the previous page."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_thread(
        self, thread_id: uuid.UUID, employee_id: uuid.UUID
    ) -> ChatThreadModel | None:
        """The thread if it belongs to the employee, else None."""
        return await self.db.scalar(
            select(ChatThreadModel).where(
                ChatThreadModel.id == thread_id, ChatThreadModel.employee_id == employee_id
            )
        )

    async def list_threads(
        self, employee_id: uuid.UUID, before: uuid.UUID | None, limit: int
    ) -> Sequence[ChatThreadModel]:
        """The employee's threads, newest first."""
        query = (
            select(ChatThreadModel)
            .where(ChatThreadModel.employee_id == employee_id)
            .order_by(ChatThreadModel.id.desc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(ChatThreadModel.id < before)
        return (await self.db.scalars(query)).all()

    async def list_messages(
        self, thread_id: uuid.UUID, before: uuid.UUID | None, limit: int
    ) -> Sequence[ChatMessageModel]:
        """The thread's messages, newest first. Ownership is the caller's
        check: load the thread with get_thread first."""
        query = (
            select(ChatMessageModel)
            .where(ChatMessageModel.thread_id == thread_id)
            .order_by(ChatMessageModel.id.desc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(ChatMessageModel.id < before)
        return (await self.db.scalars(query)).all()
