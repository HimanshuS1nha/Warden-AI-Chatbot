import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from warden.models import ChatFeedbackModel, ChatMessageModel, ChatThreadModel


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
        """The thread's messages, newest first, each with its pending action
        (one extra query for the page). Ownership is the caller's check: load
        the thread with get_thread first."""
        query = (
            select(ChatMessageModel)
            .options(selectinload(ChatMessageModel.pending_action))
            .where(ChatMessageModel.thread_id == thread_id)
            .order_by(ChatMessageModel.id.desc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(ChatMessageModel.id < before)
        return (await self.db.scalars(query)).all()

    async def get_message(
        self, message_id: uuid.UUID, employee_id: uuid.UUID
    ) -> ChatMessageModel | None:
        """The message if it's in one of the employee's threads, else None."""
        return await self.db.scalar(
            select(ChatMessageModel)
            .join(ChatThreadModel, ChatThreadModel.id == ChatMessageModel.thread_id)
            .where(ChatMessageModel.id == message_id, ChatThreadModel.employee_id == employee_id)
        )

    # Feedback

    async def feedback_for(
        self, employee_id: uuid.UUID, message_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ChatFeedbackModel]:
        """The employee's feedback on these messages, by message id."""
        if not message_ids:
            return {}
        rows = await self.db.scalars(
            select(ChatFeedbackModel).where(
                ChatFeedbackModel.employee_id == employee_id,
                ChatFeedbackModel.message_id.in_(message_ids),
            )
        )
        return {row.message_id: row for row in rows}

    async def upsert_feedback(
        self, employee_id: uuid.UUID, message_id: uuid.UUID, rating: int, comment: str | None
    ) -> ChatFeedbackModel:
        """Set the employee's rating of a message, replacing any earlier one.
        Doesn't commit."""
        statement = (
            insert(ChatFeedbackModel)
            .values(employee_id=employee_id, message_id=message_id, rating=rating, comment=comment)
            .on_conflict_do_update(
                index_elements=[ChatFeedbackModel.employee_id, ChatFeedbackModel.message_id],
                set_={"rating": rating, "comment": comment},
            )
            .returning(ChatFeedbackModel)
        )
        # An upsert always returns its row.
        rows = await self.db.scalars(statement, execution_options={"populate_existing": True})
        return rows.one()

    async def delete_feedback(self, employee_id: uuid.UUID, message_id: uuid.UUID) -> None:
        """Remove the employee's rating of a message, if any. Doesn't commit."""
        await self.db.execute(
            delete(ChatFeedbackModel).where(
                ChatFeedbackModel.employee_id == employee_id,
                ChatFeedbackModel.message_id == message_id,
            )
        )
