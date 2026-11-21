"""Chat threads, the messages the employee sees, and unanswered questions.

Every thread belongs to one employee. Each read goes through
`WHERE employee_id = :me`, so someone else's thread is NotFound, the same
as one that doesn't exist.

These are the visible history only. What the agent remembers (tool calls
and results) lives in the LangGraph checkpointer under the same thread id.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import (
    ChatFeedbackModel,
    ChatMessageModel,
    ChatThreadModel,
    MessageRole,
    UnansweredQueryModel,
    UnansweredReason,
)
from warden.repositories import ChatRepository
from warden.services.errors import InvalidInput, NotFound

MAX_MESSAGE_LENGTH = 2000
MAX_FEEDBACK_COMMENT_LENGTH = 1000
MAX_PAGE_SIZE = 100
TITLE_LENGTH = 80


@dataclass(frozen=True)
class Page[T: (ChatThreadModel, ChatMessageModel)]:
    items: Sequence[T]
    # Pass as `before` to get the next page. None on the last page.
    next: uuid.UUID | None


def thread_title(message: str) -> str:
    """A thread's title from its first message: the first line, cut at a
    word boundary if it's long."""
    line = message.strip().splitlines()[0].strip()
    if len(line) <= TITLE_LENGTH:
        return line
    cut = line[: TITLE_LENGTH - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


class ChatService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.chat = ChatRepository(db)

    # Threads

    async def create_thread(self, user: CurrentUser) -> ChatThreadModel:
        thread = ChatThreadModel(employee_id=user.employee_id)
        self.db.add(thread)
        await self.db.commit()
        return thread

    async def get_thread(self, user: CurrentUser, thread_id: uuid.UUID) -> ChatThreadModel:
        thread = await self.chat.get_thread(thread_id, user.employee_id)
        if thread is None:
            raise NotFound("Thread not found.")
        return thread

    async def list_threads(
        self, user: CurrentUser, before: uuid.UUID | None = None, limit: int = 20
    ) -> Page[ChatThreadModel]:
        """The user's threads, newest first."""
        limit = _clamp(limit)
        rows = await self.chat.list_threads(user.employee_id, before, limit + 1)
        return _page(rows, limit)

    # Messages

    async def list_messages(
        self,
        user: CurrentUser,
        thread_id: uuid.UUID,
        before: uuid.UUID | None = None,
        limit: int = 50,
    ) -> Page[ChatMessageModel]:
        """A thread's messages, newest first."""
        thread = await self.get_thread(user, thread_id)
        limit = _clamp(limit)
        rows = await self.chat.list_messages(thread.id, before, limit + 1)
        return _page(rows, limit)

    async def add_user_message(self, thread: ChatThreadModel, content: str) -> ChatMessageModel:
        """Saved before the agent runs, so the question is kept even if the
        answer fails. The first message also names the thread."""
        if thread.title is None:
            thread.title = thread_title(content)
        return await self._add(thread, MessageRole.USER, content)

    async def add_assistant_message(
        self,
        thread: ChatThreadModel,
        content: str,
        citations: list[dict[str, Any]],
        pending_action_id: uuid.UUID | None = None,
    ) -> ChatMessageModel:
        """`pending_action_id` is the write the reply proposed, if any, so the
        confirmation card comes back when the thread is reopened."""
        return await self._add(thread, MessageRole.ASSISTANT, content, citations, pending_action_id)

    async def _add(
        self,
        thread: ChatThreadModel,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        pending_action_id: uuid.UUID | None = None,
    ) -> ChatMessageModel:
        message = ChatMessageModel(
            thread_id=thread.id,
            role=role,
            content=content,
            citations=citations or [],
            pending_action_id=pending_action_id,
        )
        self.db.add(message)
        await self.db.commit()
        return message

    # Feedback

    async def feedback_for(
        self, user: CurrentUser, messages: Sequence[ChatMessageModel]
    ) -> dict[uuid.UUID, int]:
        """The user's ratings of these messages, by message id."""
        rows = await self.chat.feedback_for(user.employee_id, [m.id for m in messages])
        return {message_id: row.rating for message_id, row in rows.items()}

    async def rate_message(
        self, user: CurrentUser, message_id: uuid.UUID, rating: int, comment: str | None
    ) -> ChatFeedbackModel:
        """Thumbs up (1) or down (-1) on one of the assistant's replies in the
        user's own threads. Rating again replaces the earlier rating."""
        await self._reply_to_rate(user, message_id)
        feedback = await self.chat.upsert_feedback(
            user.employee_id, message_id, rating, comment or None
        )
        await self.db.commit()
        return feedback

    async def clear_rating(self, user: CurrentUser, message_id: uuid.UUID) -> None:
        """Remove the user's rating of a reply. Fine if there was none."""
        await self._reply_to_rate(user, message_id)
        await self.chat.delete_feedback(user.employee_id, message_id)
        await self.db.commit()

    async def _reply_to_rate(self, user: CurrentUser, message_id: uuid.UUID) -> ChatMessageModel:
        message = await self.chat.get_message(message_id, user.employee_id)
        if message is None:
            raise NotFound("Message not found.")
        if message.role != MessageRole.ASSISTANT:
            raise InvalidInput("Only the assistant's replies can be rated.")
        return message

    # Unanswered questions

    async def record_unanswered(
        self, user: CurrentUser, query: str, reason: UnansweredReason
    ) -> None:
        """For HR to review gaps in the policy documents."""
        self.db.add(UnansweredQueryModel(employee_id=user.employee_id, query=query, reason=reason))
        await self.db.commit()


def _clamp(limit: int) -> int:
    return max(1, min(limit, MAX_PAGE_SIZE))


def _page[T: (ChatThreadModel, ChatMessageModel)](rows: Sequence[T], limit: int) -> Page[T]:
    """`rows` was fetched with limit + 1: an extra row means there's more."""
    if len(rows) <= limit:
        return Page(rows, None)
    items = rows[:limit]
    return Page(items, items[-1].id)
