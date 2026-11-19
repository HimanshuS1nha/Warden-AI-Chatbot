import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import MessageRole, pg_enum


class ChatMessageModel(UUIDPrimaryKey, Base):
    """A message the employee sees in the chat: what they typed and the
    assistant's final reply.

    This is the display history, kept for the employee. The agent's own
    memory (tool calls, tool results) stays in the LangGraph checkpointer.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        # Infinite scroll, newest first:
        # WHERE thread_id = :t AND id < :cursor ORDER BY id DESC LIMIT n
        Index("ix_chat_messages_thread_id_id", "thread_id", "id"),
    )

    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_threads.id", ondelete="CASCADE"))
    role: Mapped[MessageRole] = mapped_column(pg_enum(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    # Citations shown under an assistant reply, so the chips still work when
    # the thread is reopened: [{"title", "section", "page", "text"}, ...]
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    # Set on the reply that proposed an action, so the confirmation card is
    # shown again with the action's current status.
    pending_action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pending_actions.id", ondelete="SET NULL")
    )
