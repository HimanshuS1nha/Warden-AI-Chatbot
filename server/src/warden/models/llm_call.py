import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey


class LLMCallModel(UUIDPrimaryKey, Base):
    """One row per chat model call, for latency and cost stats."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_created_at", "created_at"),)

    # Null for calls outside a chat thread (evaluation runs), and after the
    # thread is deleted: the stats outlive the conversation.
    thread_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chat_threads.id", ondelete="SET NULL"), index=True
    )
    model: Mapped[str] = mapped_column(String(100))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    # Names of the tools the model asked for, e.g. ["get_my_leave_balance"].
    tool_calls: Mapped[list[Any]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
