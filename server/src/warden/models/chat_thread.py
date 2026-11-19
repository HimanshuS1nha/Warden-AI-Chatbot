import uuid

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey


class ChatThreadModel(UUIDPrimaryKey, Base):
    """A conversation. Its id is also the LangGraph thread_id, so the agent's
    checkpoint and the visible history share one key.

    Every read goes through `WHERE employee_id = :me`, so an employee can only
    open their own threads.
    """

    __tablename__ = "chat_threads"
    __table_args__ = (
        # Thread list, newest first, paginated by id:
        # WHERE employee_id = :me AND id < :cursor ORDER BY id DESC
        Index("ix_chat_threads_employee_id_id", "employee_id", "id"),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    # Set from the first message. Null until then.
    title: Mapped[str | None] = mapped_column(String(200))
