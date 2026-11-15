import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    SmallInteger,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey


class ChatFeedbackModel(UUIDPrimaryKey, Base):
    __tablename__ = "chat_feedback"
    __table_args__ = (
        # One rating per message. Changing it updates the row.
        UniqueConstraint("employee_id", "message_id"),
        CheckConstraint("rating IN (-1, 1)", name="thumbs_up_or_down"),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE")
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    rating: Mapped[int] = mapped_column(SmallInteger)  # 1 = thumbs up, -1 = thumbs down
    comment: Mapped[str | None] = mapped_column(Text)
