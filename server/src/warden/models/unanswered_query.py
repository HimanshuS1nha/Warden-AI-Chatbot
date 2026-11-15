import uuid

from sqlalchemy import ForeignKey, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import UnansweredReason, pg_enum


class UnansweredQueryModel(UUIDPrimaryKey, Base):
    """Questions policy search couldn't answer. HR reviews these to find gaps
    in the documents."""

    __tablename__ = "unanswered_queries"
    __table_args__ = (Index("ix_unanswered_queries_created_at", "created_at"),)

    employee_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), index=True
    )
    query: Mapped[str] = mapped_column(Text)
    reason: Mapped[UnansweredReason] = mapped_column(pg_enum(UnansweredReason, "unanswered_reason"))
