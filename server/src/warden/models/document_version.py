import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import DocumentVersionStatus, pg_enum

if TYPE_CHECKING:
    from warden.models.document import DocumentModel


class DocumentVersionModel(UUIDPrimaryKey, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version"),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint("chunk_count >= 0", name="chunk_count_non_negative"),
    )

    # No separate index: the unique constraint's index starts with document_id.
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))
    # SHA-256 hex digest of the file. Ingest skips a version whose hash hasn't changed.
    content_hash: Mapped[str] = mapped_column(String(64))
    effective_from: Mapped[date] = mapped_column(Date)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    chunk_count: Mapped[int] = mapped_column(Integer, server_default="0")
    status: Mapped[DocumentVersionStatus] = mapped_column(
        pg_enum(DocumentVersionStatus, "document_version_status"),
        server_default=DocumentVersionStatus.INDEXING.value,
    )

    document: Mapped["DocumentModel"] = relationship(
        back_populates="versions", foreign_keys=[document_id], lazy="raise"
    )
