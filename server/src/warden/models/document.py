import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import AccessLevel, pg_enum

if TYPE_CHECKING:
    from warden.models.document_version import DocumentVersionModel


class DocumentModel(UUIDPrimaryKey, Base):
    __tablename__ = "documents"

    slug: Mapped[str] = mapped_column(String(100), unique=True)  # "leave-policy"
    title: Mapped[str] = mapped_column(String(255))
    access_level: Mapped[AccessLevel] = mapped_column(pg_enum(AccessLevel, "access_level"))
    # The version retrieval serves. Null until the first version finishes indexing.
    # documents and document_versions reference each other, so this foreign
    # key is added after both tables exist (use_alter).
    active_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL", use_alter=True)
    )

    versions: Mapped[list["DocumentVersionModel"]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersionModel.document_id",
        lazy="raise",
    )
    active_version: Mapped["DocumentVersionModel | None"] = relationship(
        foreign_keys=[active_version_id],
        # Breaks the insert cycle: insert both rows, then UPDATE this column.
        post_update=True,
        lazy="raise",
    )
