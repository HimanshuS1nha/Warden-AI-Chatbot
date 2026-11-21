import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from warden.models import AccessLevel, DocumentModel, DocumentVersionModel
from warden.models.enums import DocumentVersionStatus


class DocumentVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    status: DocumentVersionStatus
    chunk_count: int
    effective_from: date
    ingested_at: datetime | None
    created_at: datetime


class DocumentResponse(BaseModel):
    id: uuid.UUID
    slug: str
    title: str
    access_level: AccessLevel
    # The version search serves. None until one finishes indexing.
    active_version: int | None
    # Newest first.
    versions: list[DocumentVersionResponse]

    @classmethod
    def build(cls, document: DocumentModel) -> "DocumentResponse":
        versions = sorted(document.versions, key=lambda v: v.version, reverse=True)
        active = next((v.version for v in versions if v.id == document.active_version_id), None)
        return cls(
            id=document.id,
            slug=document.slug,
            title=document.title,
            access_level=document.access_level,
            active_version=active,
            versions=[DocumentVersionResponse.model_validate(v) for v in versions],
        )


class IndexingStartedResponse(BaseModel):
    """Returned with 202: the version is `indexing`. Poll GET
    /api/admin/documents for the outcome."""

    document_id: uuid.UUID
    slug: str
    version: DocumentVersionResponse

    @classmethod
    def build(cls, document: DocumentModel, row: DocumentVersionModel) -> "IndexingStartedResponse":
        return cls(
            document_id=document.id,
            slug=document.slug,
            version=DocumentVersionResponse.model_validate(row),
        )
