"""Policy documents for HR: upload a new version, reindex, list.

Indexing is slow (parsing, embedding API calls), so these methods only
record what to index and return. The router hands the version to
`DocumentIndexer.run` as a background task. A version is `indexing` until
that finishes, then `ready` or `failed`.

Uploaded files are saved under `upload_dir/<slug>/v<version>.<ext>`. The
client's file name is never used in a path.
"""

import asyncio
import hashlib
import os
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Any, NoReturn, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from warden.core.dependencies import CurrentUser
from warden.models import (
    AccessLevel,
    AuditAction,
    AuditOutcome,
    AuditTarget,
    DocumentModel,
    DocumentVersionModel,
    DocumentVersionStatus,
    Role,
)
from warden.services import audit
from warden.services.access import ensure_role
from warden.services.errors import (
    NotFound,
    RuleViolation,
    Unavailable,
    VersionConflict,
)
from warden.services.uploads import check_upload


class DocumentFieldsUpdater(Protocol):
    async def set_document_fields(
        self, slug: str, title: str, access_level: AccessLevel
    ) -> None: ...


class DocumentService:
    def __init__(self, db: AsyncSession, *, upload_dir: Path) -> None:
        self.db = db
        self.upload_dir = upload_dir

    async def list_documents(self, user: CurrentUser) -> list[DocumentModel]:
        await ensure_role(self.db, user, Role.HR, operation="list_documents")
        rows = await self.db.scalars(
            select(DocumentModel)
            .options(selectinload(DocumentModel.versions))
            .order_by(DocumentModel.slug)
        )
        return list(rows)

    async def add_version(
        self,
        user: CurrentUser,
        *,
        slug: str,
        title: str,
        access_level: AccessLevel,
        version: int,
        effective_from: date,
        filename: str | None,
        data: bytes,
        index: DocumentFieldsUpdater,
    ) -> tuple[DocumentModel, DocumentVersionModel]:
        """Save the file and record the version as `indexing`.

        A new slug creates the document. For an existing one, the version
        must be newer than every version it has, and a changed title or
        access level applies to the whole document at once, including the
        points already in Qdrant, so a document made stricter is never
        searchable at its old level while the new version indexes.
        """
        await ensure_role(self.db, user, Role.HR, operation="upload_document")
        facts: dict[str, Any] = {
            "slug": slug,
            "version": version,
            "access_level": access_level.value,
            "size_bytes": len(data),
        }
        try:
            kind = check_upload(filename, data)
        except RuleViolation as exc:
            await self._deny(user, AuditAction.DOCUMENT_UPLOAD, exc, facts)

        document = await self.db.scalar(
            select(DocumentModel)
            .where(DocumentModel.slug == slug)
            .options(selectinload(DocumentModel.versions))
        )
        newest = max((v.version for v in document.versions), default=0) if document else 0
        if version <= newest:
            await self._deny(
                user,
                AuditAction.DOCUMENT_UPLOAD,
                VersionConflict(
                    f"{slug} already has version {newest}. Upload version {newest + 1} or later.",
                    latest_version=newest,
                ),
                facts,
            )

        fields_changed = document is not None and (
            document.title != title or document.access_level is not access_level
        )
        if document is None:
            document = DocumentModel(slug=slug, title=title, access_level=access_level)
            self.db.add(document)
            await self.db.flush()
        else:
            facts["previous_access_level"] = document.access_level.value
            document.title = title
            document.access_level = access_level

        path = self.upload_dir / slug / f"v{version}{kind.suffix}"
        await asyncio.to_thread(_atomic_write, path, data)
        row = DocumentVersionModel(
            document_id=document.id,
            version=version,
            file_path=str(path),
            content_hash=hashlib.sha256(data).hexdigest(),
            effective_from=effective_from,
            status=DocumentVersionStatus.INDEXING,
        )
        self.db.add(row)
        await self.db.flush()

        if fields_changed:
            try:
                await index.set_document_fields(slug, title, access_level)
            except Exception as exc:
                # Nothing is committed, so the document keeps its old fields
                # everywhere.
                raise Unavailable("Couldn't update the search index. Try again later.") from exc

        audit.record(
            self.db,
            action=AuditAction.DOCUMENT_UPLOAD,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.DOCUMENT_VERSION,
            target_id=row.id,
            details=facts | {"title": title, "file_type": kind.value},
        )
        await self.db.commit()
        return document, row

    async def start_reindex(
        self, user: CurrentUser, document_id: uuid.UUID
    ) -> tuple[DocumentModel, DocumentVersionModel]:
        """Mark the document's newest version for indexing again: a failed
        upload gets another try, a ready one is re-parsed and re-embedded."""
        await ensure_role(self.db, user, Role.HR, operation="reindex_document")
        document = await self.db.scalar(
            select(DocumentModel)
            .where(DocumentModel.id == document_id)
            .options(selectinload(DocumentModel.versions))
        )
        if document is None or not document.versions:
            raise NotFound("Document not found")

        row = max(document.versions, key=lambda v: v.version)
        previous = row.status
        row.status = DocumentVersionStatus.INDEXING
        audit.record(
            self.db,
            action=AuditAction.DOCUMENT_REINDEX,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.DOCUMENT_VERSION,
            target_id=row.id,
            details={
                "slug": document.slug,
                "version": row.version,
                "previous_status": previous.value,
            },
        )
        await self.db.commit()
        return document, row

    async def _deny(
        self,
        user: CurrentUser,
        action: AuditAction,
        error: RuleViolation,
        facts: dict[str, Any],
    ) -> NoReturn:
        audit.record(
            self.db,
            action=action,
            outcome=AuditOutcome.DENIED,
            actor_id=user.employee_id,
            target_type=AuditTarget.DOCUMENT_VERSION,
            details=facts | {"reason": error.code},
        )
        await self.db.commit()
        raise error


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
