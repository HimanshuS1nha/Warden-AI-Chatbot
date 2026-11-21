"""Indexes one document version inside the app, for the admin endpoints.

The CLI (`warden.rag.ingest`) indexes the manifest. Uploads and reindex
requests come through the API instead: the request records the version and
returns, and `DocumentIndexer.run` does the slow part as a background task,
with the same steps the CLI uses.
"""

import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import AccessLevel, DocumentModel, DocumentVersionModel
from warden.rag.chunking import TokenCounter
from warden.rag.embeddings import Embedder
from warden.rag.index import PolicyIndex
from warden.rag.ingest import Outcome, index_version, rebuild_store, switch_active_version

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        *,
        cache_dir: Path,
        embedder: Embedder,
        counter: TokenCounter,
        index: PolicyIndex,
    ) -> None:
        self._session_factory = session_factory
        self._cache_dir = cache_dir
        self._embedder = embedder
        self._counter = counter
        self.index = index

    async def run(self, version_id: uuid.UUID) -> Outcome:
        """Parse, embed and index the version, then make the document's
        newest ready version active. Never raises: it runs after the
        response is sent, so failures go to the log and the version's
        status."""
        try:
            async with self._session_factory() as session:
                row = await session.get(DocumentVersionModel, version_id)
                document = (
                    await session.get(DocumentModel, row.document_id) if row is not None else None
                )
                if row is None or document is None:
                    logger.error("Document version %s doesn't exist", version_id)
                    return Outcome.FAILED

                outcome = await index_version(
                    session,
                    document,
                    row,
                    cache_dir=self._cache_dir,
                    embedder=self._embedder,
                    counter=self._counter,
                )
                await self.index.ensure_collection()
                await switch_active_version(
                    session,
                    document,
                    cache_dir=self._cache_dir,
                    embedder=self._embedder,
                    counter=self._counter,
                    index=self.index,
                )
                await rebuild_store(session, self._cache_dir, self._embedder)
                return outcome
        except Exception:
            logger.exception("Indexing document version %s failed", version_id)
            return Outcome.FAILED

    async def set_document_fields(self, slug: str, title: str, access_level: AccessLevel) -> None:
        """Change the title and access level on a document's existing points
        right away, instead of when its next version is indexed."""
        await self.index.set_document_fields(slug, title=title, access_level=access_level)
