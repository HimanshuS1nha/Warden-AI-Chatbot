"""Index the policy documents listed in the manifest.

    uv run python -m warden.rag.ingest             # index new and changed versions
    uv run python -m warden.rag.ingest --force     # re-parse every version
    uv run python -m warden.rag.ingest --rebuild   # recreate the Qdrant collection

For each version in `documents/manifest.yaml`: hash the file and skip it if
Postgres already has it `ready` with the same hash and its chunk file is in
the cache. Otherwise parse, clean, chunk and embed it, save its chunks, and
mark it `ready`. A version that fails is marked `failed` and the run carries
on with the rest.

Then each document switches to its highest ready version: that version's
chunks are upserted to Qdrant, Postgres marks it active, and the document's
other points (older versions, chunks the version no longer has) are marked
inactive and deleted. Upserts overwrite points with the same ID, so a run
with nothing new leaves the collection as it was.

Last, the chunk store is rewritten with every active version's chunks and
vectors, for `InMemoryRetriever`. Vectors come from the embeddings cache, so
a run with nothing new makes no embedding API calls.

`--rebuild` ignores the manifest: it drops the collection and indexes every
active version recorded in Postgres, including uploaded ones. Use it after
changing the embedding model, or if the collection is lost.

Never run at app startup: it calls a paid API and can take minutes.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from itertools import groupby
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.config import Settings, get_settings
from warden.core.database import SessionLocal, engine
from warden.core.logging import configure_logging
from warden.models import AccessLevel, DocumentModel, DocumentVersionModel, DocumentVersionStatus
from warden.rag.chunking import Chunk, DocumentInfo, HFTokenCounter, TokenCounter, chunk_blocks
from warden.rag.cleaning import clean_blocks
from warden.rag.embeddings import Embedder, EmbeddingError, EmbedTask, build_embedding_client
from warden.rag.index import PolicyIndex, PolicyIndexError, build_index
from warden.rag.parsing import parse_document
from warden.rag.store import (
    read_version_chunks,
    store_dir,
    version_chunks_path,
    write_store,
    write_version_chunks,
)

# Named explicitly: under `python -m` this module is "__main__", outside the
# "warden" logger that configure_logging sets to INFO.
logger = logging.getLogger("warden.rag.ingest")


class ManifestError(Exception):
    pass


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=100)
    title: str = Field(min_length=1, max_length=255)
    access_level: AccessLevel
    version: int = Field(gt=0)
    effective_date: date
    # Relative to the manifest's folder.
    file: str


class _Manifest(BaseModel):
    documents: list[ManifestEntry]


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Read and check the manifest: valid fields, no duplicate versions, the
    same title and access level for every version of a document, and every
    file present inside the manifest's folder."""
    try:
        manifest = _Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ManifestError(f"Can't read {path}: {exc}") from exc

    folder = path.parent.resolve()
    first_by_slug: dict[str, ManifestEntry] = {}
    seen: set[tuple[str, int]] = set()
    for entry in manifest.documents:
        if (entry.slug, entry.version) in seen:
            raise ManifestError(f"{entry.slug} version {entry.version} is listed twice")
        seen.add((entry.slug, entry.version))

        first = first_by_slug.setdefault(entry.slug, entry)
        if (first.title, first.access_level) != (entry.title, entry.access_level):
            raise ManifestError(
                f"Every version of {entry.slug} needs the same title and access level"
            )

        file = (folder / entry.file).resolve()
        if not file.is_relative_to(folder):
            raise ManifestError(f"{entry.file} is outside {folder}")
        if not file.is_file():
            raise ManifestError(f"{entry.file} doesn't exist")
    return manifest.documents


class Outcome(StrEnum):
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class IngestReport:
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    store_chunks: int = 0
    index_points: int = 0
    embedding_api_calls: int = 0
    embedding_cache_hits: int = 0
    seconds: float = 0.0

    def count(self, outcome: Outcome) -> int:
        return sum(1 for value in self.outcomes.values() if value is outcome)


async def ingest(
    session: AsyncSession,
    manifest_path: Path,
    *,
    cache_dir: Path,
    embedder: Embedder,
    counter: TokenCounter,
    index: PolicyIndex,
    force: bool = False,
) -> IngestReport:
    started = time.perf_counter()
    calls_before, hits_before = embedder.api_calls, embedder.cache_hits
    entries = load_manifest(manifest_path)
    await index.ensure_collection()
    report = IngestReport()

    ordered = sorted(entries, key=lambda entry: (entry.slug, entry.version))
    for _, group in groupby(ordered, key=lambda entry: entry.slug):
        versions = list(group)
        document = await _upsert_document(session, versions[-1])
        for entry in versions:
            report.outcomes[f"{entry.slug} v{entry.version}"] = await _ingest_version(
                session,
                document,
                entry,
                manifest_path.parent,
                cache_dir=cache_dir,
                embedder=embedder,
                counter=counter,
                force=force,
            )
        await switch_active_version(
            session, document, cache_dir=cache_dir, embedder=embedder, counter=counter, index=index
        )

    return await _finish(
        report, session, cache_dir, embedder, index, started, calls_before, hits_before
    )


async def rebuild_index(
    session: AsyncSession,
    *,
    cache_dir: Path,
    embedder: Embedder,
    counter: TokenCounter,
    index: PolicyIndex,
) -> IngestReport:
    """Drop the collection and index every active version in Postgres.
    Doesn't read the manifest or change which versions are active."""
    started = time.perf_counter()
    calls_before, hits_before = embedder.api_calls, embedder.cache_hits
    await index.recreate_collection()
    report = IngestReport()

    for document, version in await _active_versions(session):
        label = f"{document.slug} v{version.version}"
        chunks = await _load_chunks(document, version, cache_dir, counter)
        if chunks is None:
            report.outcomes[label] = Outcome.FAILED
            continue
        vectors = await embedder.embed([c.embedding_text for c in chunks], EmbedTask.DOCUMENT)
        await index.upsert(chunks, vectors)
        report.outcomes[label] = Outcome.INDEXED

    return await _finish(
        report, session, cache_dir, embedder, index, started, calls_before, hits_before
    )


async def _finish(
    report: IngestReport,
    session: AsyncSession,
    cache_dir: Path,
    embedder: Embedder,
    index: PolicyIndex,
    started: float,
    calls_before: int,
    hits_before: int,
) -> IngestReport:
    report.store_chunks = await rebuild_store(session, cache_dir, embedder)
    report.index_points = await index.count()
    report.embedding_api_calls = embedder.api_calls - calls_before
    report.embedding_cache_hits = embedder.cache_hits - hits_before
    report.seconds = time.perf_counter() - started
    logger.info(
        "Ingest finished in %.1fs: %d indexed, %d unchanged, %d failed; "
        "%d points in Qdrant, %d chunks in store; %d embedding API calls, %d cache hits",
        report.seconds,
        report.count(Outcome.INDEXED),
        report.count(Outcome.SKIPPED),
        report.count(Outcome.FAILED),
        report.index_points,
        report.store_chunks,
        report.embedding_api_calls,
        report.embedding_cache_hits,
    )
    return report


async def _upsert_document(session: AsyncSession, entry: ManifestEntry) -> DocumentModel:
    document = await session.scalar(select(DocumentModel).where(DocumentModel.slug == entry.slug))
    if document is None:
        document = DocumentModel(
            slug=entry.slug, title=entry.title, access_level=entry.access_level
        )
        session.add(document)
    else:
        document.title = entry.title
        document.access_level = entry.access_level
    await session.flush()
    return document


async def _ingest_version(
    session: AsyncSession,
    document: DocumentModel,
    entry: ManifestEntry,
    documents_dir: Path,
    *,
    cache_dir: Path,
    embedder: Embedder,
    counter: TokenCounter,
    force: bool,
) -> Outcome:
    label = f"{entry.slug} v{entry.version}"
    path = documents_dir / entry.file
    content_hash = hashlib.sha256(await asyncio.to_thread(path.read_bytes)).hexdigest()
    chunks_path = version_chunks_path(cache_dir, entry.slug, entry.version)

    row = await session.scalar(
        select(DocumentVersionModel).where(
            DocumentVersionModel.document_id == document.id,
            DocumentVersionModel.version == entry.version,
        )
    )
    if (
        not force
        and row is not None
        and row.status is DocumentVersionStatus.READY
        and row.content_hash == content_hash
    ):
        saved = read_version_chunks(chunks_path)
        if saved is not None and saved[0] == content_hash:
            logger.info("%s unchanged, skipped", label)
            return Outcome.SKIPPED

    if row is None:
        row = DocumentVersionModel(document_id=document.id, version=entry.version)
        session.add(row)
    row.file_path = str(path)
    row.content_hash = content_hash
    row.effective_from = entry.effective_date
    row.status = DocumentVersionStatus.INDEXING
    await session.commit()
    return await index_version(
        session, document, row, cache_dir=cache_dir, embedder=embedder, counter=counter
    )


async def index_version(
    session: AsyncSession,
    document: DocumentModel,
    row: DocumentVersionModel,
    *,
    cache_dir: Path,
    embedder: Embedder,
    counter: TokenCounter,
) -> Outcome:
    """Parse, chunk and embed the version's file, save its chunks to the
    cache and mark it `ready`, or `failed` if any of that fails. Doesn't
    touch Qdrant: `switch_active_version` does that. Raises EmbeddingError,
    which the next version would hit too."""
    label = f"{document.slug} v{row.version}"
    path = Path(row.file_path)
    chunks_path = version_chunks_path(cache_dir, document.slug, row.version)
    started = time.perf_counter()
    info = DocumentInfo(document.slug, document.title, row.version, document.access_level)
    try:
        chunks = await asyncio.to_thread(_parse_and_chunk, path, info, counter)
        # Embedding here, not only when the store is built, means `ready`
        # always has its vectors in the cache.
        await embedder.embed([chunk.embedding_text for chunk in chunks], EmbedTask.DOCUMENT)
        write_version_chunks(chunks_path, row.content_hash, chunks)
    except EmbeddingError:
        # Not this document's fault, and the next one would fail the same way.
        row.status = DocumentVersionStatus.FAILED
        await session.commit()
        raise
    except Exception:
        logger.exception("%s failed", label)
        row.status = DocumentVersionStatus.FAILED
        await session.commit()
        return Outcome.FAILED

    row.status = DocumentVersionStatus.READY
    row.chunk_count = len(chunks)
    row.ingested_at = datetime.now(UTC)
    await session.commit()
    logger.info("%s indexed: %d chunks in %.1fs", label, len(chunks), time.perf_counter() - started)
    return Outcome.INDEXED


def _parse_and_chunk(path: Path, info: DocumentInfo, counter: TokenCounter) -> list[Chunk]:
    chunks = chunk_blocks(clean_blocks(parse_document(path)), info, counter)
    if not chunks:
        raise ValueError(f"No text found in {path.name}")
    return chunks


async def switch_active_version(
    session: AsyncSession,
    document: DocumentModel,
    *,
    cache_dir: Path,
    embedder: Embedder,
    counter: TokenCounter,
    index: PolicyIndex,
) -> None:
    """Make the document's newest ready version the one search serves:
    index it, mark it active in Postgres, then remove every other point of
    the document from Qdrant. With no ready version, the document's points
    are all removed.

    The document row stays locked from start to finish. Two switches for one
    document (an upload and a reindex, or two app workers) would otherwise
    interleave, and one could delete the points the other just wrote."""
    await session.execute(
        select(DocumentModel.id).where(DocumentModel.id == document.id).with_for_update()
    )
    latest = await session.scalar(
        select(DocumentVersionModel)
        .where(
            DocumentVersionModel.document_id == document.id,
            DocumentVersionModel.status == DocumentVersionStatus.READY,
        )
        .order_by(DocumentVersionModel.version.desc())
        .limit(1)
    )
    keep: list[str] = []
    if latest is not None:
        chunks = await _load_chunks(document, latest, cache_dir, counter)
        if chunks is None:
            # Leave Postgres and Qdrant as they are rather than serve nothing.
            await session.commit()
            return
        vectors = await embedder.embed([c.embedding_text for c in chunks], EmbedTask.DOCUMENT)
        keep = await index.upsert(chunks, vectors)

    document.active_version_id = latest.id if latest is not None else None
    await session.flush()
    try:
        removed = await index.retire_stale(document.slug, keep)
    finally:
        # Commits the new active version and releases the lock, even if
        # Qdrant failed: the new points are in, and the next run retries.
        await session.commit()
    if removed:
        logger.info("%s: removed %d points of other versions", document.slug, removed)


async def _active_versions(
    session: AsyncSession,
) -> list[tuple[DocumentModel, DocumentVersionModel]]:
    rows = await session.execute(
        select(DocumentModel, DocumentVersionModel)
        .join(DocumentVersionModel, DocumentModel.active_version_id == DocumentVersionModel.id)
        .order_by(DocumentModel.slug)
    )
    return [(document, version) for document, version in rows]


async def _load_chunks(
    document: DocumentModel,
    version: DocumentVersionModel,
    cache_dir: Path,
    counter: TokenCounter | None = None,
) -> list[Chunk] | None:
    """A ready version's chunks, from the cache. With a `counter`, a version
    missing from the cache is re-parsed from its file, as long as the file
    still has the hash Postgres recorded. None if neither works.

    Title and access level come from the document row, so changing them
    takes effect without re-parsing the file."""
    label = f"{document.slug} v{version.version}"
    path = version_chunks_path(cache_dir, document.slug, version.version)
    saved = read_version_chunks(path)
    if saved is not None and saved[0] == version.content_hash:
        chunks = saved[1]
    elif counter is None:
        logger.warning("%s is active but its chunks aren't in the cache", label)
        return None
    else:
        file = Path(version.file_path)
        try:
            content_hash = hashlib.sha256(await asyncio.to_thread(file.read_bytes)).hexdigest()
        except OSError:
            content_hash = None
        if content_hash != version.content_hash:
            logger.warning(
                "%s isn't in the cache and %s is missing or changed. Run ingest without "
                "--rebuild to index the file as it is now.",
                label,
                file,
            )
            return None
        info = DocumentInfo(document.slug, document.title, version.version, document.access_level)
        try:
            chunks = await asyncio.to_thread(_parse_and_chunk, file, info, counter)
        except Exception:
            logger.exception("%s failed", label)
            return None
        write_version_chunks(path, content_hash, chunks)

    return [
        dataclasses.replace(chunk, title=document.title, access_level=document.access_level)
        for chunk in chunks
    ]


async def rebuild_store(session: AsyncSession, cache_dir: Path, embedder: Embedder) -> int:
    chunks: list[Chunk] = []
    for document, version in await _active_versions(session):
        loaded = await _load_chunks(document, version, cache_dir)
        if loaded is None:
            continue
        chunks.extend(loaded)

    vectors = await embedder.embed([chunk.embedding_text for chunk in chunks], EmbedTask.DOCUMENT)
    write_store(store_dir(cache_dir), chunks, vectors, embedder.model_id)
    return len(chunks)


async def _run(
    settings: Settings, manifest_path: Path, *, force: bool, rebuild: bool
) -> IngestReport:
    client = build_embedding_client(settings)
    embedder = Embedder(client, settings.cache_dir, batch_size=settings.embedding_batch_size)
    counter = HFTokenCounter()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    index = build_index(settings, qdrant)
    try:
        async with SessionLocal() as session:
            if rebuild:
                return await rebuild_index(
                    session,
                    cache_dir=settings.cache_dir,
                    embedder=embedder,
                    counter=counter,
                    index=index,
                )
            return await ingest(
                session,
                manifest_path,
                cache_dir=settings.cache_dir,
                embedder=embedder,
                counter=counter,
                index=index,
                force=force,
            )
    finally:
        await client.aclose()
        await qdrant.close()
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index the policy documents for retrieval.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force",
        action="store_true",
        help="re-parse and re-chunk every version, even unchanged ones "
        "(embeddings still come from the cache)",
    )
    mode.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the Qdrant collection and re-index every active version in Postgres "
        "(ignores the manifest)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.documents_dir / "manifest.yaml",
        help="manifest file (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)
    try:
        report = asyncio.run(_run(settings, args.manifest, force=args.force, rebuild=args.rebuild))
    except (ManifestError, EmbeddingError, PolicyIndexError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    if report.count(Outcome.FAILED):
        sys.exit(1)


if __name__ == "__main__":
    main()
