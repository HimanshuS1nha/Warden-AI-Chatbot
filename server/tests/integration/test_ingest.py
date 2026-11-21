from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.rag_fakes import FakeEmbeddingClient, WhitespaceTokenCounter, no_sleep
from warden.models import AccessLevel, DocumentModel, DocumentVersionModel, DocumentVersionStatus
from warden.rag.embeddings import Embedder
from warden.rag.index import PolicyIndex, point_id
from warden.rag.ingest import IngestReport, Outcome, ingest, rebuild_index
from warden.rag.store import read_store, store_dir

LEAVE_V1 = """# Leave Policy

## Carry forward

Up to 5 days of earned leave carry forward.
"""

LEAVE_V2 = """# Leave Policy

## Carry forward

Up to 8 days of earned leave carry forward.

## Sick leave

Sick leave can be backdated by up to 7 days.
"""

HANDBOOK = """# Manager Handbook

## Approving leave

Check team cover before approving leave.
"""


class Corpus:
    """A documents folder with a manifest, and a cache folder, in tmp_path."""

    def __init__(self, root: Path, index: PolicyIndex) -> None:
        self.documents = root / "documents"
        self.documents.mkdir()
        self.cache = root / "cache"
        self.index = index
        self.entries: list[dict[str, object]] = []

    def add(
        self,
        slug: str,
        version: int,
        content: str,
        access_level: str = "all",
        suffix: str = ".md",
    ) -> Path:
        file = f"{slug}-v{version}{suffix}"
        (self.documents / file).write_text(content)
        self.entries.append(
            {
                "slug": slug,
                "title": slug.replace("-", " ").title(),
                "access_level": access_level,
                "version": version,
                "effective_date": "2026-01-01",
                "file": file,
            }
        )
        return self.documents / file

    @property
    def manifest(self) -> Path:
        path = self.documents / "manifest.yaml"
        path.write_text(yaml.safe_dump({"documents": self.entries}))
        return path

    async def ingest(
        self, db: AsyncSession, client: FakeEmbeddingClient | None = None, force: bool = False
    ) -> IngestReport:
        # A new Embedder per run, like separate CLI runs sharing one cache.
        embedder = Embedder(client or FakeEmbeddingClient(), self.cache, sleep=no_sleep)
        return await ingest(
            db,
            self.manifest,
            cache_dir=self.cache,
            embedder=embedder,
            counter=WhitespaceTokenCounter(),
            index=self.index,
            force=force,
        )

    async def rebuild(self, db: AsyncSession) -> IngestReport:
        return await rebuild_index(
            db,
            cache_dir=self.cache,
            embedder=Embedder(FakeEmbeddingClient(), self.cache, sleep=no_sleep),
            counter=WhitespaceTokenCounter(),
            index=self.index,
        )

    async def points(self) -> list[dict[str, Any]]:
        """Payloads of every point in the collection, in chunk ID order."""
        records, _ = await self.index.client.scroll(
            self.index.collection, limit=1000, with_payload=True
        )
        assert all(record.id == point_id(record.payload["chunk_id"]) for record in records)  # type: ignore[index]
        return sorted((record.payload or {} for record in records), key=lambda p: p["chunk_id"])


@pytest.fixture
def corpus(tmp_path: Path, policy_index: PolicyIndex) -> Corpus:
    return Corpus(tmp_path, policy_index)


async def _versions(db: AsyncSession, slug: str) -> dict[int, DocumentVersionModel]:
    rows = await db.scalars(
        select(DocumentVersionModel)
        .join(DocumentModel, DocumentVersionModel.document_id == DocumentModel.id)
        .where(DocumentModel.slug == slug)
    )
    return {row.version: row for row in rows}


async def _document(db: AsyncSession, slug: str) -> DocumentModel:
    document = await db.scalar(select(DocumentModel).where(DocumentModel.slug == slug))
    assert document is not None
    return document


async def test_second_ingest_makes_no_embedding_calls(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 2, LEAVE_V2)
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")

    first = await corpus.ingest(db)
    assert first.count(Outcome.INDEXED) == 2
    assert first.embedding_api_calls > 0

    client = FakeEmbeddingClient()
    second = await corpus.ingest(db, client)

    assert second.count(Outcome.SKIPPED) == 2
    assert second.embedding_api_calls == 0
    assert client.calls == []
    assert second.store_chunks == first.store_chunks == 3
    assert second.index_points == first.index_points == 3


async def test_versions_are_recorded_in_postgres(db: AsyncSession, corpus: Corpus) -> None:
    path = corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)

    document = await _document(db, "leave-policy")
    version = (await _versions(db, "leave-policy"))[2]
    assert document.title == "Leave Policy"
    assert document.access_level is AccessLevel.ALL
    assert document.active_version_id == version.id
    assert version.status is DocumentVersionStatus.READY
    assert version.chunk_count == 2
    assert version.file_path == str(path)
    assert len(version.content_hash) == 64
    assert version.ingested_at is not None


async def test_store_holds_only_the_newest_ready_version(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 1, LEAVE_V1)
    corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)

    chunks, vectors, model_id = read_store(store_dir(corpus.cache))
    assert {c.version for c in chunks} == {2}
    assert [c.section for c in chunks] == [
        "Leave Policy > Carry forward",
        "Leave Policy > Sick leave",
    ]
    assert "8 days" in chunks[0].text
    assert len(vectors) == len(chunks)
    assert model_id == FakeEmbeddingClient().model_id

    versions = await _versions(db, "leave-policy")
    document = await _document(db, "leave-policy")
    assert document.active_version_id == versions[2].id
    # The old version stays ready in Postgres, just not active or searchable.
    assert versions[1].status is DocumentVersionStatus.READY
    assert [(p["chunk_id"], p["is_active"]) for p in await corpus.points()] == [
        ("leave-policy:2:0", True),
        ("leave-policy:2:1", True),
    ]


async def test_new_version_replaces_the_old_one_in_qdrant(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 1, LEAVE_V1)
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")
    await corpus.ingest(db)
    assert {p["chunk_id"] for p in await corpus.points()} == {
        "leave-policy:1:0",
        "manager-handbook:1:0",
    }

    corpus.add("leave-policy", 2, LEAVE_V2)
    report = await corpus.ingest(db)

    points = await corpus.points()
    assert [p["chunk_id"] for p in points] == [
        "leave-policy:2:0",
        "leave-policy:2:1",
        "manager-handbook:1:0",
    ]
    assert "8 days" in points[0]["text"]
    assert report.index_points == 3


async def test_reindexed_version_with_fewer_chunks_drops_the_extra_points(
    db: AsyncSession, corpus: Corpus
) -> None:
    leave = corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)

    leave.write_text(LEAVE_V1)
    await corpus.ingest(db)

    assert [p["chunk_id"] for p in await corpus.points()] == ["leave-policy:2:0"]


async def test_points_carry_the_chunk_metadata(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")
    await corpus.ingest(db)

    assert await corpus.points() == [
        {
            "chunk_id": "manager-handbook:1:0",
            "doc_slug": "manager-handbook",
            "title": "Manager Handbook",
            "version": 1,
            "access_level": "manager",
            "chunk_index": 0,
            "section": "Manager Handbook > Approving leave",
            "page": None,
            "text": "Check team cover before approving leave.",
            "token_count": 6,
            "is_active": True,
        }
    ]


async def test_changed_file_is_reindexed_and_unchanged_ones_skipped(
    db: AsyncSession, corpus: Corpus
) -> None:
    leave = corpus.add("leave-policy", 2, LEAVE_V2)
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")
    await corpus.ingest(db)
    old_hash = (await _versions(db, "leave-policy"))[2].content_hash

    leave.write_text(LEAVE_V2.replace("7 days", "10 days"))
    client = FakeEmbeddingClient()
    report = await corpus.ingest(db, client)

    assert report.outcomes == {
        "leave-policy v2": Outcome.INDEXED,
        "manager-handbook v1": Outcome.SKIPPED,
    }
    # Only the edited chunk is new; the unchanged one comes from the cache.
    assert [texts for texts, _ in client.calls] == [
        ["Leave Policy\nLeave Policy > Sick leave\n\nSick leave can be backdated by up to 10 days."]
    ]
    assert (await _versions(db, "leave-policy"))[2].content_hash != old_hash


async def test_force_reparses_but_embeddings_still_come_from_cache(
    db: AsyncSession, corpus: Corpus
) -> None:
    corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)

    report = await corpus.ingest(db, force=True)

    assert report.outcomes == {"leave-policy v2": Outcome.INDEXED}
    assert report.embedding_api_calls == 0


async def test_failed_version_is_marked_and_the_rest_carry_on(
    db: AsyncSession, corpus: Corpus
) -> None:
    corpus.add("leave-policy", 1, LEAVE_V1)
    corpus.add("leave-policy", 2, "not a document", suffix=".txt")
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")

    report = await corpus.ingest(db)

    assert report.outcomes == {
        "leave-policy v1": Outcome.INDEXED,
        "leave-policy v2": Outcome.FAILED,
        "manager-handbook v1": Outcome.INDEXED,
    }
    versions = await _versions(db, "leave-policy")
    assert versions[2].status is DocumentVersionStatus.FAILED
    # The failed v2 doesn't replace v1.
    assert (await _document(db, "leave-policy")).active_version_id == versions[1].id
    chunks, _, _ = read_store(store_dir(corpus.cache))
    assert {(c.doc_slug, c.version) for c in chunks} == {
        ("leave-policy", 1),
        ("manager-handbook", 1),
    }
    assert {(p["doc_slug"], p["version"]) for p in await corpus.points()} == {
        ("leave-policy", 1),
        ("manager-handbook", 1),
    }


async def test_failed_version_is_retried_on_the_next_run(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 1, "")  # no text: fails
    first = await corpus.ingest(db)
    assert first.outcomes == {"leave-policy v1": Outcome.FAILED}
    assert first.index_points == 0

    (corpus.documents / "leave-policy-v1.md").write_text(LEAVE_V1)
    second = await corpus.ingest(db)
    assert second.outcomes == {"leave-policy v1": Outcome.INDEXED}


async def test_access_level_change_applies_without_reparsing(
    db: AsyncSession, corpus: Corpus
) -> None:
    corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)

    corpus.entries[0]["access_level"] = "hr"
    report = await corpus.ingest(db)

    assert report.outcomes == {"leave-policy v2": Outcome.SKIPPED}
    assert report.embedding_api_calls == 0
    assert (await _document(db, "leave-policy")).access_level is AccessLevel.HR
    chunks, _, _ = read_store(store_dir(corpus.cache))
    assert {c.access_level for c in chunks} == {AccessLevel.HR}
    assert {p["access_level"] for p in await corpus.points()} == {"hr"}


async def test_deleted_chunk_cache_is_rebuilt(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)
    (corpus.cache / "chunks" / "leave-policy" / "v2.json").unlink()

    report = await corpus.ingest(db)

    assert report.outcomes == {"leave-policy v2": Outcome.INDEXED}
    assert report.store_chunks == 2


async def test_rebuild_reindexes_every_active_version(db: AsyncSession, corpus: Corpus) -> None:
    corpus.add("leave-policy", 1, LEAVE_V1)
    corpus.add("leave-policy", 2, LEAVE_V2)
    corpus.add("manager-handbook", 1, HANDBOOK, access_level="manager")
    await corpus.ingest(db)
    before = await corpus.points()
    await corpus.index.recreate_collection()  # the collection is lost

    report = await corpus.rebuild(db)

    assert report.outcomes == {
        "leave-policy v2": Outcome.INDEXED,
        "manager-handbook v1": Outcome.INDEXED,
    }
    assert report.index_points == 3
    assert await corpus.points() == before


async def test_rebuild_reparses_a_version_missing_from_the_cache(
    db: AsyncSession, corpus: Corpus
) -> None:
    corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)
    (corpus.cache / "chunks" / "leave-policy" / "v2.json").unlink()

    report = await corpus.rebuild(db)

    assert report.outcomes == {"leave-policy v2": Outcome.INDEXED}
    assert report.index_points == 2


async def test_rebuild_skips_a_version_whose_file_changed(db: AsyncSession, corpus: Corpus) -> None:
    leave = corpus.add("leave-policy", 2, LEAVE_V2)
    await corpus.ingest(db)
    (corpus.cache / "chunks" / "leave-policy" / "v2.json").unlink()
    leave.write_text(LEAVE_V1)

    report = await corpus.rebuild(db)

    # Postgres records the old file's hash: indexing the new text under it
    # would serve content nobody ingested.
    assert report.outcomes == {"leave-policy v2": Outcome.FAILED}
    assert report.index_points == 0
