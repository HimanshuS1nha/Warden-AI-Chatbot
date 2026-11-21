"""Access and versioning checks over the real policy documents (8.12, 8.13).

The PDFs in documents/ are parsed and chunked for real; only the embedding
model, BM25 and the cross-encoder are fakes, so this runs offline. Searches
go through the agent's search_policies tool, whose filter comes from the
signed-in user's role, never from the query.

Parsing takes about a second per PDF, so each file is parsed once per run
and the chunks reused (`parse_once`).
"""

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.messages import ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from tests.agent_fakes import call
from tests.factories import user_with_role
from tests.rag_fakes import FakeEmbeddingClient, FakeReranker, WhitespaceTokenCounter, no_sleep
from warden.core.config import get_settings
from warden.graph.tools.policy import MAX_QUERY_LENGTH, build_policy_tools
from warden.models import AccessLevel, Role
from warden.rag import ingest as ingest_module
from warden.rag.chunking import Chunk, DocumentInfo
from warden.rag.embeddings import Embedder, EmbedTask
from warden.rag.index import PolicyIndex
from warden.rag.ingest import ingest, load_manifest
from warden.rag.retrievers import (
    InMemoryRetriever,
    QdrantDenseRetriever,
    QdrantHybridRetriever,
    RerankingRetriever,
    Retriever,
)

pytestmark = [pytest.mark.security, pytest.mark.slow]

DOCUMENTS = Path(__file__).resolve().parents[2] / get_settings().documents_dir
RESTRICTED = {"manager-handbook": AccessLevel.MANAGER, "disciplinary-procedure": AccessLevel.HR}

RETRIEVERS: dict[str, Callable[[PolicyIndex, Embedder, list[Chunk], Any], Retriever]] = {
    "memory": lambda index, embedder, chunks, vectors: InMemoryRetriever(chunks, vectors, embedder),
    "dense": lambda index, embedder, chunks, vectors: QdrantDenseRetriever(index, embedder),
    "hybrid": lambda index, embedder, chunks, vectors: QdrantHybridRetriever(index, embedder),
    "hybrid_rerank": lambda index, embedder, chunks, vectors: RerankingRetriever(
        QdrantHybridRetriever(index, embedder), FakeReranker()
    ),
}


_parsed: dict[tuple[str, DocumentInfo], list[Chunk]] = {}
_parse_and_chunk = ingest_module._parse_and_chunk


def parse_once(path: Path, info: DocumentInfo, counter: Any) -> list[Chunk]:
    """Ingest's parse step, remembered by file content and document info."""
    key = (hashlib.sha256(path.read_bytes()).hexdigest(), info)
    if key not in _parsed:
        _parsed[key] = _parse_and_chunk(path, info, counter)
    return _parsed[key]


@pytest.fixture(scope="module")
def corpus_chunks() -> list[Chunk]:
    """Chunks of each document's newest version, as ingest would index them."""
    latest = {
        e.slug: e
        for e in sorted(load_manifest(DOCUMENTS / "manifest.yaml"), key=lambda e: e.version)
    }
    chunks: list[Chunk] = []
    for entry in latest.values():
        info = DocumentInfo(entry.slug, entry.title, entry.version, entry.access_level)
        chunks.extend(parse_once(DOCUMENTS / entry.file, info, WhitespaceTokenCounter()))
    return chunks


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    return Embedder(FakeEmbeddingClient(), tmp_path, sleep=no_sleep)


@pytest.fixture(params=list(RETRIEVERS))
async def retriever(
    request: pytest.FixtureRequest,
    policy_index: PolicyIndex,
    embedder: Embedder,
    corpus_chunks: list[Chunk],
) -> Retriever:
    vectors = await embedder.embed([c.embedding_text for c in corpus_chunks], EmbedTask.DOCUMENT)
    await policy_index.upsert(corpus_chunks, vectors)
    return RETRIEVERS[request.param](policy_index, embedder, corpus_chunks, vectors)


async def search(retriever: Retriever, role: Role, query: str) -> ToolMessage:
    [tool] = build_policy_tools(user_with_role(role), retriever)
    message = await tool.ainvoke(call(tool.name, query=query[:MAX_QUERY_LENGTH]))
    assert isinstance(message, ToolMessage)
    return message


def restricted_chunks(chunks: list[Chunk], slug: str) -> list[Chunk]:
    found = [c for c in chunks if c.doc_slug == slug]
    assert found, f"no chunks for {slug}"
    assert {c.access_level for c in found} == {RESTRICTED[slug]}
    return found


# 8.12


async def test_corpus_has_the_restricted_documents(corpus_chunks: list[Chunk]) -> None:
    for slug in RESTRICTED:
        restricted_chunks(corpus_chunks, slug)


@pytest.mark.parametrize("slug", list(RESTRICTED))
async def test_employee_gets_nothing_restricted_even_with_its_own_text(
    retriever: Retriever, corpus_chunks: list[Chunk], slug: str
) -> None:
    for chunk in restricted_chunks(corpus_chunks, slug):
        employee = await search(retriever, Role.EMPLOYEE, chunk.text)
        hr = await search(retriever, Role.HR, chunk.text)

        # The query does find the chunk for someone allowed to see it...
        assert chunk.chunk_id in {c["chunk_id"] for c in hr.artifact}
        # ...and nothing restricted, from this document or the other, for
        # an employee: not in the citations, not in the text the model reads.
        assert {c["doc_slug"] for c in employee.artifact}.isdisjoint(RESTRICTED)
        assert chunk.text not in str(employee.content)


async def test_manager_gets_the_handbook_but_not_the_disciplinary_procedure(
    retriever: Retriever, corpus_chunks: list[Chunk]
) -> None:
    for chunk in restricted_chunks(corpus_chunks, "manager-handbook"):
        manager = await search(retriever, Role.MANAGER, chunk.text)
        assert chunk.chunk_id in {c["chunk_id"] for c in manager.artifact}

    for chunk in restricted_chunks(corpus_chunks, "disciplinary-procedure"):
        manager = await search(retriever, Role.MANAGER, chunk.text)
        assert "disciplinary-procedure" not in {c["doc_slug"] for c in manager.artifact}
        assert chunk.text not in str(manager.content)


# 8.13

CARRY_FORWARD = "How many days of unused earned leave carry forward into the next leave year?"


async def test_switching_to_leave_policy_v2_serves_only_v2(
    db: AsyncSession,
    tmp_path: Path,
    policy_index: PolicyIndex,
    embedder: Embedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_module, "_parse_and_chunk", parse_once)
    folder = tmp_path / "documents"
    folder.mkdir()
    entries: list[dict[str, Any]] = []

    async def ingest_version(version: int) -> None:
        file = f"leave-policy-v{version}.pdf"
        shutil.copy(DOCUMENTS / file, folder / file)
        entries.append(
            {
                "slug": "leave-policy",
                "title": "Leave Policy",
                "access_level": "all",
                "version": version,
                "effective_date": f"{2024 + version}-01-01",
                "file": file,
            }
        )
        (folder / "manifest.yaml").write_text(yaml.safe_dump({"documents": entries}))
        await ingest(
            db,
            folder / "manifest.yaml",
            cache_dir=tmp_path / "cache",
            embedder=embedder,
            counter=WhitespaceTokenCounter(),
            index=policy_index,
        )

    retrievers = {
        kind: RETRIEVERS[kind](policy_index, embedder, [], None)
        for kind in ("dense", "hybrid", "hybrid_rerank")
    }

    await ingest_version(1)
    for kind, retriever in retrievers.items():
        before = await search(retriever, Role.EMPLOYEE, CARRY_FORWARD)
        assert {c["version"] for c in before.artifact} == {1}, kind
        assert "Up to 5 days of unused earned leave carry forward" in str(before.content), kind

    await ingest_version(2)
    for kind, retriever in retrievers.items():
        after = await search(retriever, Role.EMPLOYEE, CARRY_FORWARD)
        assert {c["version"] for c in after.artifact} == {2}, kind
        # What the model answers from has v2's limit and not v1's.
        assert "Up to 8 days of unused earned leave carry forward" in str(after.content), kind
        assert "Up to 5 days" not in str(after.content), kind

    # Not just outranked: v1's points are gone from the collection.
    records, _ = await policy_index.client.scroll(policy_index.collection, limit=1000)
    assert records
    assert {(r.payload or {})["version"] for r in records} == {2}
