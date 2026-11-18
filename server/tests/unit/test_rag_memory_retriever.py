from pathlib import Path

import pytest

from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from warden.models import AccessLevel, Role
from warden.rag.chunking import Chunk
from warden.rag.embeddings import Embedder, EmbedTask
from warden.rag.retrievers import InMemoryRetriever, SearchFilters
from warden.rag.store import ChunkStoreError, store_dir, write_store


def make_chunk(slug: str, index: int, text: str, access: AccessLevel) -> Chunk:
    return Chunk(
        doc_slug=slug,
        title=slug.replace("-", " ").title(),
        version=1,
        access_level=access,
        chunk_index=index,
        section="1. Section",
        page=index + 1,
        text=text,
        token_count=len(text.split()),
    )


CHUNKS = [
    make_chunk("leave-policy", 0, "casual leave is twelve days a year", AccessLevel.ALL),
    make_chunk("leave-policy", 1, "sick leave needs a medical certificate", AccessLevel.ALL),
    make_chunk("manager-handbook", 0, "approve casual leave for your team", AccessLevel.MANAGER),
    make_chunk("disciplinary-procedure", 0, "casual leave misuse hearing", AccessLevel.HR),
    make_chunk("travel-policy", 0, "hotel limits for metro cities", AccessLevel.ALL),
]


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    return Embedder(FakeEmbeddingClient(), tmp_path, sleep=no_sleep)


@pytest.fixture
async def retriever(embedder: Embedder) -> InMemoryRetriever:
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    return InMemoryRetriever(CHUNKS, vectors, embedder)


async def test_returns_best_matches_first(retriever: InMemoryRetriever) -> None:
    results = await retriever.search("hotel limits metro", 2, SearchFilters.for_role(Role.HR))
    assert results[0].chunk_id == "travel-policy:1:0"
    assert results[0].score >= results[1].score
    assert results[0].title == "Travel Policy"
    assert results[0].page == 1


@pytest.mark.parametrize(
    ("role", "allowed"),
    [
        (Role.EMPLOYEE, {"leave-policy", "travel-policy"}),
        (Role.MANAGER, {"leave-policy", "travel-policy", "manager-handbook"}),
        (
            Role.HR,
            {"leave-policy", "travel-policy", "manager-handbook", "disciplinary-procedure"},
        ),
    ],
)
async def test_access_level_filter_by_role(
    retriever: InMemoryRetriever, role: Role, allowed: set[str]
) -> None:
    results = await retriever.search("casual leave", 10, SearchFilters.for_role(role))
    assert {r.doc_slug for r in results} == allowed


async def test_employee_never_gets_restricted_text_even_with_its_exact_words(
    retriever: InMemoryRetriever,
) -> None:
    results = await retriever.search(
        "casual leave misuse hearing", 5, SearchFilters.for_role(Role.EMPLOYEE)
    )
    assert all(r.access_level is AccessLevel.ALL for r in results)


async def test_doc_slug_filter(retriever: InMemoryRetriever) -> None:
    filters = SearchFilters(
        access_levels=frozenset(AccessLevel), doc_slugs=frozenset({"leave-policy"})
    )
    results = await retriever.search("casual leave", 10, filters)
    assert [r.chunk_id for r in results] == ["leave-policy:1:0", "leave-policy:1:1"]


async def test_k_limits_results(retriever: InMemoryRetriever) -> None:
    assert len(await retriever.search("leave", 2, SearchFilters.for_role(Role.HR))) == 2
    assert await retriever.search("leave", 0, SearchFilters.for_role(Role.HR)) == []


async def test_no_allowed_chunks_skips_the_query_embedding(
    retriever: InMemoryRetriever, embedder: Embedder
) -> None:
    calls = embedder.api_calls
    assert await retriever.search("leave", 5, SearchFilters(access_levels=frozenset())) == []
    assert embedder.api_calls == calls


async def test_load_round_trips_the_chunk_store(tmp_path: Path, embedder: Embedder) -> None:
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    write_store(store_dir(tmp_path), CHUNKS, vectors, embedder.model_id)

    retriever = InMemoryRetriever.load(tmp_path, embedder)

    assert len(retriever) == len(CHUNKS)
    results = await retriever.search("hotel limits", 1, SearchFilters.for_role(Role.EMPLOYEE))
    assert results[0].doc_slug == "travel-policy"


async def test_load_refuses_a_store_from_another_model(tmp_path: Path, embedder: Embedder) -> None:
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    write_store(store_dir(tmp_path), CHUNKS, vectors, "gemini/other-model/768")
    with pytest.raises(ChunkStoreError, match="Run ingest again"):
        InMemoryRetriever.load(tmp_path, embedder)


def test_load_without_a_store_says_to_run_ingest(tmp_path: Path, embedder: Embedder) -> None:
    with pytest.raises(ChunkStoreError, match="warden.rag.ingest"):
        InMemoryRetriever.load(tmp_path, embedder)
