from pathlib import Path

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from warden.models import AccessLevel, Role
from warden.rag.chunking import Chunk
from warden.rag.embeddings import Embedder, EmbedTask
from warden.rag.index import PolicyIndex
from warden.rag.retrievers import (
    QdrantDenseRetriever,
    QdrantHybridRetriever,
    Retriever,
    SearchFilters,
)


def make_chunk(slug: str, index: int, text: str, access: AccessLevel, version: int = 1) -> Chunk:
    return Chunk(
        doc_slug=slug,
        title=slug.replace("-", " ").title(),
        version=version,
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
async def index(policy_index: PolicyIndex, embedder: Embedder) -> PolicyIndex:
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    await policy_index.upsert(CHUNKS, vectors)
    return policy_index


@pytest.fixture(params=["dense", "hybrid"])
def retriever(request: pytest.FixtureRequest, index: PolicyIndex, embedder: Embedder) -> Retriever:
    if request.param == "dense":
        return QdrantDenseRetriever(index, embedder)
    return QdrantHybridRetriever(index, embedder)


async def test_returns_best_matches_first(retriever: Retriever) -> None:
    results = await retriever.search("hotel limits metro", 2, SearchFilters.for_role(Role.HR))

    assert results[0].chunk_id == "travel-policy:1:0"
    assert results[0].score > results[1].score
    assert results[0].title == "Travel Policy"
    assert results[0].access_level is AccessLevel.ALL
    assert results[0].section == "1. Section"
    assert results[0].page == 1
    assert results[0].text == "hotel limits for metro cities"


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
    retriever: Retriever, role: Role, allowed: set[str]
) -> None:
    results = await retriever.search("casual leave", 10, SearchFilters.for_role(role))

    assert {r.doc_slug for r in results} == allowed


async def test_employee_never_gets_restricted_text_even_with_its_exact_words(
    retriever: Retriever,
) -> None:
    results = await retriever.search(
        "casual leave misuse hearing", 10, SearchFilters.for_role(Role.EMPLOYEE)
    )

    assert all(r.access_level is AccessLevel.ALL for r in results)


async def test_doc_slug_filter(retriever: Retriever) -> None:
    filters = SearchFilters(
        access_levels=frozenset(AccessLevel), doc_slugs=frozenset({"leave-policy"})
    )

    results = await retriever.search("casual leave", 10, filters)

    assert {r.doc_slug for r in results} == {"leave-policy"}


async def test_inactive_points_are_never_returned(retriever: Retriever, index: PolicyIndex) -> None:
    await index.client.set_payload(
        index.collection,
        {"is_active": False},
        points=Filter(
            must=[FieldCondition(key="doc_slug", match=MatchValue(value="travel-policy"))]
        ),
        wait=True,
    )

    results = await retriever.search("hotel limits metro", 10, SearchFilters.for_role(Role.HR))

    assert "travel-policy" not in {r.doc_slug for r in results}


async def test_k_limits_results(retriever: Retriever) -> None:
    filters = SearchFilters.for_role(Role.HR)

    assert len(await retriever.search("leave", 2, filters)) == 2
    assert await retriever.search("leave", 0, filters) == []


async def test_hybrid_finds_a_rare_term_dense_search_misses(
    index: PolicyIndex, embedder: Embedder
) -> None:
    # With bag-of-words embeddings, the repeated "casual" dominates the
    # dense vector. BM25 weights terms by rarity, so "medical", which only
    # one chunk has, still counts.
    query, filters = "casual casual medical", SearchFilters.for_role(Role.HR)

    dense = await QdrantDenseRetriever(index, embedder).search(query, 3, filters)
    hybrid = await QdrantHybridRetriever(index, embedder).search(query, 3, filters)

    assert "leave-policy:1:1" not in [r.chunk_id for r in dense]
    assert "leave-policy:1:1" in [r.chunk_id for r in hybrid]
