from tests.rag_fakes import FakeReranker
from warden.models import AccessLevel, Role
from warden.rag.retrievers import RerankingRetriever, RetrievedChunk, SearchFilters

FILTERS = SearchFilters.for_role(Role.HR)


def chunk(n: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc:1:{n}",
        doc_slug="doc",
        title="Leave Policy",
        version=1,
        access_level=AccessLevel.ALL,
        section="Carry forward",
        page=1,
        text=text,
        score=score,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, SearchFilters]] = []

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        self.calls.append((query, k, filters))
        return self.chunks[:k]


CANDIDATES = [
    chunk(0, "hotel limits", 0.9),
    chunk(1, "sick leave certificate rules", 0.8),
    chunk(2, "sick leave", 0.7),
]


async def test_reorders_by_rerank_score_and_keeps_the_top_k() -> None:
    reranking = RerankingRetriever(StubRetriever(CANDIDATES), FakeReranker(), candidates=20)

    results = await reranking.search("sick leave certificate", 2, FILTERS)

    assert [r.chunk_id for r in results] == ["doc:1:1", "doc:1:2"]
    # The score is the rerank score now, not the base retriever's.
    assert [r.score for r in results] == [3.0, 2.0]


async def test_asks_the_base_for_candidates_with_the_same_filters() -> None:
    base = StubRetriever(CANDIDATES)
    reranking = RerankingRetriever(base, FakeReranker(), candidates=20)

    await reranking.search("sick leave", 5, FILTERS)

    assert base.calls == [("sick leave", 20, FILTERS)]


async def test_reranks_the_text_with_its_title_and_section() -> None:
    reranker = FakeReranker()
    reranking = RerankingRetriever(StubRetriever(CANDIDATES[:1]), reranker)

    await reranking.search("hotel", 1, FILTERS)

    assert reranker.calls == [("hotel", ["Leave Policy\nCarry forward\n\nhotel limits"])]


async def test_ties_keep_the_base_order() -> None:
    reranking = RerankingRetriever(StubRetriever(CANDIDATES), FakeReranker())

    results = await reranking.search("nothing matches", 3, FILTERS)

    assert [r.chunk_id for r in results] == ["doc:1:0", "doc:1:1", "doc:1:2"]


async def test_no_candidates_skips_the_reranker() -> None:
    reranker = FakeReranker()
    reranking = RerankingRetriever(StubRetriever([]), reranker)

    assert await reranking.search("sick leave", 5, FILTERS) == []
    assert reranker.calls == []
