import dataclasses

from warden.rag.reranker import Reranker
from warden.rag.retrievers.base import RetrievedChunk, Retriever, SearchFilters

RERANK_CANDIDATES = 20


class RerankingRetriever:
    """Fetches `candidates` chunks from `base`, scores each against the query
    with a cross-encoder and returns the top k by that score. Each returned
    chunk's score is its rerank score."""

    def __init__(
        self, base: Retriever, reranker: Reranker, candidates: int = RERANK_CANDIDATES
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        chunks = await self.base.search(query, max(k, self.candidates), filters)
        if not chunks:
            return []
        scores = await self.reranker.rerank(query, [chunk.headed_text for chunk in chunks])
        ranked = sorted(
            zip(scores, range(len(chunks)), chunks, strict=True),
            key=lambda item: (-item[0], item[1]),
        )
        return [dataclasses.replace(chunk, score=score) for score, _, chunk in ranked[:k]]
