from warden.rag.retrievers.base import RetrievedChunk, Retriever, SearchFilters


class MinScoreRetriever:
    """Returns nothing when even the best match scores below `min_score`.

    A dense search always returns its k nearest chunks, however far they
    are: "how to bake sourdough bread" still finds the leave policy. Gating
    on the best score turns that into "no relevant policy", which the agent
    reports honestly and the chat logs as an unanswered query.

    Only the best score is checked. When the question is relevant, the
    weaker chunks after it are kept: they're often the neighbouring sections
    the answer needs.
    """

    def __init__(self, retriever: Retriever, min_score: float) -> None:
        self.retriever = retriever
        self.min_score = min_score

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        chunks = await self.retriever.search(query, k, filters)
        if not chunks or chunks[0].score < self.min_score:
            return []
        return chunks
