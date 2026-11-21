"""Search over the Qdrant collection that ingest fills.

Every search filters inside Qdrant on access level, active version and
(optionally) document slug, using the payload indexes. A chunk the caller
may not see is never a candidate, so it can't take up one of the k places
or leak through a later step.
"""

from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    ScoredPoint,
)

from warden.models import AccessLevel
from warden.rag.embeddings import Embedder
from warden.rag.index import DENSE, SPARSE, PolicyIndex
from warden.rag.retrievers.base import RetrievedChunk, SearchFilters

# Candidates each half of a hybrid search contributes before fusion.
HYBRID_PREFETCH = 20


def qdrant_filter(filters: SearchFilters) -> Filter:
    must: list[Any] = [
        FieldCondition(key="is_active", match=MatchValue(value=True)),
        FieldCondition(
            key="access_level",
            match=MatchAny(any=sorted(level.value for level in filters.access_levels)),
        ),
    ]
    if filters.doc_slugs is not None:
        must.append(FieldCondition(key="doc_slug", match=MatchAny(any=sorted(filters.doc_slugs))))
    return Filter(must=must)


class QdrantDenseRetriever:
    """Nearest neighbours by the dense vector. Scores are cosine similarity."""

    def __init__(self, index: PolicyIndex, embedder: Embedder) -> None:
        self._index = index
        self._embedder = embedder

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        if k <= 0 or not filters.access_levels:
            return []
        vector = await self._embedder.embed_query(query)
        response = await self._index.client.query_points(
            self._index.collection,
            query=vector.tolist(),
            using=DENSE,
            query_filter=qdrant_filter(filters),
            limit=k,
            with_payload=True,
        )
        return [_retrieved(point) for point in response.points]


class QdrantHybridRetriever:
    """Dense and BM25 candidates fused with reciprocal rank fusion, in one
    Qdrant query. BM25 catches exact terms the embedding blurs ("form 16",
    a city name); the dense half catches paraphrases with no shared words.

    Scores are RRF scores: they rank results but say nothing about how
    relevant the best one is, so they can't gate "no relevant policy".
    """

    def __init__(
        self, index: PolicyIndex, embedder: Embedder, prefetch: int = HYBRID_PREFETCH
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._prefetch = prefetch

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        if k <= 0 or not filters.access_levels:
            return []
        dense = await self._embedder.embed_query(query)
        sparse = await self._index.sparse.encode_query(query)
        # The filter goes on each prefetch, so both candidate lists hold only
        # chunks the caller may see.
        where = qdrant_filter(filters)
        response = await self._index.client.query_points(
            self._index.collection,
            prefetch=[
                Prefetch(query=dense.tolist(), using=DENSE, filter=where, limit=self._prefetch),
                Prefetch(query=sparse, using=SPARSE, filter=where, limit=self._prefetch),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=where,
            limit=k,
            with_payload=True,
        )
        return [_retrieved(point) for point in response.points]


def _retrieved(point: ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        doc_slug=payload["doc_slug"],
        title=payload["title"],
        version=payload["version"],
        access_level=AccessLevel(payload["access_level"]),
        section=payload["section"],
        page=payload["page"],
        text=payload["text"],
        score=point.score,
    )
