"""Brute-force dense search over the chunk store, in process.

Fine for a corpus of a few hundred chunks. It stays as the baseline in the
retriever comparison once Qdrant takes over (Phase 8).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from warden.rag.chunking import Chunk
from warden.rag.embeddings import Embedder, Vectors
from warden.rag.retrievers.base import RetrievedChunk, SearchFilters
from warden.rag.store import ChunkStoreError, read_store, store_dir


class InMemoryRetriever:
    def __init__(self, chunks: Sequence[Chunk], vectors: Vectors, embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._chunks = list(chunks)
        self._vectors = vectors
        self._embedder = embedder

    @classmethod
    def load(cls, cache_dir: Path, embedder: Embedder) -> "InMemoryRetriever":
        """Load the chunk store that ingest wrote. Its vectors must come from
        the same embedding model as `embedder`, or scores would be meaningless."""
        chunks, vectors, model_id = read_store(store_dir(cache_dir))
        if model_id != embedder.model_id:
            raise ChunkStoreError(
                f"Chunk store was embedded with {model_id}, but the configured model is "
                f"{embedder.model_id}. Run ingest again."
            )
        return cls(chunks, vectors, embedder)

    def __len__(self) -> int:
        return len(self._chunks)

    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        allowed = np.fromiter(
            (
                chunk.access_level in filters.access_levels
                and (filters.doc_slugs is None or chunk.doc_slug in filters.doc_slugs)
                for chunk in self._chunks
            ),
            dtype=bool,
            count=len(self._chunks),
        )
        candidates = np.flatnonzero(allowed)
        if k <= 0 or len(candidates) == 0:
            return []

        query_vector = await self._embedder.embed_query(query)
        scores = self._vectors[candidates] @ query_vector
        best = np.argsort(-scores, kind="stable")[:k]
        return [_retrieved(self._chunks[candidates[i]], float(scores[i])) for i in best]


def _retrieved(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_slug=chunk.doc_slug,
        title=chunk.title,
        version=chunk.version,
        access_level=chunk.access_level,
        section=chunk.section,
        page=chunk.page,
        text=chunk.text,
        score=score,
    )
