"""BM25 sparse vectors, the keyword half of hybrid search.

The vectors hold term frequencies only. Qdrant applies IDF itself (the
collection's sparse vector has the IDF modifier), so adding or removing a
document never means re-encoding the others.

Encoding runs locally, needs no API key and costs nothing, so unlike dense
vectors it isn't cached.
"""

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fastembed import SparseEmbedding, SparseTextEmbedding
from qdrant_client.models import SparseVector

BM25_MODEL = "Qdrant/bm25"


class SparseEncoder(Protocol):
    async def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def encode_query(self, text: str) -> SparseVector: ...


class Bm25Encoder:
    """fastembed's BM25 (English stemming and stopwords), loaded on first
    use. The first load downloads a few small files to `models_dir`."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._model: SparseTextEmbedding | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model now instead of on the first search."""
        self._get_model()

    async def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        def encode() -> list[SparseVector]:
            return [_to_qdrant(e) for e in self._get_model().embed(list(texts))]

        return await asyncio.to_thread(encode)

    async def encode_query(self, text: str) -> SparseVector:
        def encode() -> SparseVector:
            return _to_qdrant(next(iter(self._get_model().query_embed(text))))

        return await asyncio.to_thread(encode)

    def _get_model(self) -> SparseTextEmbedding:
        # Searches run in worker threads, so two could try to load at once.
        with self._lock:
            if self._model is None:
                self._model = SparseTextEmbedding(BM25_MODEL, cache_dir=str(self._models_dir))
            return self._model


def _to_qdrant(embedding: SparseEmbedding) -> SparseVector:
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
