"""Build the retriever the RETRIEVER setting names, ready to search."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient

from warden.core.config import Settings
from warden.rag.embeddings import Embedder
from warden.rag.index import PolicyIndexError, build_index
from warden.rag.reranker import CrossEncoderReranker
from warden.rag.retrievers.base import Retriever
from warden.rag.retrievers.memory import InMemoryRetriever
from warden.rag.retrievers.min_score import MinScoreRetriever
from warden.rag.retrievers.qdrant import QdrantDenseRetriever, QdrantHybridRetriever
from warden.rag.retrievers.reranking import RerankingRetriever
from warden.rag.sparse import Bm25Encoder
from warden.rag.store import ChunkStoreError

logger = logging.getLogger(__name__)


class RetrieverUnavailable(Exception):
    pass


@asynccontextmanager
async def open_retriever(
    settings: Settings, embedder: Embedder, *, gated: bool = True
) -> AsyncIterator[Retriever]:
    """The configured retriever. Models load and the index is checked before
    it's returned, so the first search isn't slow and a missing index shows
    up at startup.

    With `gated`, searches whose best match scores below the setting's
    threshold return nothing (not for `hybrid`, whose scores can't say).
    Raises RetrieverUnavailable if the index or a model can't be loaded.
    """
    kind = settings.retriever
    if kind == "memory":
        try:
            store = InMemoryRetriever.load(settings.cache_dir, embedder)
        except ChunkStoreError as exc:
            raise RetrieverUnavailable(str(exc)) from exc
        logger.info("Policy search: %d chunks in memory", len(store))
        yield MinScoreRetriever(store, settings.policy_min_score) if gated else store
        return

    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        bm25 = Bm25Encoder(settings.models_dir)
        index = build_index(settings, client, bm25)
        try:
            points = await index.verify()
        except PolicyIndexError as exc:
            raise RetrieverUnavailable(str(exc)) from exc

        retriever: Retriever
        if kind == "dense":
            dense = QdrantDenseRetriever(index, embedder)
            retriever = MinScoreRetriever(dense, settings.policy_min_score) if gated else dense
        else:
            await _load("BM25", bm25.load)
            retriever = QdrantHybridRetriever(index, embedder)
            if kind == "hybrid_rerank":
                reranker = CrossEncoderReranker(settings.reranker_model, settings.models_dir)
                await _load(settings.reranker_model, reranker.load)
                reranking = RerankingRetriever(
                    retriever, reranker, candidates=settings.rerank_candidates
                )
                retriever = (
                    MinScoreRetriever(reranking, settings.rerank_min_score) if gated else reranking
                )
        logger.info("Policy search: %s over %d points in Qdrant", kind, points)
        yield retriever
    finally:
        await client.close()


async def _load(name: str, load: Callable[[], None]) -> None:
    try:
        await asyncio.to_thread(load)
    except Exception as exc:
        raise RetrieverUnavailable(f"Can't load {name}: {exc}") from exc
