from pathlib import Path

import pytest

from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from warden.core.config import get_settings
from warden.rag.embeddings import Embedder
from warden.rag.retrievers.factory import RetrieverUnavailable, open_retriever


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    return Embedder(FakeEmbeddingClient(), tmp_path, sleep=no_sleep)


async def test_memory_without_a_chunk_store_is_unavailable(
    tmp_path: Path, embedder: Embedder
) -> None:
    settings = get_settings().model_copy(update={"retriever": "memory", "cache_dir": tmp_path})

    with pytest.raises(RetrieverUnavailable, match="Run `uv run python -m warden.rag.ingest`"):
        async with open_retriever(settings, embedder):
            pass


@pytest.mark.parametrize("kind", ["dense", "hybrid", "hybrid_rerank"])
async def test_qdrant_retrievers_without_qdrant_are_unavailable(
    kind: str, embedder: Embedder
) -> None:
    settings = get_settings().model_copy(
        update={"retriever": kind, "qdrant_url": "http://127.0.0.1:9"}
    )

    with pytest.raises(RetrieverUnavailable, match="Can't reach Qdrant"):
        async with open_retriever(settings, embedder):
            pass
