from pathlib import Path

import numpy as np
import pytest

from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from warden.rag.embeddings import Embedder, EmbeddingError, EmbedTask, TransientEmbeddingError


def make_embedder(
    tmp_path: Path, client: FakeEmbeddingClient | None = None, **kwargs: int
) -> Embedder:
    return Embedder(client or FakeEmbeddingClient(), tmp_path, sleep=no_sleep, **kwargs)


async def test_vectors_are_normalized_and_in_input_order(tmp_path: Path) -> None:
    embedder = make_embedder(tmp_path)
    vectors = await embedder.embed(["casual leave", "sick leave", "travel"], EmbedTask.DOCUMENT)
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    # "casual leave" and "sick leave" share a word, so they're closer to each
    # other than to "travel".
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


async def test_second_run_reads_the_cache_and_calls_no_api(tmp_path: Path) -> None:
    texts = ["casual leave", "sick leave"]
    first = make_embedder(tmp_path)
    expected = await first.embed(texts, EmbedTask.DOCUMENT)
    assert first.api_calls == 1

    # A new embedder, as in a second ingest run, with the same cache folder.
    client = FakeEmbeddingClient()
    second = make_embedder(tmp_path, client)
    vectors = await second.embed(texts, EmbedTask.DOCUMENT)

    assert client.calls == []
    assert second.api_calls == 0
    assert second.cache_hits == 2
    np.testing.assert_array_equal(vectors, expected)


async def test_cache_is_keyed_by_model_and_task(tmp_path: Path) -> None:
    await make_embedder(tmp_path).embed(["leave"], EmbedTask.DOCUMENT)

    other_task = make_embedder(tmp_path)
    await other_task.embed(["leave"], EmbedTask.QUERY)
    assert other_task.api_calls == 1

    other_model = make_embedder(tmp_path, FakeEmbeddingClient(model_id="fake/other"))
    await other_model.embed(["leave"], EmbedTask.DOCUMENT)
    assert other_model.api_calls == 1


async def test_only_uncached_and_unique_texts_are_sent(tmp_path: Path) -> None:
    await make_embedder(tmp_path).embed(["cached"], EmbedTask.DOCUMENT)
    client = FakeEmbeddingClient()
    embedder = make_embedder(tmp_path, client)

    vectors = await embedder.embed(["new", "cached", "new"], EmbedTask.DOCUMENT)

    assert client.calls == [(["new"], EmbedTask.DOCUMENT)]
    np.testing.assert_array_equal(vectors[0], vectors[2])


async def test_texts_are_sent_in_batches(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()
    embedder = make_embedder(tmp_path, client, batch_size=2)
    await embedder.embed(["a", "b", "c", "d", "e"], EmbedTask.DOCUMENT)
    assert [texts for texts, _ in client.calls] == [["a", "b"], ["c", "d"], ["e"]]


async def test_transient_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    client = FakeEmbeddingClient(fail_times=2)
    embedder = Embedder(client, tmp_path, base_delay=2.0, max_delay=60.0, sleep=record)

    vectors = await embedder.embed(["leave"], EmbedTask.DOCUMENT)

    assert vectors.shape == (1, 64)
    assert embedder.api_calls == 3
    # Exponential with jitter: half to all of 2s, then of 4s.
    assert 1.0 <= delays[0] <= 2.0
    assert 2.0 <= delays[1] <= 4.0


async def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(fail_times=10)
    embedder = make_embedder(tmp_path, client, max_attempts=3)
    with pytest.raises(TransientEmbeddingError):
        await embedder.embed(["leave"], EmbedTask.DOCUMENT)
    assert len(client.calls) == 3


async def test_non_transient_errors_are_not_retried(tmp_path: Path) -> None:
    class BadKeyClient(FakeEmbeddingClient):
        async def embed(self, texts: object, task: EmbedTask) -> list[list[float]]:
            self.calls.append(([], task))
            raise EmbeddingError("400 API key not valid")

    client = BadKeyClient()
    with pytest.raises(EmbeddingError):
        await make_embedder(tmp_path, client).embed(["leave"], EmbedTask.DOCUMENT)
    assert len(client.calls) == 1


async def test_corrupt_cache_file_is_re_embedded(tmp_path: Path) -> None:
    await make_embedder(tmp_path).embed(["leave"], EmbedTask.DOCUMENT)
    for path in (tmp_path / "embeddings").rglob("*.npy"):
        path.write_bytes(b"not a numpy file")

    embedder = make_embedder(tmp_path)
    await embedder.embed(["leave"], EmbedTask.DOCUMENT)
    assert embedder.api_calls == 1


async def test_embed_query_returns_one_vector(tmp_path: Path) -> None:
    vector = await make_embedder(tmp_path).embed_query("how many casual leaves")
    assert vector.shape == (64,)
