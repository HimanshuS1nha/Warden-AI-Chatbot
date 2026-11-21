"""Dense embeddings with an on-disk cache.

`Embedder` sends texts to an `EmbeddingClient` in batches, retries rate limits
and server errors with exponential backoff, and caches every vector in
`<cache_dir>/embeddings/`. A cached text never goes to the API again, so
re-running ingest costs nothing unless the text or the model changes.

The cache key is sha256 of the client's `model_id` (model and dimensions),
the task and the text. The task is part of the key because Gemini embeds a
query and a document differently, even for the same text.

Vectors come back L2-normalized, so a dot product is the cosine similarity.
"""

import asyncio
import hashlib
import logging
import os
import random
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import numpy.typing as npt
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from warden.core.config import Settings

logger = logging.getLogger(__name__)

Vectors = npt.NDArray[np.float32]


class EmbedTask(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingError(Exception):
    pass


class TransientEmbeddingError(EmbeddingError):
    """A failure that may succeed on retry: rate limit, server error or network."""


class EmbeddingClient(Protocol):
    @property
    def model_id(self) -> str:
        """Identifies the vectors this client returns. Vectors from clients
        with different IDs can't be compared."""
        ...

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]: ...


class GeminiEmbeddingClient:
    _TASK_TYPES = {EmbedTask.DOCUMENT: "RETRIEVAL_DOCUMENT", EmbedTask.QUERY: "RETRIEVAL_QUERY"}

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"gemini/{self._model}/{self._dimensions}"

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(
            task_type=self._TASK_TYPES[task], output_dimensionality=self._dimensions
        )
        # Annotated so the list takes the SDK's element type, not list[str].
        contents: list[genai_types.PartUnion] = list(texts)
        try:
            response = await self._client.aio.models.embed_content(
                model=self._model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientEmbeddingError(str(exc)) from exc
            raise EmbeddingError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise TransientEmbeddingError(str(exc)) from exc

        vectors = [embedding.values for embedding in response.embeddings or []]
        if len(vectors) != len(texts) or any(v is None for v in vectors):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return [v for v in vectors if v is not None]

    async def aclose(self) -> None:
        await self._client.aio.aclose()


def build_embedding_client(settings: Settings) -> GeminiEmbeddingClient:
    api_key = settings.gemini_api_key.get_secret_value()
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY is not set. Add it to server/.env.")
    return GeminiEmbeddingClient(api_key, settings.embedding_model, settings.embedding_dimensions)


class Embedder:
    def __init__(
        self,
        client: EmbeddingClient,
        cache_dir: Path,
        *,
        batch_size: int = 100,
        max_attempts: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self._cache_dir = cache_dir / "embeddings"
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        # Counted for ingest's log line and tests.
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def model_id(self) -> str:
        return self.client.model_id

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> Vectors:
        """One normalized vector per text, as an (n, dimensions) array."""
        keys = [self._key(text, task) for text in texts]
        found: dict[str, npt.NDArray[np.float32]] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key in found or key in missing:
                continue
            vector = self._read(key)
            if vector is None:
                missing[key] = text
            else:
                found[key] = vector
                self.cache_hits += 1

        pending = list(missing.items())
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            vectors = await self._embed_with_retry([text for _, text in batch], task)
            for (key, _), values in zip(batch, vectors, strict=True):
                vector = np.asarray(values, dtype=np.float32)
                self._write(key, vector)
                found[key] = vector

        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([found[key] for key in keys])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized: Vectors = (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)
        return normalized

    async def embed_query(self, text: str) -> Vectors:
        vectors = await self.embed([text], EmbedTask.QUERY)
        query: Vectors = vectors[0]
        return query

    async def _embed_with_retry(self, texts: list[str], task: EmbedTask) -> list[list[float]]:
        for attempt in range(1, self._max_attempts + 1):
            self.api_calls += 1
            try:
                return await self.client.embed(texts, task)
            except TransientEmbeddingError as exc:
                if attempt == self._max_attempts:
                    raise
                delay = min(self._max_delay, self._base_delay * 2 ** (attempt - 1))
                # Jitter spreads out retries from parallel runs. Not security relevant.
                delay *= random.uniform(0.5, 1.0)  # noqa: S311
                logger.warning(
                    "Embedding request failed (attempt %d of %d), retrying in %.1fs: %s",
                    attempt,
                    self._max_attempts,
                    delay,
                    exc,
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")

    def _key(self, text: str, task: EmbedTask) -> str:
        payload = f"{self.client.model_id}\0{task.value}\0{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _path(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.npy"

    def _read(self, key: str) -> npt.NDArray[np.float32] | None:
        path = self._path(key)
        try:
            vector: npt.NDArray[np.float32] = np.load(path)
        except (OSError, ValueError):
            return None
        return vector

    def _write(self, key: str, vector: npt.NDArray[np.float32]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename, so an interrupted run never
        # leaves a half-written vector behind.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as file:
                np.save(file, vector)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
