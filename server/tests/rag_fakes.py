"""Stand-ins for the embedding API and tokenizer, so RAG tests run offline."""

import hashlib
import re
from collections.abc import Sequence

from warden.rag.embeddings import EmbedTask, TransientEmbeddingError

DIMENSIONS = 64
_WORD = re.compile(r"\w+")


def bag_of_words(text: str) -> list[float]:
    """A vector with one count per hashed word, so texts that share words
    score higher than texts that don't."""
    vector = [0.0] * DIMENSIONS
    for word in _WORD.findall(text.lower()):
        index = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMENSIONS
        vector[index] += 1.0
    return vector


class FakeEmbeddingClient:
    def __init__(self, model_id: str = "fake/bag-of-words/64", fail_times: int = 0) -> None:
        self._model_id = model_id
        self.calls: list[tuple[list[str], EmbedTask]] = []
        # The first `fail_times` calls raise a transient error.
        self.fail_times = fail_times

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        self.calls.append((list(texts), task))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TransientEmbeddingError("429 RESOURCE_EXHAUSTED")
        return [bag_of_words(text) for text in texts]


class WhitespaceTokenCounter:
    """One token per whitespace-separated word."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in re.finditer(r"\S+", text)]


async def no_sleep(_: float) -> None:
    return None
