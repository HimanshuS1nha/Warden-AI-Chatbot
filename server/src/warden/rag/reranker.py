"""Cross-encoder reranking.

A cross-encoder reads the query and a passage together and scores how well
the passage answers it. That's more accurate than comparing two vectors made
separately, and too slow to run over the whole corpus, so it reorders the
few candidates a cheaper search found.

Scores are the model's raw logits: higher is more relevant, and unlike RRF
scores they mean the same thing from one query to the next, so a threshold
on the best one can tell "no relevant policy" apart.
"""

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker(Protocol):
    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """One score per passage, in the passages' order."""
        ...


class CrossEncoderReranker:
    """A fastembed cross-encoder. Call `load()` at startup: the first load
    downloads the model to `models_dir`, and later ones still take a moment."""

    def __init__(self, model: str, models_dir: Path) -> None:
        self._model_name = model
        self._models_dir = models_dir
        self._model: TextCrossEncoder | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        self._get_model()

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        def score() -> list[float]:
            return [float(s) for s in self._get_model().rerank(query, list(passages))]

        return await asyncio.to_thread(score)

    def _get_model(self) -> TextCrossEncoder:
        with self._lock:
            if self._model is None:
                self._model = TextCrossEncoder(self._model_name, cache_dir=str(self._models_dir))
            return self._model
