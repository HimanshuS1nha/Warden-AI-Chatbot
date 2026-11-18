"""Chunk files on disk, under the cache directory.

- `chunks/<slug>/v<version>.json`: the chunks of one document version, with
  the hash of the file they came from. Ingest skips a version whose file
  hash matches.
- `chunk_store/`: every active version's chunks and their vectors, for
  `InMemoryRetriever`. `chunks.json` holds the chunks and the embedding
  model ID, `vectors.npy` one row per chunk in the same order.

Everything here can be rebuilt by running ingest again.
"""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from warden.models import AccessLevel
from warden.rag.chunking import Chunk
from warden.rag.embeddings import Vectors


class ChunkStoreError(Exception):
    pass


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(**{**data, "access_level": AccessLevel(data["access_level"])})


def version_chunks_path(cache_dir: Path, slug: str, version: int) -> Path:
    return cache_dir / "chunks" / slug / f"v{version}.json"


def write_version_chunks(path: Path, content_hash: str, chunks: Sequence[Chunk]) -> None:
    payload = {"content_hash": content_hash, "chunks": [chunk_to_dict(c) for c in chunks]}
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1).encode())


def read_version_chunks(path: Path) -> tuple[str, list[Chunk]] | None:
    """(content_hash, chunks), or None if the file is missing or unreadable."""
    try:
        payload = json.loads(path.read_bytes())
        return payload["content_hash"], [chunk_from_dict(c) for c in payload["chunks"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store_dir(cache_dir: Path) -> Path:
    return cache_dir / "chunk_store"


def write_store(directory: Path, chunks: Sequence[Chunk], vectors: Vectors, model_id: str) -> None:
    if len(chunks) != len(vectors):
        raise ChunkStoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    directory.mkdir(parents=True, exist_ok=True)
    # Vectors first: a reader that sees the new chunks.json also sees the new
    # vectors. The row count check in read_store catches anything in between.
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as file:
        np.save(file, vectors.astype(np.float32))
    os.replace(file.name, directory / "vectors.npy")
    payload = {
        "model_id": model_id,
        "count": len(chunks),
        "chunks": [chunk_to_dict(c) for c in chunks],
    }
    _atomic_write(directory / "chunks.json", json.dumps(payload, ensure_ascii=False).encode())


def read_store(directory: Path) -> tuple[list[Chunk], Vectors, str]:
    """(chunks, vectors, model_id) as written by `write_store`."""
    try:
        payload = json.loads((directory / "chunks.json").read_bytes())
        vectors: Vectors = np.load(directory / "vectors.npy").astype(np.float32)
    except FileNotFoundError as exc:
        raise ChunkStoreError(
            f"No chunk store in {directory}. Run `uv run python -m warden.rag.ingest`."
        ) from exc
    chunks = [chunk_from_dict(c) for c in payload["chunks"]]
    if len(chunks) != payload["count"] or len(chunks) != len(vectors):
        raise ChunkStoreError(f"Chunk store in {directory} is inconsistent. Run ingest again.")
    return chunks, vectors, payload["model_id"]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
