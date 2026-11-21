"""Shared by the evaluation scripts."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from warden.core.config import Settings, get_settings
from warden.rag.embeddings import Embedder, build_embedding_client

EVAL_DIR = Path(__file__).parent
DATASETS = EVAL_DIR / "datasets"
FIXTURES = EVAL_DIR / "fixtures"
RESULTS = EVAL_DIR / "results"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_result(name: str, data: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def require_api_key(settings: Settings) -> None:
    if not settings.gemini_api_key.get_secret_value():
        raise SystemExit("GEMINI_API_KEY is not set. The evaluation calls the real models.")


@asynccontextmanager
async def open_embedder(settings: Settings | None = None) -> AsyncIterator[Embedder]:
    """The app's embedder, with its on-disk cache: a question embedded once
    isn't sent to the API again."""
    settings = settings or get_settings()
    require_api_key(settings)
    client = build_embedding_client(settings)
    try:
        yield Embedder(client, settings.cache_dir, batch_size=settings.embedding_batch_size)
    finally:
        await client.aclose()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
