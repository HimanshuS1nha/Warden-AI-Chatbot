"""Retrieval quality for each retriever (task 10.5).

    uv run python -m eval.retrieval

Needs ingest to have run (chunk store and Qdrant index) and GEMINI_API_KEY
for question embeddings, which are cached after the first run.

For every question in datasets/policy_qa.jsonl, each retriever returns its
top 10 chunks, searching as the question's role. Metrics over the questions
that have an answer:

- Recall@k: share of the labelled sections found in the top k. A section
  counts as found when any of its chunks is there.
- MRR: 1 / rank of the first chunk from a labelled section, 0 if none in
  the top 10.

The search tool returns 5 chunks, so Recall@5 is what the agent sees.

Questions with no answer in the corpus check the score threshold instead:
a retriever with a threshold (all but plain hybrid, whose RRF scores only
rank) should return nothing for them, and something for every real question.
"""

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.common import DATASETS, markdown_table, open_embedder, read_jsonl, write_result
from eval.keys import chunk_ids_by_key, resolve
from warden.core.config import Settings, get_settings
from warden.models import Role
from warden.rag.retrievers import RetrievedChunk, SearchFilters
from warden.rag.retrievers.factory import open_retriever
from warden.rag.store import read_store, store_dir

RETRIEVERS = ("memory", "dense", "hybrid", "hybrid_rerank")
TOP_K = 10
DATASET = DATASETS / "policy_qa.jsonl"


@dataclass
class QuestionResult:
    id: str
    type: str
    recall_5: float | None
    recall_10: float | None
    reciprocal_rank: float | None
    rejected: bool | None
    # The best chunk's score, on the retriever's own scale.
    top_score: float | None
    top: list[str]


@dataclass
class RetrieverResult:
    retriever: str
    threshold: float | None
    recall_5: float
    recall_10: float
    mrr: float
    # Share of no-answer questions the threshold turned away, and of real
    # questions it wrongly turned away. None without a threshold.
    no_answer_rejected: float | None
    answerable_rejected: float | None
    # The highest threshold that turns away no real question (just under the
    # lowest top score among them), and the share of no-answer questions it
    # still turns away. Measured on this dataset, so it's a guide, not a
    # setting to copy blindly.
    safe_threshold: float | None
    no_answer_rejected_at_safe: float | None
    by_type: dict[str, float]
    seconds: float
    questions: list[QuestionResult] = field(default_factory=list)


def threshold(settings: Settings, kind: str) -> float | None:
    """The score below which the app's policy search returns nothing."""
    if kind in ("memory", "dense"):
        return settings.policy_min_score
    if kind == "hybrid_rerank":
        return settings.rerank_min_score
    return None


def score_question(
    question: dict[str, Any], relevant: list[set[str]], chunks: list[RetrievedChunk]
) -> tuple[float | None, float | None, float | None]:
    if not relevant:
        return None, None, None
    ids = [c.chunk_id for c in chunks]

    def recall(k: int) -> float:
        top = set(ids[:k])
        return sum(1 for section in relevant if section & top) / len(relevant)

    wanted = set().union(*relevant)
    rank = next((i for i, chunk_id in enumerate(ids, 1) if chunk_id in wanted), None)
    return recall(5), recall(10), 1 / rank if rank else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def evaluate(
    kind: str, questions: list[dict[str, Any]], settings: Settings
) -> RetrieverResult:
    chunks, _, _ = read_store(store_dir(settings.cache_dir))
    index = chunk_ids_by_key(chunks)
    cutoff = threshold(settings, kind)
    results: list[QuestionResult] = []

    started = time.perf_counter()
    async with (
        open_embedder(settings) as embedder,
        open_retriever(
            settings.model_copy(update={"retriever": kind}), embedder, gated=False
        ) as retriever,
    ):
        for q in questions:
            relevant = resolve(q["relevant"], index)
            found = await retriever.search(
                q["question"], TOP_K, SearchFilters.for_role(Role(q["role"]))
            )
            recall_5, recall_10, rr = score_question(q, relevant, found)
            rejected = None if cutoff is None else (not found or found[0].score < cutoff)
            results.append(
                QuestionResult(
                    id=q["id"],
                    type=q["type"],
                    recall_5=recall_5,
                    recall_10=recall_10,
                    reciprocal_rank=rr,
                    rejected=rejected,
                    top_score=found[0].score if found else None,
                    top=[c.chunk_id for c in found[:5]],
                )
            )
    seconds = time.perf_counter() - started

    answerable = [r for r in results if r.recall_5 is not None]
    unanswerable = [r for r in results if r.recall_5 is None]
    types = sorted({r.type for r in answerable})
    safe = None
    if cutoff is not None:
        safe = min(r.top_score if r.top_score is not None else float("-inf") for r in answerable)
    return RetrieverResult(
        retriever=kind,
        threshold=cutoff,
        recall_5=mean([r.recall_5 for r in answerable if r.recall_5 is not None]),
        recall_10=mean([r.recall_10 for r in answerable if r.recall_10 is not None]),
        mrr=mean([r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]),
        no_answer_rejected=None
        if cutoff is None
        else mean([1.0 if r.rejected else 0.0 for r in unanswerable]),
        answerable_rejected=None
        if cutoff is None
        else mean([1.0 if r.rejected else 0.0 for r in answerable]),
        safe_threshold=safe,
        no_answer_rejected_at_safe=None
        if safe is None
        else mean(
            [1.0 if r.top_score is None or r.top_score < safe else 0.0 for r in unanswerable]
        ),
        by_type={
            t: mean([r.recall_5 for r in answerable if r.type == t and r.recall_5 is not None])
            for t in types
        },
        seconds=seconds,
        questions=results,
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def tables(results: list[RetrieverResult], answerable: int, unanswerable: int) -> str:
    main = markdown_table(
        [
            "Retriever",
            "Recall@5",
            "Recall@10",
            "MRR",
            "Threshold",
            "No-answer rejected",
            "Real rejected",
            "Safe threshold",
            "No-answer rejected at safe",
        ],
        [
            [
                r.retriever,
                f"{r.recall_5:.3f}",
                f"{r.recall_10:.3f}",
                f"{r.mrr:.3f}",
                "n/a" if r.threshold is None else r.threshold,
                pct(r.no_answer_rejected),
                pct(r.answerable_rejected),
                "n/a" if r.safe_threshold is None else f"{r.safe_threshold:.2f}",
                pct(r.no_answer_rejected_at_safe),
            ]
            for r in results
        ],
    )
    types = list(results[0].by_type)
    by_type = markdown_table(
        ["Retriever", *types],
        [[r.retriever, *(f"{r.by_type[t]:.2f}" for t in types)] for r in results],
    )
    return (
        f"{answerable} questions with an answer, {unanswerable} without.\n\n"
        f"{main}\n\nRecall@5 by question type:\n\n{by_type}"
    )


async def main() -> None:
    settings = get_settings()
    questions = read_jsonl(DATASET)
    # Every label must match a chunk before anything is searched.
    chunks, _, _ = read_store(store_dir(settings.cache_dir))
    index = chunk_ids_by_key(chunks)
    for q in questions:
        resolve(q["relevant"], index)

    results = []
    for kind in RETRIEVERS:
        print(f"Evaluating {kind}...", flush=True)
        results.append(await evaluate(kind, questions, settings))

    answerable = sum(1 for q in questions if q["relevant"])
    report = tables(results, answerable, len(questions) - answerable)
    path = write_result(
        "retrieval",
        {
            "dataset": str(Path(DATASET).name),
            "questions": len(questions),
            "answerable": answerable,
            "top_k": TOP_K,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "rerank_candidates": settings.rerank_candidates,
            "results": [asdict(r) for r in results],
            "markdown": report,
        },
    )
    print(f"\n{report}\n\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main())
