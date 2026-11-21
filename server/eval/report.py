"""Combine the evaluation results into eval/results/summary.md (task 10.10).

    uv run python -m eval.report

Reads the JSON that eval.retrieval and eval.security wrote. Run those first;
a missing result is noted in the summary rather than failing.
"""

import json
from datetime import UTC, datetime
from typing import Any

from eval.common import RESULTS


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def retrieval_section(data: dict[str, Any] | None) -> str:
    if data is None:
        return "Not run yet: `uv run python -m eval.retrieval`."
    return (
        f"{data['questions']} questions (`eval/datasets/{data['dataset']}`), labelled with the "
        f"sections that answer them. Embeddings: `{data['embedding_model']}`. Reranker: "
        f"`{data['reranker_model']}` over {data['rerank_candidates']} hybrid candidates. "
        "Policy search shows the agent 5 chunks, so Recall@5 is what it sees.\n\n"
        f"{data['markdown']}\n\n"
        "*Threshold* is the app's setting: below it, policy search returns nothing. "
        "*Safe threshold* is the highest one that would turn away no real question in this "
        "dataset."
    )


def security_section(data: dict[str, Any] | None) -> str:
    if data is None:
        return "Not run yet: `uv run python -m eval.security`."
    return (
        f"{data['attacks']} attacks (`eval/datasets/{data['dataset']}`) against the real agent "
        f"and model (`{data['model']}`), each on a freshly seeded database. Judged by the "
        "database, the tool results and the reply, not by asking a model (see the "
        "`eval/security.py` docstring for every check).\n\n"
        f"{data['markdown']}"
    )


def main() -> None:
    retrieval = load("retrieval")
    security = load("security")
    summary = "\n".join(
        [
            "# Evaluation summary",
            "",
            f"Generated {datetime.now(UTC):%Y-%m-%d} by `uv run python -m eval.report`.",
            "",
            "## Retrieval",
            "",
            retrieval_section(retrieval),
            "",
            "## Security",
            "",
            security_section(security),
            "",
        ]
    )
    path = RESULTS / "summary.md"
    path.write_text(summary)
    print(summary)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
