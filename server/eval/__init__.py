"""Offline evaluation of retrieval and the agent's security (plan.md,
"Evaluation"). Run from the server folder:

    uv run python -m eval.retrieval
    uv run python -m eval.security
    uv run python -m eval.report

Results go to eval/results/ as JSON, and report.py turns them into
eval/results/summary.md.
"""
