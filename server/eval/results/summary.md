# Evaluation summary

Generated 2026-11-21 by `uv run python -m eval.report`.

## Retrieval

62 questions (`eval/datasets/policy_qa.jsonl`), labelled with the sections that answer them. Embeddings: `gemini-embedding-001`. Reranker: `Xenova/ms-marco-MiniLM-L-6-v2` over 20 hybrid candidates. Policy search shows the agent 5 chunks, so Recall@5 is what it sees.

56 questions with an answer, 6 without.

| Retriever | Recall@5 | Recall@10 | MRR | Threshold | No-answer rejected | Real rejected | Safe threshold | No-answer rejected at safe |
|---|---|---|---|---|---|---|---|---|
| memory | 0.991 | 0.991 | 0.978 | 0.6 | 33% | 0% | 0.62 | 33% |
| dense | 0.991 | 0.991 | 0.978 | 0.6 | 33% | 0% | 0.62 | 33% |
| hybrid | 0.991 | 1.000 | 0.917 | n/a | n/a | n/a | n/a | n/a |
| hybrid_rerank | 0.973 | 1.000 | 0.919 | -5.0 | 83% | 21% | -10.61 | 50% |

Recall@5 by question type:

| Retriever | lookup | multi | paraphrase | restricted | table |
|---|---|---|---|---|---|
| memory | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| dense | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| hybrid | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| hybrid_rerank | 0.93 | 0.92 | 1.00 | 1.00 | 1.00 |

*Threshold* is the app's setting: below it, policy search returns nothing. *Safe threshold* is the highest one that would turn away no real question in this dataset.

## Security

31 attacks (`eval/datasets/attacks.jsonl`) against the real agent and model (`google_genai:gemini-3.8-flash`), each on a freshly seeded database. Judged by the database, the tool results and the reply, not by asking a model (see the `eval/security.py` docstring for every check).

Unauthorized writes: 0. Unauthorized data accesses: 0. Attacks with an error: 0.

| Category | Attacks | Passed | Failed |
|---|---|---|---|
| impersonation | 5 | 5 | 0 |
| scope_probing | 5 | 5 | 0 |
| role_escalation | 6 | 6 | 0 |
| indirect_injection | 6 | 6 | 0 |
| exfiltration | 4 | 4 | 0 |
| self_confirmation | 5 | 5 | 0 |
| **Total** | 31 | 31 | 0 |
