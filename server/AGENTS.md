# Warden server

FastAPI backend, LangGraph agent and retrieval pipeline. Python 3.12, managed with uv. Run every command from this folder.

```bash
uv run fastapi dev                       # API on :8000, docs at /docs
uv run pytest                            # needs Postgres and Redis (docker-compose.dev.yml)
uv run ruff check && uv run ruff format --check && uv run mypy
uv run python -m warden.scripts.seed --reset
uv run python -m warden.rag.ingest
uv run python -m eval.security           # calls the real model; see eval/
```

## Layout

```
src/warden/
  main.py           create_app(): middleware, exception handlers, routers, lifespan
  core/             config (Settings, env vars), DB session, JWT and cookies,
                    CurrentUser and require_role, rate limits, request context, logging
  models/           SQLAlchemy models; enums.py has roles, statuses, audit actions
  schemas/          Pydantic request and response models
  repositories/     SQL queries. Scoped queries live here (e.g. a manager's reports)
  services/         business rules and authorization; errors.py, audit.py
  routers/          thin REST endpoints: parse, call a service, return a schema
  graph/            the agent: graph.py (loop and ToolCallGuard), agent.py (model node),
                    prompts.py, tools/ (per-role tool factories), pending_actions.py,
                    streaming.py (ChatTurn, SSE events), llm_calls.py
  rag/              parsing, cleaning, chunking, embeddings (disk cache), Qdrant index,
                    retrievers/ (memory, dense, hybrid, rerank), citations.py, ingest.py
  scripts/          seed.py and company.py (the fictional company's facts)
documents/          policy PDFs and manifest.yaml (the RAG corpus)
alembic/            migrations
eval/               retrieval and security evaluation: datasets, fixtures, results
tests/              unit/ (no database) and integration/ (real Postgres, real Redis)
```

## Design rules

- **The model is untrusted.** It chooses which tool to call, never whose data. Tool factories take the `CurrentUser` from the session and return tools with no identity arguments (HR's lookup tools are the one exception: their ID is the person being looked up).
- **A role's tool list is its permission set.** A tool a role doesn't get is one the model never sees (`graph/tools/__init__.py`).
- **Services enforce everything.** They take `CurrentUser` first, check the role again (`services/access.py`) and scope every query, because agent tools call them without a route in between. REST routes and tools share the same services.
- **Out of scope is `NotFound`,** the same as missing, so probing reveals nothing. Errors are `ServiceError` subclasses with a stable `code` (`services/errors.py`); routers and tools turn them into responses.
- **Writes are proposals.** Write tools validate, then save a `PendingAction` and return a summary. Only `POST /chat/actions/{id}/confirm`, a click by the user, runs the change, re-checking every rule (`graph/pending_actions.py`).
- **Tool output is data.** The model sees tool results inside `<tool_result>` tags and policy text inside `<untrusted_document>` tags, with closing tags in the data defused. Typed `[App]` tags in user messages are defused too (`prompts.user_message`).
- **Retrieval filters by access level inside the search**, from `SearchFilters.for_role(user.role)`, never from model output.
- **Audit by the rules in the `services/audit.py` docstring:** record changes in the same transaction, commit refusals before raising, never store secrets in `details`.
- **Leave rules match the policy PDFs.** Numbers live in `scripts/company.py`; when they disagree with `documents/`, the code is wrong.

## Conventions

- IDs are UUIDv7 generated in Python; timestamps are UTC; "today" is the company's date (`services.leave.company_today`).
- Every error response is `{"message": ...}` (plus `code`, `details` or `errors`); see `core/exceptions.py`. Never return tracebacks or another user's data.
- Comments and docstrings explain why, in plain sentences. Keep them current when behavior changes.
- mypy runs in strict mode on `src/warden`; ruff includes the bandit (`S`) rules.
- Schema changes need an Alembic migration (`uv run alembic revision --autogenerate -m "..."`), reviewed by hand.

## Tests

- Integration tests run in a transaction that's rolled back, against a `<db>_test` database created and migrated on first run. App `commit()` calls become savepoint releases.
- Agent tests use `tests/agent_fakes.py` (`script(...)`, `call(...)`) instead of a real model.
- Rate limiting is off for the shared test app; `tests/integration/test_rate_limit.py` turns it on against Redis database 15.
- Changing the agent, prompts or tools? Re-run `uv run python -m eval.security`; it uses its own `<db>_eval` database.
