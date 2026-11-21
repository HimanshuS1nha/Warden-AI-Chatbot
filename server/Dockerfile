# syntax=docker/dockerfile:1

# Build: install the locked dependencies and the app into a virtualenv, and
# download the local models so the container never fetches them at startup.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# BM25 and the reranker (fastembed) and the chunking tokenizer (Hugging Face
# Hub). Paths match MODELS_DIR and HF_HOME in the runtime stage.
ENV MODELS_DIR=/opt/models \
    HF_HOME=/opt/hf
# Importing the app reads its settings, so this step gets placeholders. They
# exist only for this command and aren't kept in the image.
RUN DATABASE_URL=postgresql+asyncpg://unused/unused \
    CHECKPOINTER_DATABASE_URL=postgresql://unused/unused \
    JWT_SECRET_KEY=build-step-placeholder-not-a-real-secret \
    /app/.venv/bin/python - <<'PY'
from pathlib import Path

from warden.core.config import Settings
from warden.rag.chunking import HFTokenCounter
from warden.rag.reranker import CrossEncoderReranker
from warden.rag.sparse import Bm25Encoder

models = Path("/opt/models")
Bm25Encoder(models).load()
CrossEncoderReranker(Settings.model_fields["reranker_model"].default, models).load()
HFTokenCounter().offsets("warm up")
PY
# Some cache files are written owner-only; the app runs as another user.
RUN chmod -R a+rX /opt/models /opt/hf


# Runtime: the virtualenv, models, migrations and documents, run as a
# non-root user. Same Python base image as the build stage, so the
# virtualenv's interpreter path is valid.
FROM python:3.12-slim-bookworm

RUN groupadd --system --gid 10001 warden \
    && useradd --system --uid 10001 --gid warden --home-dir /app warden \
    && mkdir -p /data/cache /data/uploads \
    && chown -R warden:warden /data

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/models /opt/models
COPY --from=build /opt/hf /opt/hf
COPY alembic.ini ./
COPY alembic ./alembic
COPY documents ./documents

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/opt/models \
    HF_HOME=/opt/hf \
    CACHE_DIR=/data/cache \
    UPLOAD_DIR=/data/uploads

USER warden
EXPOSE 8000

# Behind the client's nginx, which is the only thing that can reach this
# port in docker-compose.yml. --proxy-headers makes the client IP (for rate
# limits and audit rows) come from the X-Forwarded-For nginx sets.
CMD ["uvicorn", "warden.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
