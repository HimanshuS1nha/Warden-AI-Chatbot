from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where the app runs. Behavior meant only for production checks for "prod".
    env: Literal["dev", "prod"] = "dev"

    # SQLAlchemy URL with the asyncpg driver, for the app's models.
    database_url: str
    # Plain libpq URL (postgresql://...) for the agent's checkpointer, which
    # uses psycopg. Usually the same database as database_url.
    checkpointer_database_url: str
    frontend_url: str = "http://localhost:5173"
    # Rate limits.
    redis_url: str = "redis://localhost:6379/0"
    # Vector store for policy chunks. 6333 is Qdrant's HTTP port.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "policy_chunks"

    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    cookie_secure: bool = False

    log_level: str = "INFO"

    # Paths are relative to the server folder, where commands run (like .env).
    documents_dir: Path = Path("documents")
    # Embeddings cache and chunk store. Safe to delete: ingest rebuilds it.
    cache_dir: Path = Path(".cache")
    # Local models (BM25, reranker), downloaded on first use.
    models_dir: Path = Path(".cache/models")

    # One key per provider, shared by everything that calls it: embeddings
    # and the agent's chat model.
    gemini_api_key: SecretStr = SecretStr("")

    # The agent's chat model, passed to LangChain's init_chat_model.
    llm_provider: Literal["google_genai"] = "google_genai"
    llm_model: str = "gemini-3.8-flash"

    embedding_provider: Literal["gemini"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    # gemini-embedding-001 returns 3072 dimensions by default and can return
    # fewer. 768 keeps the cache and index small with little loss in quality.
    embedding_dimensions: int = Field(default=768, gt=0)
    # Gemini accepts at most 100 texts per request.
    embedding_batch_size: int = Field(default=100, gt=0, le=100)
    # Which search the agent's policy tool runs:
    #   memory        - in-process cosine search over the chunk store (baseline)
    #   dense         - Qdrant, dense vectors only
    #   hybrid        - Qdrant, dense and BM25 fused with RRF
    #   hybrid_rerank - hybrid candidates reordered by a cross-encoder
    retriever: Literal["memory", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    # For memory and dense: policy search finds nothing when its best match
    # scores below this (cosine similarity with the embedding model above).
    # Measured on the demo documents: relevant questions score 0.70+,
    # off-topic ones ("how to bake bread") 0.51-0.59. Change it with the
    # embedding model. Hybrid search has no threshold: RRF scores only rank.
    policy_min_score: float = Field(default=0.6, ge=0, le=1)

    # A fastembed cross-encoder, for hybrid_rerank.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Hybrid results the cross-encoder scores before the top k are kept.
    rerank_candidates: int = Field(default=20, gt=0)
    # For hybrid_rerank: policy search finds nothing when the best rerank
    # score is below this. The scale is the cross-encoder's raw logits, so
    # change it with the reranker model. Measured with MiniLM-L-6 on the demo
    # documents: relevant questions score -0.6 and up ("can I accept a gift
    # from a vendor" is the lowest), off-topic ones about -11.
    rerank_min_score: float = -5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
