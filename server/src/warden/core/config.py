from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SQLAlchemy URL with the asyncpg driver, for the app's models.
    database_url: str
    # Plain libpq URL (postgresql://...) for the agent's checkpointer, which
    # uses psycopg. Usually the same database as database_url.
    checkpointer_database_url: str
    frontend_url: str = "http://localhost:5173"

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
    # Policy search finds nothing when its best match scores below this
    # (cosine similarity with the embedding model above). Measured on the
    # demo documents: relevant questions score 0.70+, off-topic ones
    # ("how to bake bread") 0.51-0.59. Change it with the embedding model.
    policy_min_score: float = Field(default=0.6, ge=0, le=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
