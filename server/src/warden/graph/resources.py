"""What the agent shares across requests, opened once at startup.

The graph itself is built per request (its tools are bound to the signed-in
user) from these shared pieces.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from warden.core.config import Settings
from warden.graph.agent import build_chat_model
from warden.graph.checkpointer import open_checkpointer
from warden.rag.embeddings import Embedder, build_embedding_client
from warden.rag.retrievers import InMemoryRetriever, MinScoreRetriever, Retriever
from warden.rag.store import ChunkStoreError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResources:
    checkpointer: BaseCheckpointSaver[str]
    # None without GEMINI_API_KEY: chat is unavailable, the rest of the app runs.
    chat_model: BaseChatModel | None
    # None without the key or without a chunk store (run ingest): the agent
    # works without policy search.
    retriever: Retriever | None


@asynccontextmanager
async def open_agent_resources(settings: Settings) -> AsyncIterator[AgentResources]:
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            open_checkpointer(settings.checkpointer_database_url)
        )
        if not settings.gemini_api_key.get_secret_value():
            logger.warning("GEMINI_API_KEY is not set; the chat agent is disabled")
            yield AgentResources(checkpointer=checkpointer, chat_model=None, retriever=None)
            return

        embedding_client = build_embedding_client(settings)
        stack.push_async_callback(embedding_client.aclose)
        embedder = Embedder(
            embedding_client, settings.cache_dir, batch_size=settings.embedding_batch_size
        )
        retriever: Retriever | None = None
        try:
            store = InMemoryRetriever.load(settings.cache_dir, embedder)
            logger.info("Loaded %d policy chunks for search", len(store))
            retriever = MinScoreRetriever(store, settings.policy_min_score)
        except ChunkStoreError as exc:
            logger.warning("Policy search is disabled: %s", exc)

        yield AgentResources(
            checkpointer=checkpointer,
            chat_model=build_chat_model(settings),
            retriever=retriever,
        )
