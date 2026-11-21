"""What the agent and the document admin endpoints share across requests,
opened once at startup.

The graph itself is built per request (its tools are bound to the signed-in
user) from these shared pieces.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient

from warden.core.config import Settings
from warden.core.database import SessionLocal
from warden.graph.agent import build_chat_model
from warden.graph.checkpointer import open_checkpointer
from warden.rag.chunking import HFTokenCounter
from warden.rag.embeddings import Embedder, build_embedding_client
from warden.rag.index import build_index
from warden.rag.indexer import DocumentIndexer
from warden.rag.retrievers import Retriever
from warden.rag.retrievers.factory import RetrieverUnavailable, open_retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResources:
    checkpointer: BaseCheckpointSaver[str]
    # None without GEMINI_API_KEY: chat is unavailable, the rest of the app runs.
    chat_model: BaseChatModel | None
    # None without the key, or when the configured search can't start (no
    # index yet, Qdrant down): the agent works without policy search.
    retriever: Retriever | None
    # Indexes uploaded documents. None without the key. Doesn't need Qdrant
    # up at startup: an upload made while it's down ends as `failed`.
    indexer: DocumentIndexer | None


@asynccontextmanager
async def open_agent_resources(settings: Settings) -> AsyncIterator[AgentResources]:
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            open_checkpointer(settings.checkpointer_database_url)
        )
        if not settings.gemini_api_key.get_secret_value():
            logger.warning("GEMINI_API_KEY is not set; the chat agent is disabled")
            yield AgentResources(
                checkpointer=checkpointer, chat_model=None, retriever=None, indexer=None
            )
            return

        embedding_client = build_embedding_client(settings)
        stack.push_async_callback(embedding_client.aclose)
        embedder = Embedder(
            embedding_client, settings.cache_dir, batch_size=settings.embedding_batch_size
        )
        retriever: Retriever | None = None
        try:
            retriever = await stack.enter_async_context(open_retriever(settings, embedder))
        except RetrieverUnavailable as exc:
            logger.warning("Policy search is disabled: %s", exc)

        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        stack.push_async_callback(qdrant.close)
        indexer = DocumentIndexer(
            SessionLocal,
            cache_dir=settings.cache_dir,
            embedder=embedder,
            counter=HFTokenCounter(),
            index=build_index(settings, qdrant),
        )

        yield AgentResources(
            checkpointer=checkpointer,
            chat_model=build_chat_model(settings),
            retriever=retriever,
            indexer=indexer,
        )
