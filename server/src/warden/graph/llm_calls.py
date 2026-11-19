"""Records every chat model call in `llm_calls`: model, tokens, latency and
the tools it asked for. `GET /admin/llm-stats` and the latency and cost
evaluation read these rows.

Pass a recorder in the run's config and every model call in that run is
recorded, however many times the agent loops:

    graph.astream(..., config={"callbacks": [LLMCallRecorder(SessionLocal, thread.id)]})
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import LLMCallModel

logger = logging.getLogger(__name__)

UNKNOWN_MODEL = "unknown"


class LLMCallRecorder(AsyncCallbackHandler):
    """One row per model call, written as soon as the call ends.

    Each row is written in its own session and transaction, never the
    request's: tools share that session, and a stats row must not depend on
    whether the turn's own writes commit. A failed write is logged and
    otherwise ignored, so stats can't break a chat.

    Failed model calls aren't recorded: the table has no column for errors,
    and a row with zero tokens would skew the averages. They show up in the
    logs instead.
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        thread_id: uuid.UUID | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_factory = session_factory
        self.thread_id = thread_id
        self.clock = clock
        # Model calls in progress: run ID -> (start time, model name).
        self._started: dict[uuid.UUID, tuple[float, str]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = (metadata or {}).get("ls_model_name") or UNKNOWN_MODEL
        self._started[run_id] = (self.clock(), model)

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, None)
        if started is None:  # Not a chat model call.
            return
        start, model = started
        latency_ms = round((self.clock() - start) * 1000)

        message = _reply(response)
        usage = message.usage_metadata if message else None
        row = LLMCallModel(
            thread_id=self.thread_id,
            model=_model_name(message, model),
            prompt_tokens=usage["input_tokens"] if usage else 0,
            completion_tokens=usage["output_tokens"] if usage else 0,
            latency_ms=latency_ms,
            tool_calls=[c["name"] for c in message.tool_calls] if message else [],
        )
        try:
            async with self.session_factory() as db:
                db.add(row)
                await db.commit()
        except Exception:
            logger.exception("Couldn't record an LLM call for thread %s", self.thread_id)

    async def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._started.pop(run_id, None)


def _reply(response: LLMResult) -> AIMessage | None:
    """The model's reply. A chat call has one generation."""
    if response.generations and response.generations[0]:
        generation = response.generations[0][0]
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            return generation.message
    return None


def _model_name(message: AIMessage | None, fallback: str) -> str:
    """The model that answered, as the provider reports it, else the one
    that was asked for."""
    reported = message.response_metadata.get("model_name") if message else None
    return str(reported or fallback)[:100]
