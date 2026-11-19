"""Runs one chat turn through the graph and streams it as server-sent events.

Events, in the order they can arrive:

- `token` {text}: a piece of the reply as the model writes it. Text written
  before a tool call can arrive too; `done` has the final reply.
- `tool_start` {id, name, args}: the agent asked for a tool.
- `tool_end` {id, name, ok}: the tool finished (ok is false when it failed
  or was refused).
- `citations` {citations}: the policy excerpts the reply cites, after the
  invented ones are dropped. Only sent when there are some.
- `done` {message_id, content}: the reply as saved. Show this rather than
  the joined tokens: invented citation markers are removed from it.
- `error` {message}: the turn failed. Nothing more follows.
"""

import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from warden.core.dependencies import CurrentUser
from warden.graph.state import AgentState
from warden.models import ChatThreadModel, UnansweredReason
from warden.rag.citations import resolve_citations, sources_from_artifacts
from warden.services.chat import ChatService

logger = logging.getLogger(__name__)

POLICY_TOOL = "search_policies"
ERROR_MESSAGE = "Something went wrong while answering. Please try again."


def sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def unanswered_reason(
    searches: Sequence[ToolMessage], citations: Sequence[object]
) -> UnansweredReason | None:
    """Why the turn's policy question went unanswered, if it did.

    - Every search came back empty: the documents don't cover it.
    - Searches found excerpts but the reply cites none: the closest text
      didn't answer the question.
    A turn without a policy search isn't a policy question.
    """
    if not searches:
        return None
    if not any(m.artifact for m in searches):
        return UnansweredReason.NO_RELEVANT_DOCS
    if not citations:
        return UnansweredReason.LOW_CONFIDENCE
    return None


@dataclass
class ChatTurn:
    """One user message: run the graph, stream what happens, then save the
    reply. The user's message is saved by the caller before this starts."""

    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState]
    chat: ChatService
    user: CurrentUser
    thread: ChatThreadModel
    message: str
    callbacks: list[BaseCallbackHandler]

    async def events(self) -> AsyncIterator[str]:
        reply: AIMessage | None = None
        tool_results: list[ToolMessage] = []
        config: RunnableConfig = {
            "configurable": {"thread_id": str(self.thread.id)},
            "callbacks": self.callbacks,
        }
        try:
            async for item in self.graph.astream(
                {"messages": [HumanMessage(content=self.message)]},
                config,
                stream_mode=["messages", "updates"],
            ):
                # With a list of modes, each item is (mode, data).
                mode, data = cast(tuple[str, Any], item)
                if mode == "messages":
                    chunk, metadata = data
                    if (
                        metadata.get("langgraph_node") == "agent"
                        and isinstance(chunk, AIMessageChunk)
                        and chunk.text
                    ):
                        yield sse("token", {"text": chunk.text})
                    continue

                for node, update in data.items():
                    for message in (update or {}).get("messages", []):
                        if node == "agent" and isinstance(message, AIMessage):
                            if not message.tool_calls:
                                reply = message
                            for call in message.tool_calls:
                                yield sse(
                                    "tool_start",
                                    {"id": call["id"], "name": call["name"], "args": call["args"]},
                                )
                        elif node == "tools" and isinstance(message, ToolMessage):
                            tool_results.append(message)
                            ok = message.status != "error"
                            yield sse(
                                "tool_end",
                                {"id": message.tool_call_id, "name": message.name, "ok": ok},
                            )
            if reply is None:
                raise RuntimeError("The graph ended without a reply")

            searches = [m for m in tool_results if m.name == POLICY_TOOL and m.status != "error"]
            sources = sources_from_artifacts(m.artifact or [] for m in searches)
            content, citations = resolve_citations(reply.text, sources)
            if citations:
                yield sse("citations", {"citations": citations})

            saved = await self.chat.add_assistant_message(self.thread, content, citations)
            reason = unanswered_reason(searches, citations)
            if reason is not None:
                await self.chat.record_unanswered(self.user, self.message, reason)
        except Exception:
            logger.exception("Chat turn failed in thread %s", self.thread.id)
            yield sse("error", {"message": ERROR_MESSAGE})
            return

        yield sse("done", {"message_id": str(saved.id), "content": content})
