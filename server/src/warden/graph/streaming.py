"""Runs one chat turn through the graph and streams it as server-sent events.

Events, in the order they can arrive:

- `action_result` {action_id, status, ok}: only on a confirm or cancel
  stream, first. The action's new status (`confirmed`, `failed` or
  `cancelled`) for the card; the agent's reply follows.
- `token` {text}: a piece of the reply as the model writes it. Text written
  before a tool call can arrive too; `done` has the final reply.
- `tool_start` {id, name, args}: the agent asked for a tool.
- `tool_end` {id, name, ok}: the tool finished (ok is false when it failed
  or was refused).
- `citations` {citations}: the policy excerpts the reply cites, after the
  invented ones are dropped. Only sent when there are some.
- `confirmation_required` {action_id, action_type, summary}: a write tool
  proposed a change. Show a card with Confirm and Cancel buttons; nothing
  changes until the user clicks Confirm. Sent once the reply is saved, at
  most once per turn.
- `done` {message_id, content}: the reply as saved. Show this rather than
  the joined tokens: invented citation markers are removed from it.
- `error` {message}: the turn failed. Nothing more follows.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import anyio
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.graph.graph import proposed_action
from warden.graph.state import AgentState
from warden.models import ChatThreadModel, UnansweredReason
from warden.rag.citations import resolve_citations, sources_from_artifacts
from warden.repositories import ChatRepository
from warden.services.chat import ChatService

logger = logging.getLogger(__name__)

POLICY_TOOL = "search_policies"
ERROR_MESSAGE = "Something went wrong while answering. Please try again."

# What the agent remembers in place of a reply it never finished.
UNFINISHED_TOOL = "Not run: the reply was stopped before this tool ran."
UNFINISHED_REPLY = "(This reply was stopped before it was finished.)"


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


def closing_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """What closes a turn that ended without a reply, so the agent doesn't
    try to answer it again with the user's next message.

    A turn ends without a reply when the user stops it or it fails. The
    agent's memory then ends with the question, or with tool calls that
    never got results (which the model would refuse). This answers each
    open tool call and adds a reply saying it was stopped. Empty when the
    last turn has its reply.
    """
    if not messages:
        return []
    last = messages[-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return []
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    closing: list[AnyMessage] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            closing.extend(
                ToolMessage(
                    content=UNFINISHED_TOOL,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in message.tool_calls
                if call["id"] not in answered
            )
    closing.append(AIMessage(content=UNFINISHED_REPLY))
    return closing


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
    # A session of its own for cleanup: the request's may be mid-operation
    # when the turn is cut off.
    session_factory: Callable[[], AsyncSession]
    # False when `message` comes from the app (a confirmation result), not
    # the user: it isn't a question that can go unanswered.
    from_user: bool = True

    async def events(self) -> AsyncIterator[str]:
        reply: AIMessage | None = None
        tool_results: list[ToolMessage] = []
        # Read once: after a failed write the session expires the thread, and
        # reading its attributes again would raise.
        thread_id = self.thread.id
        config: RunnableConfig = {
            "configurable": {"thread_id": str(thread_id)},
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

            # ToolCallGuard lets one write through per turn.
            proposal = next(filter(None, map(proposed_action, tool_results)), None)
            action_id = uuid.UUID(proposal["pending_action_id"]) if proposal else None
            saved = await self.chat.add_assistant_message(
                self.thread, content, citations, action_id
            )
            if proposal:
                yield sse(
                    "confirmation_required",
                    {
                        "action_id": proposal["pending_action_id"],
                        "action_type": proposal["action_type"],
                        "summary": proposal["summary"],
                    },
                )
            reason = unanswered_reason(searches, citations) if self.from_user else None
            if reason is not None:
                await self.chat.record_unanswered(self.user, self.message, reason)
        except (asyncio.CancelledError, GeneratorExit):
            # The client went away: the user pressed Stop or left the page.
            # Shielded, since every await in a cancelled scope is cancelled too.
            with anyio.CancelScope(shield=True):
                await self._close_unfinished_turn(thread_id, config)
            raise
        except Exception:
            logger.exception("Chat turn failed in thread %s", thread_id)
            await self._close_unfinished_turn(thread_id, config)
            yield sse("error", {"message": ERROR_MESSAGE})
            return

        yield sse("done", {"message_id": str(saved.id), "content": content})

    async def _close_unfinished_turn(self, thread_id: uuid.UUID, config: RunnableConfig) -> None:
        """Adds `closing_messages` to the agent's memory of the thread.

        If the thread was deleted during the turn, the run may have saved
        checkpoints after the delete cleared them: delete those instead.
        """
        try:
            async with self.session_factory() as db:
                deleted = not await ChatRepository(db).thread_exists(thread_id)
            if deleted:
                if isinstance(self.graph.checkpointer, BaseCheckpointSaver):
                    await self.graph.checkpointer.adelete_thread(str(thread_id))
                return
            state = await self.graph.aget_state(config)
            closing = closing_messages(state.values.get("messages", []))
            if closing:
                await self.graph.aupdate_state(config, {"messages": closing}, as_node="agent")
        except Exception:
            logger.exception("Couldn't close the unfinished turn in thread %s", thread_id)
