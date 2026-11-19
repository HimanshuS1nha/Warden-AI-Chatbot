import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.types import Command

from warden.graph.agent import MAX_TOOL_CALLS, AgentNode, tool_calls_this_turn
from warden.graph.state import AgentState
from warden.graph.tools.common import PARALLEL_SAFE
from warden.services.errors import ServiceError


def tool_error(exc: ToolInvocationError | ServiceError) -> str:
    """What the model reads when a tool fails, so it can fix its arguments
    or explain the refusal to the user instead of the run crashing.

    Bad arguments (an invalid year, a malformed ID) and service refusals
    (not found, forbidden, a broken leave rule) come back as messages. Any
    other exception is a bug and raises. ToolNode filters by this function's
    annotation, except on the path through a tool-call wrapper
    (`ToolCallGuard`), where it passes every exception here; hence the
    explicit re-raise.
    """
    if isinstance(exc, ToolInvocationError):
        return exc.message
    if isinstance(exc, ServiceError):
        return json.dumps({"error": exc.code, "message": exc.message} | exc.details, default=str)
    raise exc


OVER_LIMIT = (
    f"Not run: the limit of {MAX_TOOL_CALLS} tool calls for this message was reached. "
    "Answer with what you have."
)


ToolExecutor = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


class ToolCallGuard:
    """Wraps every tool call the tools node runs. One per graph, so one per
    request.

    1. Limit: a call past this turn's MAX_TOOL_CALLS isn't run. The agent node
       stops asking for tools at the limit, but one reply can ask for several
       at once: with 5 used, a reply asking for 3 runs the first and refuses
       the other 2. A refused call still gets a ToolMessage, because the model
       expects an answer to every call it made.
    2. One at a time: ToolNode runs a reply's calls concurrently, but the
       database tools share the request's session, which allows one operation
       at a time. Calls wait for each other unless the tool is marked
       PARALLEL_SAFE in its metadata (it doesn't touch the session), so a new
       tool is safe by default.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()

    async def __call__(
        self, request: ToolCallRequest, execute: ToolExecutor
    ) -> ToolMessage | Command[Any]:
        messages = request.state["messages"]
        calls = cast(AIMessage, messages[-1])  # The agent's reply that asked for these calls.
        position = [c["id"] for c in calls.tool_calls].index(request.tool_call["id"])
        if tool_calls_this_turn(messages[:-1]) + position >= MAX_TOOL_CALLS:
            return ToolMessage(
                content=OVER_LIMIT,
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        if request.tool is not None and (request.tool.metadata or {}).get(PARALLEL_SAFE):
            return await execute(request)
        async with self.lock:
            return await execute(request)


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """The agent loop: the model answers or asks for tools, the tools run, and
    their results go back to the model, until it answers without a tool call.

        START -> agent -> (tool calls?) -> tools -> agent ... -> END

    Built per request, because the tools and the system prompt (see
    `build_system_prompt`) are for the signed-in user.
    Compiling is cheap; the model and checkpointer are shared. With a
    checkpointer, every run needs `{"configurable": {"thread_id": ...}}` and
    continues that thread's saved state.
    """
    builder = StateGraph(AgentState)
    # The same tools go to both nodes on purpose. The agent node shows them
    # to the model (names, descriptions, arguments) so it can ask for one;
    # the model can't run code. The tools node runs what it asked for.
    builder.add_node("agent", AgentNode(model, tools, system_prompt))
    builder.add_node(
        "tools", ToolNode(tools, handle_tool_errors=tool_error, awrap_tool_call=ToolCallGuard())
    )
    builder.add_edge(START, "agent")
    # Goes to "tools" when the last message has tool calls, otherwise to END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
