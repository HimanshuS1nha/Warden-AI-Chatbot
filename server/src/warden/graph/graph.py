from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from warden.graph.agent import AgentNode
from warden.graph.state import AgentState


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """The agent loop: the model answers or asks for tools, the tools run, and
    their results go back to the model, until it answers without a tool call.

        START -> agent -> (tool calls?) -> tools -> agent ... -> END

    Built per request, because the tools are bound to the signed-in user.
    Compiling is cheap; the model and checkpointer are shared. With a
    checkpointer, every run needs `{"configurable": {"thread_id": ...}}` and
    continues that thread's saved state.
    """
    builder = StateGraph(AgentState)
    # The same tools go to both nodes on purpose. The agent node shows them
    # to the model (names, descriptions, arguments) so it can ask for one;
    # the model can't run code. The tools node runs what it asked for.
    builder.add_node("agent", AgentNode(model, tools))
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    # Goes to "tools" when the last message has tool calls, otherwise to END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
