from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from warden.graph.state import AgentState

DEMO_REPLY = "Hello from Warden. The agent isn't connected to a model yet."


def agent(state: AgentState) -> AgentState:
    """Placeholder that replies with a fixed message. Replaced by the chat
    model call once the model is set up."""
    return {"messages": [AIMessage(content=DEMO_REPLY)]}


def build_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the graph. With a checkpointer, every run needs a thread ID in
    its config (`{"configurable": {"thread_id": ...}}`) and continues that
    thread's saved state."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=checkpointer)
