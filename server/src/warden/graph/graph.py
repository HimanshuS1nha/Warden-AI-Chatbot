from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from warden.graph.state import AgentState

DEMO_REPLY = "Hello from Warden. The agent isn't connected to a model yet."


def agent(state: AgentState) -> AgentState:
    """Placeholder that replies with a fixed message. Replaced by the chat
    model call once the model is set up."""
    return {"messages": [AIMessage(content=DEMO_REPLY)]}


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()
