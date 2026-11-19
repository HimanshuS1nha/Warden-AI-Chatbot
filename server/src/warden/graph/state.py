from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """What the graph carries between nodes.

    `add_messages` appends the messages a node returns instead of replacing
    the list, and replaces a message that has the same ID.
    """

    messages: Annotated[list[AnyMessage], add_messages]
