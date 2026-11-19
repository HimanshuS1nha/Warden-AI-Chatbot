from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from warden.core.config import Settings
from warden.graph.prompts import TOOL_LIMIT_NOTE
from warden.graph.state import AgentState

# Tool calls the model may make while answering one user message. Enough for
# a question that needs a search, a retry and a few lookups; a model stuck
# re-searching for something that isn't there stops here.
MAX_TOOL_CALLS = 6

# Said when the model asks for more tools even though it was told it can't.
GAVE_UP = (
    "I couldn't finish looking this up within the limit for one message. "
    "Could you ask a narrower question?"
)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """The chat model from settings. Raises when the API key is missing."""
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
        api_key=settings.gemini_api_key.get_secret_value(),
    )


def tool_calls_this_turn(messages: Sequence[AnyMessage]) -> int:
    """Tool calls the model has asked for since the user's latest message.
    Earlier turns in the thread don't count."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            count += len(message.tool_calls)
    return count


class AgentNode:
    """The node that calls the model. The model sees the tools' names,
    descriptions and arguments, and either answers or asks for tool calls.

    Once this turn has used MAX_TOOL_CALLS, the model is called with
    `tool_choice="none"` and a note to answer now. The tools stay bound so
    the model still understands the tool calls in its history.
    """

    def __init__(self, model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.model = model.bind_tools(tools) if tools else model
        self.answer_only = model.bind_tools(tools, tool_choice="none") if tools else model

    async def __call__(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        if tool_calls_this_turn(messages) < MAX_TOOL_CALLS:
            reply = await self.model.ainvoke([SystemMessage(self.system_prompt), *messages])
            return {"messages": [reply]}

        prompt = f"{self.system_prompt}\n\n{TOOL_LIMIT_NOTE}"
        reply = await self.answer_only.ainvoke([SystemMessage(prompt), *messages])
        if isinstance(reply, AIMessage) and reply.tool_calls:
            # Ending here keeps the graph from looping until its recursion
            # limit. The ignored calls never reach the tools node.
            reply = AIMessage(content=GAVE_UP)
        return {"messages": [reply]}
