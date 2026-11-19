from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from warden.core.config import Settings
from warden.graph.state import AgentState

# A stand-in until the full prompt (date, user, citation and safety rules).
SYSTEM_PROMPT = (
    "You are Warden, an HR assistant for the company's employees. "
    "Answer questions about the user's own data only from tool results, "
    "never from guesses. If no tool can answer, say you don't know."
)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """The chat model from settings. Raises when the API key is missing."""
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
        api_key=settings.gemini_api_key.get_secret_value(),
    )


class AgentNode:
    """The node that calls the model. The model sees the tools' names,
    descriptions and arguments, and either answers or asks for tool calls."""

    def __init__(self, model: BaseChatModel, tools: Sequence[BaseTool]) -> None:
        self.model = model.bind_tools(tools) if tools else model

    async def __call__(self, state: AgentState) -> AgentState:
        reply = await self.model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}
