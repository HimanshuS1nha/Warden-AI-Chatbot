"""A chat model for tests that replies from a script, so tests never call the
real model."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

# A system prompt for tests that don't check the real one.
PROMPT = "You are a test assistant."


class ModelCall:
    """What one model call received."""

    def __init__(self, messages: list[BaseMessage], tool_choice: str | None) -> None:
        self.messages = messages
        self.tool_choice = tool_choice

    @property
    def system_prompt(self) -> str:
        return str(self.messages[0].content)


class ScriptedChatModel(GenericFakeChatModel):
    """Returns the scripted replies in order, one per model call, and records
    which tools were bound to it and what each call received."""

    bound_tools: list[str] = []
    calls: list[ModelCall] = []

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **_kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = [t.name for t in tools if isinstance(t, BaseTool)]
        return self.bind(tool_choice=tool_choice)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(ModelCall(messages, kwargs.pop("tool_choice", None)))
        return super()._generate(messages, stop, run_manager, **kwargs)


def script(*replies: str | ToolCall) -> ScriptedChatModel:
    """A model that gives these replies in order: text for an answer, a
    ToolCall for a request to run that tool."""
    messages = [
        AIMessage(content=reply)
        if isinstance(reply, str)
        else AIMessage(content="", tool_calls=[reply])
        for reply in replies
    ]
    return ScriptedChatModel(messages=iter(messages))


def call(name: str, call_id: str = "call_1", **args: Any) -> ToolCall:
    """A tool call as the model would make it. Also works to run a tool
    directly (`await tool.ainvoke(call(...))`), which returns a ToolMessage."""
    return tool_call(name=name, args=args, id=call_id)
