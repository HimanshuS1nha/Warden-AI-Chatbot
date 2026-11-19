"""A chat model for tests that replies from a script, so tests never call the
real model."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class ScriptedChatModel(GenericFakeChatModel):
    """Returns the scripted replies in order, one per model call, and records
    which tools were bound to it."""

    bound_tools: list[str] = []

    def bind_tools(
        self, tools: Sequence[Any], **_kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = [t.name for t in tools if isinstance(t, BaseTool)]
        return self


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
