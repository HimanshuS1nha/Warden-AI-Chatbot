import logging
from collections.abc import Callable, Iterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from tests.agent_fakes import PROMPT, ScriptedChatModel, call
from tests.factories import MakeEmployee
from warden.graph.graph import build_graph
from warden.graph.llm_calls import LLMCallRecorder
from warden.models import ChatThreadModel, LLMCallModel


@tool
async def get_my_profile() -> str:
    """The user's profile."""
    return "{}"


def usage(prompt: int, completion: int) -> UsageMetadata:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def ticking_clock(step: float) -> Callable[[], float]:
    """A clock that moves `step` seconds every time it's read, so each model
    call takes exactly `step`."""
    ticks: Iterator[float] = (n * step for n in range(1000))
    return lambda: next(ticks)


@pytest.fixture
def session_factory(connection: AsyncConnection) -> Callable[[], AsyncSession]:
    """Sessions in the test's transaction, like SessionLocal in the app."""
    return lambda: AsyncSession(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )


async def rows(db: AsyncSession) -> list[LLMCallModel]:
    result = await db.scalars(
        select(LLMCallModel).order_by(LLMCallModel.created_at, LLMCallModel.id)
    )
    return list(result.all())


async def test_every_model_call_in_a_run_is_recorded(
    db: AsyncSession,
    make_employee: MakeEmployee,
    session_factory: Callable[[], AsyncSession],
) -> None:
    me = await make_employee()
    thread = ChatThreadModel(employee_id=me.id)
    db.add(thread)
    await db.flush()
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[call("get_my_profile")],
                    usage_metadata=usage(1200, 15),
                    response_metadata={"model_name": "gemini-3.8-flash"},
                ),
                AIMessage(
                    content="You're Asha.",
                    usage_metadata=usage(1300, 40),
                    response_metadata={"model_name": "gemini-3.8-flash"},
                ),
            ]
        )
    )
    recorder = LLMCallRecorder(session_factory, thread.id, clock=ticking_clock(0.25))

    await build_graph(model, [get_my_profile], PROMPT).ainvoke(
        {"messages": [HumanMessage(content="Who am I?")]}, config={"callbacks": [recorder]}
    )

    assert [
        (r.thread_id, r.model, r.prompt_tokens, r.completion_tokens, r.latency_ms, r.tool_calls)
        for r in await rows(db)
    ] == [
        (thread.id, "gemini-3.8-flash", 1200, 15, 250, ["get_my_profile"]),
        (thread.id, "gemini-3.8-flash", 1300, 40, 250, []),
    ]


async def test_a_call_without_usage_or_thread_still_gets_a_row(
    db: AsyncSession, session_factory: Callable[[], AsyncSession]
) -> None:
    model = ScriptedChatModel(messages=iter([AIMessage(content="Hi.")]))

    await build_graph(model, [], PROMPT).ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"callbacks": [LLMCallRecorder(session_factory)]},
    )

    [row] = await rows(db)
    assert (row.thread_id, row.prompt_tokens, row.completion_tokens, row.tool_calls) == (
        None,
        0,
        0,
        [],
    )
    assert row.model  # The fake model's name, or UNKNOWN_MODEL.
    assert row.latency_ms >= 0


async def test_a_failed_write_is_logged_and_the_chat_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_session() -> AsyncSession:
        raise ConnectionError("database is down")

    model = ScriptedChatModel(messages=iter([AIMessage(content="Hi.")]))

    with caplog.at_level(logging.ERROR, logger="warden.graph.llm_calls"):
        result = await build_graph(model, [], PROMPT).ainvoke(
            {"messages": [HumanMessage(content="Hi")]},
            config={"callbacks": [LLMCallRecorder(broken_session)]},
        )

    assert result["messages"][-1].content == "Hi."
    assert "Couldn't record an LLM call" in caplog.text


async def test_a_failed_model_call_is_not_recorded(
    db: AsyncSession, session_factory: Callable[[], AsyncSession]
) -> None:
    model = ScriptedChatModel(messages=iter([]))  # Out of replies: the call fails.
    recorder = LLMCallRecorder(session_factory)

    with pytest.raises(RuntimeError):
        await build_graph(model, [], PROMPT).ainvoke(
            {"messages": [HumanMessage(content="Hi")]}, config={"callbacks": [recorder]}
        )

    assert await rows(db) == []
    assert recorder._started == {}
