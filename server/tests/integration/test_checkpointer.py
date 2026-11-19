"""The checkpointer against the real test database.

Checkpoints are written through their own psycopg pool with autocommit, so
they aren't rolled back with the test's transaction. Each test uses fresh
thread IDs and deletes them at the end.
"""

import uuid

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tests.agent_fakes import PROMPT, script
from warden.core.config import get_settings
from warden.graph.checkpointer import open_checkpointer
from warden.graph.graph import build_graph


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.usefixtures("database_url")
async def test_a_thread_continues_where_it_stopped() -> None:
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    url = get_settings().checkpointer_database_url

    async with open_checkpointer(url) as checkpointer:
        try:
            graph = build_graph(script("Hello", "Hi again"), [], PROMPT, checkpointer)
            await graph.ainvoke({"messages": [HumanMessage(content="Hi")]}, thread(first))
            await graph.ainvoke({"messages": [HumanMessage(content="Again")]}, thread(second))

            # A new pool and graph, like after a restart, still sees the thread.
            async with open_checkpointer(url) as reopened:
                result = await build_graph(script("Yes"), [], PROMPT, reopened).ainvoke(
                    {"messages": [HumanMessage(content="Still there?")]}, thread(first)
                )

            assert [m.content for m in result["messages"]] == ["Hi", "Hello", "Still there?", "Yes"]
            other = await graph.aget_state(thread(second))
            assert [m.content for m in other.values["messages"]] == ["Again", "Hi again"]
        finally:
            await checkpointer.adelete_thread(first)
            await checkpointer.adelete_thread(second)
