import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from tests.agent_fakes import call
from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from tests.unit.test_policy_tool import CHUNKS
from warden.graph.streaming import (
    UNFINISHED_REPLY,
    UNFINISHED_TOOL,
    closing_messages,
    sse,
    unanswered_reason,
)
from warden.models import AccessLevel, UnansweredReason
from warden.rag.embeddings import Embedder, EmbedTask
from warden.rag.retrievers import InMemoryRetriever, MinScoreRetriever, SearchFilters
from warden.services.chat import TITLE_LENGTH, thread_title


def search_result(found: bool) -> ToolMessage:
    return ToolMessage(
        content="...",
        tool_call_id=str(uuid.uuid4()),
        name="search_policies",
        artifact=[{"n": 1}] if found else [],
    )


@pytest.mark.parametrize(
    ("searches", "citations", "expected"),
    [
        ([], [], None),  # Not a policy question.
        ([search_result(False), search_result(False)], [], UnansweredReason.NO_RELEVANT_DOCS),
        ([search_result(False), search_result(True)], [], UnansweredReason.LOW_CONFIDENCE),
        ([search_result(True)], [{"n": 1}], None),
    ],
)
def test_unanswered_reason(
    searches: list[ToolMessage], citations: list[object], expected: UnansweredReason | None
) -> None:
    assert unanswered_reason(searches, citations) == expected


def test_sse_format() -> None:
    assert sse("token", {"text": "Hi"}) == 'event: token\ndata: {"text": "Hi"}\n\n'


def test_thread_title_is_the_first_line_cut_at_a_word() -> None:
    assert thread_title("  How many leaves?\nAnd more  ") == "How many leaves?"
    long = "How many casual leave days can I carry forward " * 5
    title = thread_title(long)
    assert len(title) <= TITLE_LENGTH and title.endswith("…")
    assert not title.removesuffix("…").endswith(" ")


async def test_min_score_returns_nothing_when_the_best_match_is_weak(tmp_path: Path) -> None:
    embedder = Embedder(FakeEmbeddingClient(), tmp_path, sleep=no_sleep)
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    store = InMemoryRetriever(CHUNKS, vectors, embedder)
    everything = SearchFilters(access_levels=frozenset(AccessLevel))

    [best, *rest] = await store.search("casual leave carry forward", 3, everything)
    at_best = await MinScoreRetriever(store, best.score).search(
        "casual leave carry forward", 3, everything
    )
    above_best = await MinScoreRetriever(store, best.score + 0.01).search(
        "casual leave carry forward", 3, everything
    )

    # The weaker matches stay when the best one passes.
    assert [c.chunk_id for c in at_best] == [best.chunk_id] + [c.chunk_id for c in rest]
    assert above_best == []


def summary(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    return [
        (m.type, m.tool_call_id if isinstance(m, ToolMessage) else str(m.content)) for m in messages
    ]


def test_a_finished_turn_needs_no_closing() -> None:
    assert closing_messages([]) == []
    assert closing_messages([HumanMessage("Hi"), AIMessage("Hello.")]) == []


def test_an_unanswered_question_gets_a_stopped_reply() -> None:
    messages: list[AnyMessage] = [
        HumanMessage("Hi"),
        AIMessage("Hello."),
        HumanMessage("Summarise"),
    ]
    assert summary(closing_messages(messages)) == [("ai", UNFINISHED_REPLY)]


def test_open_tool_calls_get_results_before_the_stopped_reply() -> None:
    # An earlier turn's calls are answered; this turn's second call isn't.
    messages: list[AnyMessage] = [
        HumanMessage("Hi"),
        AIMessage("", tool_calls=[call("get_my_manager", "old")]),
        ToolMessage("Vikram", tool_call_id="old"),
        AIMessage("Vikram."),
        HumanMessage("Summarise"),
        AIMessage("", tool_calls=[call("search_policies", "a"), call("search_policies", "b")]),
        ToolMessage("...", tool_call_id="a"),
    ]
    closing = closing_messages(messages)
    assert summary(closing) == [("tool", "b"), ("ai", UNFINISHED_REPLY)]
    assert isinstance(closing[0], ToolMessage)
    assert (closing[0].content, closing[0].status) == (UNFINISHED_TOOL, "error")
