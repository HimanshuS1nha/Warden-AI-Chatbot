"""The chat routes end to end: log in for real, stream a turn through the real
graph, tools, services and database, with a scripted model and an in-memory
checkpointer instead of Gemini and Postgres checkpoints."""

import json
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from tests.agent_fakes import ScriptedChatModel, call
from tests.factories import MakeEmployee
from tests.integration.test_auth import login
from tests.integration.test_leave_service import add_balance
from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from tests.unit.test_policy_tool import CHUNKS
from warden.graph.graph import ONE_PROPOSAL
from warden.graph.resources import AgentResources
from warden.graph.streaming import ERROR_MESSAGE
from warden.main import app
from warden.models import (
    ChatMessageModel,
    EmployeeModel,
    LeaveRequestModel,
    LLMCallModel,
    MessageRole,
    PendingActionModel,
    PendingActionStatus,
    UnansweredQueryModel,
    UnansweredReason,
)
from warden.rag.embeddings import Embedder, EmbedTask
from warden.rag.retrievers import InMemoryRetriever, MinScoreRetriever, Retriever
from warden.routers.chat import get_agent, get_session_factory
from warden.services.chat import MAX_MESSAGE_LENGTH
from warden.services.leave import company_today

Event = tuple[str, dict[str, Any]]


@pytest.fixture
async def retriever(tmp_path: Path) -> InMemoryRetriever:
    embedder = Embedder(FakeEmbeddingClient(), tmp_path, sleep=no_sleep)
    vectors = await embedder.embed([c.embedding_text for c in CHUNKS], EmbedTask.DOCUMENT)
    return InMemoryRetriever(CHUNKS, vectors, embedder)


@pytest.fixture
def use_agent(client: AsyncClient, connection: AsyncConnection) -> Iterator[Callable[..., None]]:
    """Point the chat routes at a scripted model. Depends on `client` so the
    overrides are set after, and removed before, the client's own."""
    checkpointer = InMemorySaver()

    def _use(model: ScriptedChatModel | None, retriever: Retriever | None = None) -> None:
        resources = AgentResources(checkpointer=checkpointer, chat_model=model, retriever=retriever)
        app.dependency_overrides[get_agent] = lambda: resources

    app.dependency_overrides[get_session_factory] = lambda: (
        lambda: AsyncSession(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
    )
    yield _use
    app.dependency_overrides.pop(get_agent, None)
    app.dependency_overrides.pop(get_session_factory, None)


def scripted(*replies: AIMessage | str) -> ScriptedChatModel:
    return ScriptedChatModel(
        messages=iter(AIMessage(content=r) if isinstance(r, str) else r for r in replies)
    )


def parse(body: str) -> list[Event]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def new_thread(client: AsyncClient) -> str:
    response = await client.post("/api/chat/threads")
    assert response.status_code == 201
    return str(response.json()["id"])


async def send(client: AsyncClient, thread_id: str, message: str) -> list[Event]:
    response = await client.post(
        f"/api/chat/threads/{thread_id}/messages", json={"message": message}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse(response.text)


def named(events: list[Event], name: str) -> list[dict[str, Any]]:
    return [data for event, data in events if event == name]


async def saved_messages(db: AsyncSession) -> list[tuple[MessageRole, str]]:
    rows = await db.scalars(
        select(ChatMessageModel)
        .order_by(ChatMessageModel.id)
        .execution_options(populate_existing=True)
    )
    return [(m.role, m.content) for m in rows]


# Threads


async def test_threads_are_listed_newest_first_and_paginated(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    me, other = await make_employee(), await make_employee()
    await login(client, other)
    await new_thread(client)
    await login(client, me)
    mine = [await new_thread(client) for _ in range(3)]

    first = (await client.get("/api/chat/threads", params={"limit": 2})).json()
    second = (
        await client.get("/api/chat/threads", params={"limit": 2, "before": first["next"]})
    ).json()

    assert [t["id"] for t in first["threads"]] == mine[:0:-1]
    assert [t["id"] for t in second["threads"]] == [mine[0]]
    assert second["next"] is None
    assert first["threads"][0]["title"] is None


async def test_someone_elses_thread_is_not_found(
    client: AsyncClient, make_employee: MakeEmployee, use_agent: Callable[..., None]
) -> None:
    owner, intruder = await make_employee(), await make_employee()
    use_agent(scripted("Should never run."))
    await login(client, owner)
    thread_id = await new_thread(client)
    await login(client, intruder)

    history = await client.get(f"/api/chat/threads/{thread_id}/messages")
    sent = await client.post(f"/api/chat/threads/{thread_id}/messages", json={"message": "Hi"})

    assert (history.status_code, sent.status_code) == (404, 404)


@pytest.mark.parametrize("message", ["", "   \n ", "x" * (MAX_MESSAGE_LENGTH + 1)])
async def test_empty_and_overlong_messages_are_refused(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
    message: str,
) -> None:
    use_agent(scripted("Should never run."))
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    response = await client.post(
        f"/api/chat/threads/{thread_id}/messages", json={"message": message}
    )

    assert response.status_code == 422
    assert await saved_messages(db) == []


async def test_chat_without_a_model_is_503(
    client: AsyncClient, make_employee: MakeEmployee, use_agent: Callable[..., None]
) -> None:
    use_agent(None)
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    response = await client.post(f"/api/chat/threads/{thread_id}/messages", json={"message": "Hi"})

    assert response.status_code == 503


# A turn


async def test_a_turn_streams_tools_tokens_and_valid_citations(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
    retriever: InMemoryRetriever,
) -> None:
    me = await make_employee()
    await add_balance(db, me, allotted=12, used=4, year=company_today().year)
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[
                call("get_my_leave_balance", "call_balance"),
                call("search_policies", "call_search", query="carry forward casual leave"),
            ],
            usage_metadata={"input_tokens": 900, "output_tokens": 20, "total_tokens": 920},
        ),
        # [7] was never returned by a search: it's invented and must go.
        "You have 8 casual days left. Up to 5 unused days carry forward [1]. "
        "Managers must approve it [7].",
    )
    use_agent(model, retriever)
    await login(client, me)
    thread_id = await new_thread(client)

    events = await send(client, thread_id, "How many casual leaves do I have and do they carry?")

    names = [event for event, _ in events]
    assert names[:2] == ["tool_start", "tool_start"]
    assert sorted(e["name"] for e in named(events, "tool_end")) == [
        "get_my_leave_balance",
        "search_policies",
    ]
    assert all(e["ok"] for e in named(events, "tool_end"))
    assert names[-2:] == ["citations", "done"]
    streamed = "".join(e["text"] for e in named(events, "token"))
    assert streamed.startswith("You have 8 casual days left.")

    [citations] = named(events, "citations")
    assert [(c["n"], c["doc_slug"]) for c in citations["citations"]] == [(1, "leave-policy")]
    [done] = named(events, "done")
    assert done["content"] == (
        "You have 8 casual days left. Up to 5 unused days carry forward [1]. "
        "Managers must approve it."
    )
    assert await saved_messages(db) == [
        (MessageRole.USER, "How many casual leaves do I have and do they carry?"),
        (MessageRole.ASSISTANT, done["content"]),
    ]

    history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()
    reply, question = history["messages"]
    assert reply["id"] == done["message_id"]
    assert reply["citations"] == citations["citations"]
    assert question["role"] == "user"
    thread = (await client.get("/api/chat/threads")).json()["threads"][0]
    assert thread["title"] == "How many casual leaves do I have and do they carry?"

    calls = (await db.scalars(select(LLMCallModel).order_by(LLMCallModel.created_at))).all()
    assert [(str(c.thread_id), c.tool_calls) for c in calls] == [
        (thread_id, ["get_my_leave_balance", "search_policies"]),
        (thread_id, []),
    ]
    assert calls[0].prompt_tokens == 900


async def test_the_agent_remembers_earlier_turns_in_the_thread(
    client: AsyncClient, make_employee: MakeEmployee, use_agent: Callable[..., None]
) -> None:
    model = scripted("Hi Asha.", "You said hello.")
    use_agent(model)
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    await send(client, thread_id, "Hello")
    await send(client, thread_id, "What did I just say?")

    second_call = [m.content for m in model.calls[-1].messages[1:]]
    assert second_call == ["Hello", "Hi Asha.", "What did I just say?"]


async def test_a_failed_turn_sends_an_error_and_keeps_the_question(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    use_agent(scripted())  # No replies: the model call fails.
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    events = await send(client, thread_id, "Hello?")

    assert events == [("error", {"message": ERROR_MESSAGE})]
    assert await saved_messages(db) == [(MessageRole.USER, "Hello?")]


# Write proposals


def next_monday() -> date:
    """A working day at least a week ahead, so leave on it is never backdated."""
    day = company_today() + timedelta(days=7)
    return day + timedelta(days=(7 - day.weekday()) % 7)


async def test_a_write_tool_streams_a_confirmation_card_and_changes_nothing(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    use_agent(
        scripted(
            AIMessage(
                content="",
                tool_calls=[
                    call(
                        "apply_for_leave",
                        leave_type="casual",
                        start_date=monday.isoformat(),
                        end_date=monday.isoformat(),
                    )
                ],
            ),
            "Check the card and click Confirm to apply.",
        )
    )
    await login(client, me)
    thread_id = await new_thread(client)

    events = await send(client, thread_id, "Apply for casual leave next Monday")

    assert [event for event, _ in events][-2:] == ["confirmation_required", "done"]
    [card] = named(events, "confirmation_required")
    action = await db.get(PendingActionModel, card["action_id"])
    assert action is not None
    assert card == {
        "action_id": str(action.id),
        "action_type": "apply_leave",
        "summary": action.summary,
    }
    assert action.summary.startswith(f"Casual leave, {monday:%a %-d %b}, 1 day.")
    assert action.status == PendingActionStatus.PENDING
    assert await db.scalar(select(LeaveRequestModel.id)) is None

    [done] = named(events, "done")
    history = (await client.get(f"/api/chat/threads/{thread_id}/messages")).json()
    reply = history["messages"][0]
    assert (reply["id"], reply["pending_action_id"]) == (done["message_id"], card["action_id"])


async def test_only_one_write_is_proposed_per_message(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    tuesday = monday + timedelta(days=1)
    await add_balance(db, me, allotted=8, year=tuesday.year)
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[
                call(
                    "apply_for_leave",
                    f"call_{n}",
                    leave_type="casual",
                    start_date=day.isoformat(),
                    end_date=day.isoformat(),
                )
                for n, day in enumerate((monday, tuesday), 1)
            ]
            + [call("get_my_leave_balance", "call_3")],
        ),
        "Confirm Monday first, then ask me again for Tuesday.",
    )
    use_agent(model)
    await login(client, me)
    thread_id = await new_thread(client)

    events = await send(client, thread_id, "Apply for Monday and Tuesday separately")

    ok = {e["id"]: e["ok"] for e in named(events, "tool_end")}
    assert ok == {"call_1": True, "call_2": False, "call_3": True}  # Reads still run.
    refused = next(
        m for m in model.calls[-1].messages if getattr(m, "tool_call_id", "") == "call_2"
    )
    assert refused.content == ONE_PROPOSAL
    assert len(named(events, "confirmation_required")) == 1
    actions = (await db.scalars(select(PendingActionModel))).all()
    assert len(actions) == 1
    assert monday.strftime("%a %-d %b") in actions[0].summary


# Unanswered policy questions


async def unanswered(db: AsyncSession) -> list[tuple[str, UnansweredReason]]:
    rows = await db.scalars(select(UnansweredQueryModel).order_by(UnansweredQueryModel.id))
    return [(r.query, r.reason) for r in rows]


async def test_a_search_that_finds_nothing_is_logged_as_unanswered(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
    retriever: InMemoryRetriever,
) -> None:
    me: EmployeeModel = await make_employee()
    # Nothing can score above 1, so every search comes back empty.
    use_agent(
        scripted(
            AIMessage(content="", tool_calls=[call("search_policies", query="pet insurance")]),
            "The policies available to you don't cover pet insurance. Please ask HR.",
        ),
        MinScoreRetriever(retriever, 1.01),
    )
    await login(client, me)
    thread_id = await new_thread(client)

    events = await send(client, thread_id, "Is my dog covered by insurance?")

    assert named(events, "citations") == []
    assert await unanswered(db) == [
        ("Is my dog covered by insurance?", UnansweredReason.NO_RELEVANT_DOCS)
    ]


async def test_an_answer_citing_none_of_the_excerpts_is_low_confidence(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
    retriever: InMemoryRetriever,
) -> None:
    use_agent(
        scripted(
            AIMessage(content="", tool_calls=[call("search_policies", query="hotel limit")]),
            "I couldn't find a clear rule for that.",
        ),
        retriever,
    )
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    await send(client, thread_id, "What's the hotel limit in Paris?")

    assert await unanswered(db) == [
        ("What's the hotel limit in Paris?", UnansweredReason.LOW_CONFIDENCE)
    ]


async def test_a_turn_without_policy_search_is_not_logged(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    use_agent(scripted("Hello!"))
    await login(client, await make_employee())

    await send(client, await new_thread(client), "Hi")

    assert await unanswered(db) == []
