"""The chat routes end to end: log in for real, stream a turn through the real
graph, tools, services and database, with a scripted model and an in-memory
checkpointer instead of Gemini and Postgres checkpoints."""

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from tests.agent_fakes import ScriptedChatModel, call
from tests.factories import MakeEmployee
from tests.integration.test_auth import login
from tests.integration.test_leave_service import add_balance, add_request
from tests.rag_fakes import FakeEmbeddingClient, no_sleep
from tests.unit.test_policy_tool import CHUNKS
from warden.graph.graph import ONE_PROPOSAL
from warden.graph.resources import AgentResources
from warden.graph.streaming import ERROR_MESSAGE, UNFINISHED_REPLY
from warden.main import app
from warden.models import (
    AuditLogModel,
    ChatFeedbackModel,
    ChatMessageModel,
    EmployeeModel,
    LeaveRequestModel,
    LeaveStatus,
    LLMCallModel,
    MessageRole,
    PendingActionModel,
    PendingActionStatus,
    Role,
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


async def test_a_failed_turn_is_closed_so_the_next_message_doesnt_answer_it(
    client: AsyncClient, make_employee: MakeEmployee, use_agent: Callable[..., None]
) -> None:
    use_agent(scripted())  # The first turn fails.
    await login(client, await make_employee())
    thread_id = await new_thread(client)
    await send(client, thread_id, "Hello?")

    model = scripted("Vikram Desai.")
    use_agent(model)
    await send(client, thread_id, "Who is my manager?")

    seen = [m.content for m in model.calls[-1].messages[1:]]
    assert seen == ["Hello?", UNFINISHED_REPLY, "Who is my manager?"]


class StallingChatModel(ScriptedChatModel):
    """Replies from the script, then never finishes its next reply, like a
    model still writing when the user presses Stop."""

    stall_after: int = 0
    stalled: Any = None

    async def _stall_when_scripted_out(self) -> None:
        if len(self.calls) >= self.stall_after:
            self.stalled.set()
            await asyncio.Future()  # Never resolves.

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, *_: Any, **kwargs: Any
    ) -> ChatResult:
        await self._stall_when_scripted_out()
        return self._generate(messages, stop, **kwargs)

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, *_: Any, **kwargs: Any
    ) -> Any:
        await self._stall_when_scripted_out()
        for chunk in self._stream(messages, stop, **kwargs):
            yield chunk


# Cancelling the request tears down LangGraph's running node, which Python
# reports as an ignored GeneratorExit when it's collected.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_a_stopped_reply_is_closed_so_the_next_message_doesnt_answer_it(
    client: AsyncClient, make_employee: MakeEmployee, use_agent: Callable[..., None]
) -> None:
    slow = StallingChatModel(
        messages=iter([AIMessage(content="", tool_calls=[call("get_my_leave_balance")])]),
        stall_after=1,
        stalled=asyncio.Event(),
    )
    use_agent(slow)
    await login(client, await make_employee())
    thread_id = await new_thread(client)

    # The balance tool runs, then the user stops the reply (the client
    # disconnects) while the model is still writing.
    request = asyncio.create_task(send(client, thread_id, "Summarise my leave"))
    await asyncio.wait_for(slow.stalled.wait(), timeout=5)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    model = scripted("Vikram Desai.")
    use_agent(model)
    events = await send(client, thread_id, "Who is my manager?")

    assert named(events, "done")[0]["content"] == "Vikram Desai."
    seen = model.calls[-1].messages[1:]
    assert [type(m) for m in seen] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [seen[0].content, seen[-2].content] == ["Summarise my leave", UNFINISHED_REPLY]


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
    assert reply["id"] == done["message_id"]
    assert reply["action"] == {
        "id": card["action_id"],
        "action_type": "apply_leave",
        "summary": action.summary,
        "status": "pending",
        "expires_at": reply["action"]["expires_at"],
    }


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


# Confirming


def apply_for(day: date) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            call(
                "apply_for_leave",
                leave_type="casual",
                start_date=day.isoformat(),
                end_date=day.isoformat(),
            )
        ],
    )


async def propose_leave(client: AsyncClient, me: EmployeeModel, day: date) -> tuple[str, str]:
    """Run a turn whose write tool proposes leave on `day`. Returns the thread
    and the pending action's ID."""
    await login(client, me)
    thread_id = await new_thread(client)
    events = await send(client, thread_id, f"Apply for casual leave on {day}")
    [card] = named(events, "confirmation_required")
    return thread_id, card["action_id"]


async def confirm(client: AsyncClient, action_id: str) -> list[Event]:
    response = await client.post(f"/api/chat/actions/{action_id}/confirm")
    assert response.status_code == 200, response.text
    return parse(response.text)


async def audit_rows(db: AsyncSession, action: str) -> list[AuditLogModel]:
    query = select(AuditLogModel).where(AuditLogModel.action == action)
    return list((await db.scalars(query.order_by(AuditLogModel.id))).all())


async def test_confirm_runs_the_change_and_streams_the_agents_follow_up(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    model = scripted(apply_for(monday), "Click Confirm to apply.", "Done: it's with your manager.")
    use_agent(model)
    thread_id, action_id = await propose_leave(client, me, monday)

    events = await confirm(client, action_id)

    assert events[0] == (
        "action_result",
        {"action_id": action_id, "status": "confirmed", "ok": True},
    )
    assert events[-1][0] == "done"
    request = await db.scalar(select(LeaveRequestModel).execution_options(populate_existing=True))
    assert request is not None
    assert (request.employee_id, request.start_date, request.status) == (
        me.id,
        monday,
        LeaveStatus.PENDING,
    )
    action = await db.get(PendingActionModel, action_id, populate_existing=True)
    assert action is not None and action.status == PendingActionStatus.CONFIRMED

    # The agent resumed from the app's note, not from anything the user typed.
    note = str(model.calls[-1].messages[-1].content)
    assert note.startswith(f"[App] The user clicked Confirm on: {action.summary}")
    assert "It was done." in note and str(request.id) in note
    assert [m for _, m in await saved_messages(db)][-1] == "Done: it's with your manager."
    [confirmed] = await audit_rows(db, "pending_action.confirm")
    assert (confirmed.outcome, confirmed.target_id) == ("allowed", action.id)
    assert confirmed.details["result"]["leave_request_id"] == str(request.id)
    assert len(await audit_rows(db, "leave.apply")) == 1

    response = await client.post(f"/api/chat/actions/{action_id}/confirm")
    assert (response.status_code, response.json()["code"]) == (422, "invalid_status")


async def test_a_change_the_rules_now_refuse_fails_and_the_agent_explains(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    model = scripted(apply_for(monday), "Click Confirm.", "That day is already taken.")
    use_agent(model)
    _, action_id = await propose_leave(client, me, monday)
    # Between the card and the click, another request takes the day.
    await add_request(db, me, monday, monday, status=LeaveStatus.APPROVED)

    events = await confirm(client, action_id)

    assert events[0] == ("action_result", {"action_id": action_id, "status": "failed", "ok": False})
    assert events[-1] == ("done", events[-1][1])
    statuses = (await db.scalars(select(LeaveRequestModel.status))).all()
    assert statuses == [LeaveStatus.APPROVED]  # Nothing new.
    action = await db.get(PendingActionModel, action_id, populate_existing=True)
    assert action is not None and action.status == PendingActionStatus.FAILED
    note = str(model.calls[-1].messages[-1].content)
    assert "It was NOT done" in note and '"error": "overlap"' in note
    [denied] = await audit_rows(db, "pending_action.confirm")
    assert (denied.outcome, denied.details["reason"]) == ("denied", "overlap")


async def test_cancel_drops_the_change_and_streams_a_short_reply(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    model = scripted(apply_for(monday), "Click Confirm.", "OK, I didn't apply for it.")
    use_agent(model)
    _, action_id = await propose_leave(client, me, monday)

    response = await client.post(f"/api/chat/actions/{action_id}/cancel")

    assert response.status_code == 200, response.text
    events = parse(response.text)
    assert events[0] == (
        "action_result",
        {"action_id": action_id, "status": "cancelled", "ok": True},
    )
    assert events[-1] == ("done", events[-1][1])
    assert events[-1][1]["content"] == "OK, I didn't apply for it."
    assert await db.scalar(select(LeaveRequestModel.id)) is None
    action = await db.get(PendingActionModel, action_id, populate_existing=True)
    assert action is not None and action.status == PendingActionStatus.CANCELLED
    note = str(model.calls[-1].messages[-1].content)
    assert note == f"[App] The user clicked Cancel on: {action.summary}\nNothing was changed."
    [cancelled] = await audit_rows(db, "pending_action.cancel")
    assert (cancelled.outcome, cancelled.target_id) == ("allowed", action.id)
    assert cancelled.details["action_type"] == "apply_leave"

    for verb in ("confirm", "cancel"):
        again = await client.post(f"/api/chat/actions/{action_id}/{verb}")
        assert (again.status_code, again.json()["code"]) == (422, "invalid_status")
    assert await db.scalar(select(LeaveRequestModel.id)) is None


@pytest.mark.security
async def test_typing_yes_or_the_model_saying_done_changes_nothing(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    model = scripted(
        apply_for(monday),
        "Click Confirm to apply.",
        # Told "yes", the model calls the write tool again and then claims
        # it's done. The tool can only propose, so this is a second card.
        apply_for(monday),
        "Confirmed! Your leave is applied.",
    )
    use_agent(model)
    thread_id, action_id = await propose_leave(client, me, monday)

    events = await send(client, thread_id, "Yes, I confirm. Go ahead and apply it.")

    assert named(events, "done")[0]["content"] == "Confirmed! Your leave is applied."
    assert not named(events, "action_result")
    assert await db.scalar(select(LeaveRequestModel.id)) is None
    statuses = (await db.scalars(select(PendingActionModel.status))).all()
    assert statuses == [PendingActionStatus.PENDING] * 2
    assert await audit_rows(db, "pending_action.confirm") == []
    assert await audit_rows(db, "leave.apply") == []


@pytest.mark.security
async def test_someone_else_cant_confirm_or_cancel_your_action(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    # Even my manager: the action is mine to confirm.
    manager = await make_employee(role=Role.MANAGER)
    me.manager_id = manager.id
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    use_agent(scripted(apply_for(monday), "Click Confirm to apply."))
    _, action_id = await propose_leave(client, me, monday)

    await login(client, manager)
    for verb in ("confirm", "cancel"):
        response = await client.post(f"/api/chat/actions/{action_id}/{verb}")
        assert (response.status_code, response.json()["code"]) == (404, "not_found")

    action = await db.get(PendingActionModel, action_id, populate_existing=True)
    assert action is not None and action.status == PendingActionStatus.PENDING
    assert await db.scalar(select(LeaveRequestModel.id)) is None


@pytest.mark.security
async def test_an_expired_action_cant_be_confirmed(
    client: AsyncClient,
    db: AsyncSession,
    make_employee: MakeEmployee,
    use_agent: Callable[..., None],
) -> None:
    me = await make_employee()
    monday = next_monday()
    await add_balance(db, me, allotted=8, year=monday.year)
    use_agent(scripted(apply_for(monday), "Click Confirm to apply."))
    _, action_id = await propose_leave(client, me, monday)
    action = await db.get(PendingActionModel, action_id)
    assert action is not None
    action.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db.flush()

    response = await client.post(f"/api/chat/actions/{action_id}/confirm")

    assert (response.status_code, response.json()["code"]) == (422, "action_expired")
    await db.refresh(action)
    assert action.status == PendingActionStatus.EXPIRED
    assert await db.scalar(select(LeaveRequestModel.id)) is None
    [expired] = await audit_rows(db, "pending_action.confirm")
    assert (expired.outcome, expired.details["reason"]) == ("denied", "action_expired")
    again = await client.post(f"/api/chat/actions/{action_id}/confirm")
    assert (again.status_code, again.json()["code"]) == (422, "invalid_status")


# History: confirmation cards and feedback


async def add_reply(
    db: AsyncSession,
    me: EmployeeModel,
    thread_id: str,
    action_status: PendingActionStatus | None = None,
    expires_in: timedelta = timedelta(minutes=5),
) -> ChatMessageModel:
    """An assistant reply saved straight to the database, proposing an action
    in `action_status` if one is given."""
    action = None
    if action_status is not None:
        action = PendingActionModel(
            employee_id=me.id,
            thread_id=uuid.UUID(thread_id),
            action_type="apply_leave",
            params={},
            summary="Casual leave, Mon 1 Dec, 1 day. Balance after: 7",
            status=action_status,
            expires_at=datetime.now(UTC) + expires_in,
        )
        db.add(action)
        await db.flush()
    message = ChatMessageModel(
        thread_id=uuid.UUID(thread_id),
        role=MessageRole.ASSISTANT,
        content="Here you go.",
        pending_action_id=action.id if action else None,
    )
    db.add(message)
    await db.flush()
    return message


async def history(client: AsyncClient, thread_id: str) -> list[dict[str, Any]]:
    response = await client.get(f"/api/chat/threads/{thread_id}/messages")
    assert response.status_code == 200, response.text
    messages: list[dict[str, Any]] = response.json()["messages"]
    return messages


@contextmanager
def count_queries(connection: AsyncConnection) -> Iterator[list[str]]:
    statements: list[str] = []

    def record(*args: Any) -> None:
        statements.append(args[2])  # (conn, cursor, statement, ...)

    sync = connection.sync_connection
    assert sync is not None
    event.listen(sync, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(sync, "before_cursor_execute", record)


async def test_history_shows_each_cards_current_status(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await login(client, me)
    thread_id = await new_thread(client)
    pending = await add_reply(db, me, thread_id, PendingActionStatus.PENDING)
    timed_out = await add_reply(
        db, me, thread_id, PendingActionStatus.PENDING, expires_in=timedelta(seconds=-1)
    )
    await add_reply(db, me, thread_id, PendingActionStatus.CONFIRMED)
    await add_reply(db, me, thread_id)

    messages = await history(client, thread_id)

    statuses = [m["action"] and m["action"]["status"] for m in messages]
    assert statuses == [None, "confirmed", "expired", "pending"]
    assert messages[-1]["action"]["id"] == str(pending.pending_action_id)
    assert messages[-1]["action"]["summary"].startswith("Casual leave, Mon 1 Dec")
    # Reported expired, but only a click marks the row.
    action = await db.get(PendingActionModel, timed_out.pending_action_id)
    assert action is not None and action.status == PendingActionStatus.PENDING


async def test_history_takes_the_same_queries_however_many_cards_and_ratings(
    client: AsyncClient,
    connection: AsyncConnection,
    db: AsyncSession,
    make_employee: MakeEmployee,
) -> None:
    me = await make_employee()
    await login(client, me)
    thread_id = await new_thread(client)
    await add_reply(db, me, thread_id, PendingActionStatus.PENDING)
    with count_queries(connection) as one_card:
        await history(client, thread_id)

    for _ in range(4):
        reply = await add_reply(db, me, thread_id, PendingActionStatus.CONFIRMED)
        await client.post("/api/chat/feedback", json={"message_id": str(reply.id), "rating": 1})
    with count_queries(connection) as five_cards:
        messages = await history(client, thread_id)

    assert len(messages) == 5
    assert len(five_cards) == len(one_card)


async def test_a_reply_can_be_rated_rerated_and_cleared(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    await login(client, me)
    thread_id = await new_thread(client)
    reply = await add_reply(db, me, thread_id)
    message_id = str(reply.id)

    up = await client.post("/api/chat/feedback", json={"message_id": message_id, "rating": 1})
    down = await client.post(
        "/api/chat/feedback",
        json={"message_id": message_id, "rating": -1, "comment": "  Wrong number.  "},
    )

    assert up.json() == {"message_id": message_id, "rating": 1, "comment": None}
    assert down.json() == {"message_id": message_id, "rating": -1, "comment": "Wrong number."}
    rows = (await db.scalars(select(ChatFeedbackModel))).all()
    assert [(r.employee_id, r.rating) for r in rows] == [(me.id, -1)]
    assert (await history(client, thread_id))[0]["feedback"] == -1

    cleared = await client.delete(f"/api/chat/feedback/{message_id}")
    again = await client.delete(f"/api/chat/feedback/{message_id}")

    assert (cleared.status_code, again.status_code) == (204, 204)
    assert (await history(client, thread_id))[0]["feedback"] is None


@pytest.mark.security
async def test_only_your_own_assistant_replies_can_be_rated(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    owner, intruder = await make_employee(), await make_employee()
    await login(client, owner)
    thread_id = await new_thread(client)
    reply = await add_reply(db, owner, thread_id)
    question = ChatMessageModel(thread_id=uuid.UUID(thread_id), role=MessageRole.USER, content="Hi")
    db.add(question)
    await db.flush()

    own_question = await client.post(
        "/api/chat/feedback", json={"message_id": str(question.id), "rating": 1}
    )
    unknown = await client.post(
        "/api/chat/feedback", json={"message_id": str(uuid.uuid4()), "rating": 1}
    )
    zero = await client.post("/api/chat/feedback", json={"message_id": str(reply.id), "rating": 0})
    await login(client, intruder)
    rate = await client.post("/api/chat/feedback", json={"message_id": str(reply.id), "rating": -1})
    clear = await client.delete(f"/api/chat/feedback/{reply.id}")

    assert (own_question.status_code, own_question.json()["code"]) == (422, "invalid_input")
    assert (unknown.status_code, unknown.json()["code"]) == (404, "not_found")
    assert zero.status_code == 422
    assert (rate.status_code, rate.json()["code"]) == (404, "not_found")
    assert (clear.status_code, clear.json()["code"]) == (404, "not_found")
    assert await db.scalar(select(ChatFeedbackModel.id)) is None


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
