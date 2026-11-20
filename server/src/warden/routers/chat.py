"""Chat threads and the streaming chat endpoint. Every route works only on
the caller's own threads: someone else's thread is a 404."""

import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.database import SessionLocal
from warden.core.dependencies import CurrentUser, CurrentUserDep, DbSession
from warden.graph.graph import build_graph
from warden.graph.llm_calls import LLMCallRecorder
from warden.graph.pending_actions import PendingActions
from warden.graph.prompts import build_system_prompt, cancellation_note, confirmation_note
from warden.graph.resources import AgentResources
from warden.graph.state import AgentState
from warden.graph.streaming import ChatTurn, sse
from warden.graph.tools import ToolServices, build_tools
from warden.models import PendingActionModel
from warden.schemas.chat import (
    MessagePage,
    MessageResponse,
    SendMessage,
    ThreadPage,
    ThreadResponse,
)
from warden.services.chat import MAX_PAGE_SIZE, ChatService
from warden.services.leave import LeaveService, company_today

router = APIRouter(prefix="/api/chat", tags=["chat"])

SessionFactory = Callable[[], AsyncSession]


def get_chat_service(db: DbSession) -> ChatService:
    return ChatService(db)


def get_agent(request: Request) -> AgentResources:
    """Opened once at startup by the lifespan."""
    agent: AgentResources = request.app.state.agent
    return agent


def get_session_factory() -> SessionFactory:
    """For writes that must not share the request's session (LLM call
    stats). Tests override it to stay in their transaction."""
    return SessionLocal


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
AgentDep = Annotated[AgentResources, Depends(get_agent)]
Before = Annotated[uuid.UUID | None, Query(description="The `next` value from the previous page.")]


@router.post("/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(user: CurrentUserDep, chat: ChatServiceDep) -> ThreadResponse:
    return ThreadResponse.model_validate(await chat.create_thread(user))


@router.get("/threads")
async def list_threads(
    user: CurrentUserDep,
    chat: ChatServiceDep,
    before: Before = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
) -> ThreadPage:
    """The caller's threads, newest first."""
    page = await chat.list_threads(user, before, limit)
    return ThreadPage(
        threads=[ThreadResponse.model_validate(t) for t in page.items], next=page.next
    )


@router.get("/threads/{thread_id}/messages")
async def list_messages(
    thread_id: uuid.UUID,
    user: CurrentUserDep,
    chat: ChatServiceDep,
    before: Before = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> MessagePage:
    """The thread's messages, newest first."""
    page = await chat.list_messages(user, thread_id, before, limit)
    return MessagePage(
        messages=[MessageResponse.model_validate(m) for m in page.items], next=page.next
    )


@router.post(
    "/threads/{thread_id}/messages",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: token, tool_start, tool_end, citations, "
            "confirmation_required, done, error.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def send_message(
    thread_id: uuid.UUID,
    body: SendMessage,
    user: CurrentUserDep,
    db: DbSession,
    chat: ChatServiceDep,
    agent: AgentDep,
    session_factory: Annotated[SessionFactory, Depends(get_session_factory)],
) -> StreamingResponse:
    """Ask the agent. The answer streams as server-sent events (see
    `warden.graph.streaming`). An unknown thread is a 404 and an empty or
    over-long message a 422, both before anything streams."""
    thread = await chat.get_thread(user, thread_id)
    model = _require_model(agent)
    await chat.add_user_message(thread, body.message)

    turn = ChatTurn(
        graph=_build_graph(model, user, db, agent, thread.id),
        chat=chat,
        user=user,
        thread=thread,
        message=body.message,
        callbacks=[LLMCallRecorder(session_factory, thread.id)],
    )
    return _event_stream(turn.events())


@router.post(
    "/actions/{action_id}/confirm",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: action_result first, then the agent's "
            "reply like a message stream.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def confirm_action(
    action_id: uuid.UUID,
    user: CurrentUserDep,
    db: DbSession,
    chat: ChatServiceDep,
    agent: AgentDep,
    session_factory: Annotated[SessionFactory, Depends(get_session_factory)],
) -> StreamingResponse:
    """Run a change the agent proposed, then stream the agent's follow-up.

    Someone else's action, or an unknown one, is a 404. One that's already
    confirmed, cancelled, failed or expired is a 422 (`invalid_status`,
    `action_expired`). These come before anything streams. If the rules
    refuse the change when it runs (the balance changed since the card was
    shown), the stream says `failed` and the agent explains why.
    """
    model = _require_model(agent)
    outcome = await PendingActions(db).confirm(user, action_id, LeaveService(db))
    note = confirmation_note(outcome.action.summary, outcome.ok, outcome.result)
    return await _follow_up(
        model, user, db, chat, agent, session_factory, outcome.action, note, ok=outcome.ok
    )


@router.post(
    "/actions/{action_id}/cancel",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: action_result first, then the agent's "
            "reply like a message stream.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def cancel_action(
    action_id: uuid.UUID,
    user: CurrentUserDep,
    db: DbSession,
    chat: ChatServiceDep,
    agent: AgentDep,
    session_factory: Annotated[SessionFactory, Depends(get_session_factory)],
) -> StreamingResponse:
    """Drop a change the agent proposed, then stream a short reply from the
    agent. Errors as for confirm, before anything streams."""
    model = _require_model(agent)
    action = await PendingActions(db).cancel(user, action_id)
    note = cancellation_note(action.summary)
    return await _follow_up(model, user, db, chat, agent, session_factory, action, note, ok=True)


async def _follow_up(
    model: BaseChatModel,
    user: CurrentUser,
    db: AsyncSession,
    chat: ChatService,
    agent: AgentResources,
    session_factory: SessionFactory,
    action: PendingActionModel,
    note: str,
    *,
    ok: bool,
) -> StreamingResponse:
    """`action_result` for the card, then the agent's reply to `note` in the
    action's thread."""
    thread = await chat.get_thread(user, action.thread_id)
    turn = ChatTurn(
        graph=_build_graph(model, user, db, agent, thread.id),
        chat=chat,
        user=user,
        thread=thread,
        message=note,
        callbacks=[LLMCallRecorder(session_factory, thread.id)],
        from_user=False,
    )
    result = {"action_id": str(action.id), "status": action.status.value, "ok": ok}

    async def events() -> AsyncIterator[str]:
        yield sse("action_result", result)
        async for event in turn.events():
            yield event

    return _event_stream(events())


def _require_model(agent: AgentResources) -> BaseChatModel:
    if agent.chat_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is unavailable: no language model is configured.",
        )
    return agent.chat_model


def _build_graph(
    model: BaseChatModel,
    user: CurrentUser,
    db: AsyncSession,
    agent: AgentResources,
    thread_id: uuid.UUID,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Built per request: the tools and the prompt are for this user, and the
    tools use this request's session."""
    return build_graph(
        model,
        build_tools(user, ToolServices.for_session(db), agent.retriever, thread_id),
        build_system_prompt(user, company_today()),
        agent.checkpointer,
    )


def _event_stream(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        # Stop proxies from buffering the stream or caching it.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
