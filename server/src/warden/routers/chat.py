"""Chat threads and the streaming chat endpoint. Every route works only on
the caller's own threads: someone else's thread is a 404."""

import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.database import SessionLocal
from warden.core.dependencies import CurrentUserDep, DbSession
from warden.graph.graph import build_graph
from warden.graph.llm_calls import LLMCallRecorder
from warden.graph.prompts import build_system_prompt
from warden.graph.resources import AgentResources
from warden.graph.streaming import ChatTurn
from warden.graph.tools import ToolServices, build_tools
from warden.schemas.chat import (
    MessagePage,
    MessageResponse,
    SendMessage,
    ThreadPage,
    ThreadResponse,
)
from warden.services.chat import MAX_PAGE_SIZE, ChatService
from warden.services.leave import company_today

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
            "done, error.",
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
    if agent.chat_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is unavailable: no language model is configured.",
        )
    await chat.add_user_message(thread, body.message)

    # Built per request: the tools and the prompt are for this user, and the
    # tools use this request's session.
    graph = build_graph(
        agent.chat_model,
        build_tools(user, ToolServices.for_session(db), agent.retriever, thread.id),
        build_system_prompt(user, company_today()),
        agent.checkpointer,
    )
    turn = ChatTurn(
        graph=graph,
        chat=chat,
        user=user,
        thread=thread,
        message=body.message,
        callbacks=[LLMCallRecorder(session_factory, thread.id)],
    )
    return StreamingResponse(
        turn.events(),
        media_type="text/event-stream",
        # Stop proxies from buffering the stream or caching it.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
