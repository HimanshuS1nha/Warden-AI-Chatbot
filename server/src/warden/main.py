import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from warden.core.config import Settings, get_settings
from warden.core.database import engine
from warden.core.exceptions import register_exception_handlers
from warden.core.logging import configure_logging
from warden.core.request_context import RequestContextMiddleware
from warden.graph.agent import build_chat_model
from warden.graph.checkpointer import open_checkpointer
from warden.routers import auth, employees, health, holidays, me, team

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` controls logging, CORS and the checkpointer;
    the database engine and JWT code still read `get_settings()` directly."""
    settings = settings or get_settings()

    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup and shutdown. Resources that live as long as the app (the
        agent's checkpointer and chat model, later the reranker model) are
        created here. The graph itself is built per request, because its
        tools are bound to the signed-in user."""
        # Without a key the rest of the app still runs; chat can't.
        app.state.chat_model = None
        if settings.gemini_api_key.get_secret_value():
            app.state.chat_model = build_chat_model(settings)
        else:
            logger.warning("GEMINI_API_KEY is not set; the chat agent is disabled")
        try:
            async with open_checkpointer(settings.checkpointer_database_url) as checkpointer:
                app.state.checkpointer = checkpointer
                yield
        finally:
            await engine.dispose()

    app = FastAPI(title="Warden", lifespan=lifespan)

    register_exception_handlers(app)

    # Added first, so it runs inside CORS: preflight requests never reach it.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(team.router)
    app.include_router(employees.router)
    app.include_router(holidays.router)

    return app


app = create_app()
