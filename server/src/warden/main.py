from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from warden.core.config import Settings, get_settings
from warden.core.database import engine
from warden.core.exceptions import register_exception_handlers
from warden.core.logging import configure_logging
from warden.core.rate_limit import RateLimitMiddleware, build_rate_limiter
from warden.core.request_context import RequestContextMiddleware
from warden.graph.resources import open_agent_resources
from warden.routers import admin, auth, chat, employees, health, holidays, me, team


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` controls logging, CORS and the agent's
    resources; the database engine and JWT code still read `get_settings()`
    directly."""
    settings = settings or get_settings()

    configure_logging(settings.log_level)
    rate_limiter = build_rate_limiter(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup and shutdown. Resources that live as long as the app (the
        agent's checkpointer, chat model and policy retriever) are created
        here. The graph itself is built per request, because its tools are
        bound to the signed-in user."""
        try:
            async with open_agent_resources(settings) as agent:
                app.state.agent = agent
                yield
        finally:
            if rate_limiter is not None:
                await rate_limiter.aclose()
            await engine.dispose()

    app = FastAPI(title="Warden", lifespan=lifespan)

    # None when rate limits are off. Read from here by the middleware and
    # by `get_rate_limiter`, so tests can swap it.
    app.state.rate_limiter = rate_limiter

    register_exception_handlers(app)

    # Middleware added first runs innermost. Rate limiting runs inside the
    # request context (it needs the client IP, and its 429 gets a request
    # ID), which runs inside CORS: preflight requests reach neither, and the
    # browser can read a 429.
    app.add_middleware(RateLimitMiddleware)
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
    app.include_router(chat.router)
    app.include_router(admin.router)

    return app


app = create_app()
