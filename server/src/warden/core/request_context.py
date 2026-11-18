"""Who sent the current request, readable from anywhere in its call stack.

Services record the client's IP in the audit log without taking a `Request`
parameter, so the same service code works when the agent calls it.

Behind a reverse proxy, the socket's peer is the proxy. Uvicorn must run with
`--proxy-headers --forwarded-allow-ips=<proxy address>` (task 11.1) so that
`scope["client"]` holds the real client address from X-Forwarded-For.
"""

import ipaddress
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_USER_AGENT_LENGTH = 512


@dataclass(frozen=True)
class RequestContext:
    ip: str | None = None
    user_agent: str | None = None
    # Set by the request ID middleware once task 0.13 adds it.
    request_id: str | None = None


# Outside a request (scripts, tests that call services directly) every field
# is None. Sharing one default is safe because RequestContext is frozen.
_request_context: ContextVar[RequestContext] = ContextVar(
    "request_context",
    default=RequestContext(),  # noqa: B039
)


def get_request_context() -> RequestContext:
    return _request_context.get()


@contextmanager
def use_request_context(context: RequestContext) -> Iterator[None]:
    """Make `context` the current one until the block exits."""
    token = _request_context.set(context)
    try:
        yield
    finally:
        _request_context.reset(token)


def valid_ip_or_none(value: str | None) -> str | None:
    """The address in canonical form, or None if it isn't an IP address.
    INET columns reject anything else."""
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


class RequestContextMiddleware:
    """Sets the RequestContext for each HTTP request.

    A plain ASGI middleware rather than BaseHTTPMiddleware, so the endpoint
    and its dependencies run in the same context the variable is set in.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        user_agent = Headers(scope=scope).get("user-agent")
        context = RequestContext(
            ip=valid_ip_or_none(client[0] if client else None),
            user_agent=user_agent[:MAX_USER_AGENT_LENGTH] if user_agent else None,
        )
        with use_request_context(context):
            await self.app(scope, receive, send)
