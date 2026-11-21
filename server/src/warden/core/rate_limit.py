"""Request rate limits, counted in Redis so every worker shares them.

Three limits, all per minute (task 9.1):

- every request, per client IP (`RateLimitMiddleware`)
- login attempts, per IP and per email (`services.auth.check_login_rate_limit`)
- chat turns, per user (`routers.chat.limit_chat`)

Each limit is a sliding window log: a sorted set per key holding the time of
every request let through in the last minute. Unlike a fixed window, it never
allows twice the limit across a window boundary, and it knows exactly when
the oldest request ages out, which is the `Retry-After` value. Refused
requests aren't added, so a client that keeps retrying gets in again as soon
as the window has room.

When Redis can't be reached the limiter lets requests through and logs the
error. The limits are a second line of defense (passwords are argon2, chat
needs a login), and a Redis outage shouldn't take the whole app down.
"""

import hashlib
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.types import ASGIApp, Receive, Scope, Send

from warden.core.config import Settings
from warden.core.request_context import get_request_context

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60

# Liveness checks come from the container runtime and uptime monitors, often
# all from one address.
EXEMPT_PATHS = frozenset({"/health"})

# Checks every rule and records the request in all of them only if none is
# full, so a request refused by one limit doesn't use up the others.
#
# KEYS: one sorted set per rule.
# ARGV: now (ms), a unique member for this request, then limit and window
#       (ms) for each key in order.
# Returns {1} when allowed. When refused: {0, retry_after_ms, new_lockout,
# indexes of the full rules...}. new_lockout is 1 when a full rule had no
# lockout marker yet, which is how callers log a lockout once instead of
# once per refused request. A lockout lasts until a request is let through.
_HIT_SCRIPT = """
local now = tonumber(ARGV[1])
local member = ARGV[2]
local retry_after = 0
local new_lockout = 0
local full = {}
for i, key in ipairs(KEYS) do
  local limit = tonumber(ARGV[1 + 2 * i])
  local window = tonumber(ARGV[2 + 2 * i])
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
  local count = redis.call('ZCARD', key)
  if count >= limit then
    -- The request that has to age out before there's room again.
    local entry = redis.call('ZRANGE', key, count - limit, count - limit, 'WITHSCORES')
    local wait = tonumber(entry[2]) + window - now
    if wait > retry_after then retry_after = wait end
    if redis.call('SET', key .. ':locked', '1', 'NX', 'PX', math.max(wait, 1)) then
      new_lockout = 1
    end
    table.insert(full, i)
  end
end
if #full > 0 then
  return {0, retry_after, new_lockout, unpack(full)}
end
for i, key in ipairs(KEYS) do
  redis.call('ZADD', key, now, member)
  redis.call('PEXPIRE', key, tonumber(ARGV[2 + 2 * i]))
  -- Let through, so any lockout is over. The marker's own expiry only
  -- cleans up after keys nobody uses again.
  redis.call('DEL', key .. ':locked')
end
return {1}
"""


@dataclass(frozen=True)
class RateLimits:
    """Requests allowed per minute."""

    login: int
    chat: int
    requests: int


@dataclass(frozen=True)
class Rule:
    """One limit to check. `name` says which limit refused a request, in
    logs and audit rows; `key` is what's counted (an IP, a user)."""

    name: str
    key: str
    limit: int
    window_seconds: int = WINDOW_SECONDS


@dataclass(frozen=True)
class Decision:
    allowed: bool
    # Whole seconds until a retry can succeed. 0 when allowed.
    retry_after: int = 0
    # Names of the rules that were full.
    exceeded: tuple[str, ...] = ()
    # The first refusal since the rule filled up. Later refusals, until there's
    # room again, are False.
    new_lockout: bool = False


ALLOWED = Decision(allowed=True)


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        limits: RateLimits,
        *,
        prefix: str = "ratelimit",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.redis = redis
        self.limits = limits
        self._prefix = prefix
        self._clock = clock
        self._script = redis.register_script(_HIT_SCRIPT)

    async def hit(self, *rules: Rule) -> Decision:
        """Count one request against every rule, unless one of them is full."""
        if not rules:
            return ALLOWED
        args: list[int | str] = [int(self._clock() * 1000), uuid.uuid4().hex]
        for rule in rules:
            args += [rule.limit, rule.window_seconds * 1000]
        try:
            result = await self._script(
                keys=[f"{self._prefix}:{rule.name}:{rule.key}" for rule in rules], args=args
            )
        except RedisError as exc:
            logger.error("Rate limiter unavailable, letting the request through: %s", exc)
            return ALLOWED

        if result[0] == 1:
            return ALLOWED
        retry_after_ms, new_lockout, *full = result[1:]
        return Decision(
            allowed=False,
            retry_after=max(1, math.ceil(int(retry_after_ms) / 1000)),
            exceeded=tuple(rules[int(i) - 1].name for i in full),
            new_lockout=bool(new_lockout),
        )

    async def aclose(self) -> None:
        await self.redis.aclose()


def build_rate_limiter(settings: Settings) -> RateLimiter | None:
    """None when RATE_LIMIT_ENABLED is false. Connects on first use."""
    if not settings.rate_limit_enabled:
        return None
    redis = Redis.from_url(
        settings.redis_url,
        # A Redis that stops answering would otherwise hold up every request.
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
    )
    return RateLimiter(
        redis,
        RateLimits(
            login=settings.login_rate_limit,
            chat=settings.chat_rate_limit,
            requests=settings.request_rate_limit,
        ),
    )


def hashed(value: str) -> str:
    """For keys built from personal data (emails), so Redis doesn't hold a
    list of them."""
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def too_many_requests(decision: Decision, message: str = "Too many requests") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"{message}. Try again in {_seconds(decision.retry_after)}.",
        headers={"Retry-After": str(decision.retry_after)},
    )


def _seconds(n: int) -> str:
    return "1 second" if n == 1 else f"{n} seconds"


def get_rate_limiter(request: Request) -> RateLimiter | None:
    """Built by `create_app`. None when rate limits are off."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    return limiter


RateLimiterDep = Annotated[RateLimiter | None, Depends(get_rate_limiter)]


class RateLimitMiddleware:
    """The per-IP limit on every request.

    Runs inside RequestContextMiddleware, which has already validated the
    client IP and adds the request ID to the 429. Reads the limiter from the
    app's state on each request, so it's a no-op when limits are off.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limiter: RateLimiter | None = getattr(scope["app"].state, "rate_limiter", None)
        ip = get_request_context().ip
        if (
            scope["type"] != "http"
            or limiter is None
            or ip is None
            or scope["path"] in EXEMPT_PATHS
        ):
            await self.app(scope, receive, send)
            return

        decision = await limiter.hit(Rule("requests:ip", ip, limiter.limits.requests))
        if decision.allowed:
            await self.app(scope, receive, send)
            return

        if decision.new_lockout:
            logger.warning("Rate limit reached for %s", ip)
        exc = too_many_requests(decision)
        response = JSONResponse(
            status_code=exc.status_code, content={"message": exc.detail}, headers=exc.headers
        )
        await response(scope, receive, send)
