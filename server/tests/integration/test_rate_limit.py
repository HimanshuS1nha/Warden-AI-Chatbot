"""Rate limits (warden.core.rate_limit) against a real Redis.

Uses database 15 of the configured Redis (TEST_REDIS_URL overrides it) and a
key prefix per test, so nothing collides with development data or another
test. The clock is fake: requests are placed in time by hand instead of
sleeping through a window.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlsplit, urlunsplit

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import PASSWORD, MakeEmployee
from tests.integration.test_auth import login
from warden.core.config import get_settings
from warden.core.rate_limit import RateLimiter, RateLimits, Rule
from warden.core.request_context import REQUEST_ID_HEADER
from warden.graph.resources import AgentResources
from warden.main import app
from warden.models import AuditAction, AuditLogModel, AuditOutcome
from warden.routers.chat import get_agent


def _test_redis_url() -> str:
    if "TEST_REDIS_URL" in os.environ:
        return os.environ["TEST_REDIS_URL"]
    return urlunsplit(urlsplit(get_settings().redis_url)._replace(path="/15"))


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(_test_redis_url())
    yield client
    await client.aclose()


@pytest.fixture
async def limiter(redis: Redis, clock: Clock) -> AsyncIterator[RateLimiter]:
    prefix = f"test:{uuid.uuid4().hex}"
    limiter = RateLimiter(
        redis, RateLimits(login=5, chat=3, requests=100), prefix=prefix, clock=clock
    )
    yield limiter
    keys = [key async for key in redis.scan_iter(f"{prefix}:*")]
    if keys:
        await redis.delete(*keys)


@pytest.fixture
def limited_app(client: AsyncClient, limiter: RateLimiter) -> Iterator[RateLimiter]:
    """Turn rate limits on for the shared app. Depends on `client` so the
    app's database override is in place for clients made from other IPs."""
    app.state.rate_limiter = limiter
    yield limiter
    app.state.rate_limiter = None


def client_from(ip: str) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app, client=(ip, 1234)), base_url="http://test")


async def rate_limited_logins(db: AsyncSession) -> list[AuditLogModel]:
    rows = await db.scalars(
        select(AuditLogModel)
        .where(
            AuditLogModel.action == AuditAction.AUTH_LOGIN,
            AuditLogModel.details["reason"].astext == "rate_limited",
        )
        .order_by(AuditLogModel.id)
    )
    return list(rows)


# The limiter


async def test_refuses_over_the_limit_until_the_oldest_request_ages_out(
    limiter: RateLimiter, clock: Clock
) -> None:
    rule = Rule("test", "key", limit=3, window_seconds=60)
    start = clock.now

    for offset in (0, 10, 20):
        clock.now = start + offset
        assert (await limiter.hit(rule)).allowed

    clock.now = start + 30
    refused = await limiter.hit(rule)
    assert not refused.allowed
    assert refused.exceeded == ("test",)
    # The request at 0 leaves the window at 60.
    assert refused.retry_after == 30

    clock.now = start + 59.9
    assert not (await limiter.hit(rule)).allowed

    # Refused requests weren't counted, so the first one ageing out makes room.
    clock.now = start + 60
    assert (await limiter.hit(rule)).allowed
    clock.now = start + 61
    assert (await limiter.hit(rule)).retry_after == 9


async def test_a_full_rule_doesnt_use_up_the_others(limiter: RateLimiter) -> None:
    small = Rule("small", "key", limit=1)
    large = Rule("large", "key", limit=3)

    assert (await limiter.hit(small, large)).allowed
    refused = await limiter.hit(small, large)
    assert not refused.allowed
    assert refused.exceeded == ("small",)

    # Only the first request counted against `large`.
    assert (await limiter.hit(large)).allowed
    assert (await limiter.hit(large)).allowed
    assert (await limiter.hit(large)).exceeded == ("large",)


async def test_only_the_first_refusal_starts_a_lockout(limiter: RateLimiter, clock: Clock) -> None:
    rule = Rule("test", "key", limit=1, window_seconds=60)

    assert (await limiter.hit(rule)).allowed
    assert (await limiter.hit(rule)).new_lockout
    assert not (await limiter.hit(rule)).new_lockout

    clock.now += 60
    assert (await limiter.hit(rule)).allowed
    assert (await limiter.hit(rule)).new_lockout


async def test_keys_are_separate(limiter: RateLimiter) -> None:
    assert (await limiter.hit(Rule("test", "a", limit=1))).allowed
    assert (await limiter.hit(Rule("test", "b", limit=1))).allowed
    assert not (await limiter.hit(Rule("test", "a", limit=1))).allowed


async def test_lets_requests_through_when_redis_is_down() -> None:
    # Nothing listens on port 1.
    redis = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.5)
    limiter = RateLimiter(redis, RateLimits(login=1, chat=1, requests=1))
    rule = Rule("test", "key", limit=1)

    assert (await limiter.hit(rule)).allowed
    assert (await limiter.hit(rule)).allowed
    await limiter.aclose()


# Every request, per IP


async def test_requests_over_the_ip_limit_get_a_429_with_retry_after(
    limited_app: RateLimiter,
) -> None:
    limit = limited_app.limits.requests
    async with client_from("10.0.0.1") as noisy, client_from("10.0.0.2") as quiet:
        for _ in range(limit):
            # Unauthenticated requests count too.
            assert (await noisy.get("/api/auth/me")).status_code == 401

        response = await noisy.get("/api/auth/me")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"
        assert response.json() == {"message": "Too many requests. Try again in 60 seconds."}
        assert response.headers[REQUEST_ID_HEADER]

        assert (await quiet.get("/api/auth/me")).status_code == 401
        # Liveness checks are never limited.
        assert (await noisy.get("/health")).status_code == 200


# Login, per IP and per email


async def test_login_is_refused_after_five_attempts_even_with_the_right_password(
    limited_app: RateLimiter, make_employee: MakeEmployee, db: AsyncSession
) -> None:
    employee = await make_employee(email="asha@warden.demo")
    async with client_from("10.0.1.1") as attacker:
        for _ in range(limited_app.limits.login):
            response = await attacker.post(
                "/api/auth/login", json={"email": employee.email, "password": "wrong"}
            )
            assert response.status_code == 401

        response = await attacker.post(
            "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
        )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"
        assert response.json()["message"] == "Too many login attempts. Try again in 60 seconds."
        assert "set-cookie" not in response.headers

        # Refused again: same lockout, so no second audit row.
        response = await attacker.post(
            "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
        )
        assert response.status_code == 429

    [row] = await rate_limited_logins(db)
    assert row.outcome == AuditOutcome.DENIED
    assert row.actor_id is None
    assert str(row.ip) == "10.0.1.1"
    assert row.details["email"] == "asha@warden.demo"
    assert sorted(row.details["limits"]) == ["login:email", "login:ip"]
    assert row.details["retry_after"] == 60
    assert row.details["request_id"]


async def test_login_limit_per_email_holds_across_ips(
    limited_app: RateLimiter, make_employee: MakeEmployee, db: AsyncSession
) -> None:
    employee = await make_employee(email="asha@warden.demo")
    # Case doesn't make it a different email.
    emails = ["asha@warden.demo", "Asha@Warden.Demo"]
    for n in range(limited_app.limits.login):
        async with client_from(f"10.0.2.{n}") as c:
            response = await c.post(
                "/api/auth/login", json={"email": emails[n % 2], "password": "wrong"}
            )
            assert response.status_code == 401

    async with client_from("10.0.2.99") as c:
        response = await c.post(
            "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
        )
        assert response.status_code == 429

    [row] = await rate_limited_logins(db)
    assert row.details["limits"] == ["login:email"]


async def test_login_limit_per_ip_holds_across_emails(
    limited_app: RateLimiter, make_employee: MakeEmployee, db: AsyncSession
) -> None:
    employee = await make_employee()
    async with client_from("10.0.3.1") as c:
        for n in range(limited_app.limits.login):
            response = await c.post(
                "/api/auth/login", json={"email": f"guess{n}@warden.demo", "password": "x"}
            )
            assert response.status_code == 401

        response = await c.post(
            "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
        )
        assert response.status_code == 429

    [row] = await rate_limited_logins(db)
    assert row.details["limits"] == ["login:ip"]
    assert row.details["email"] == employee.email

    # Someone else, from another address, is unaffected.
    async with client_from("10.0.3.2") as c:
        response = await c.post(
            "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
        )
        assert response.status_code == 200


# Chat, per user


@pytest.fixture
def no_agent() -> None:
    # Requests are refused before the model would run.
    app.dependency_overrides[get_agent] = lambda: AgentResources(
        checkpointer=InMemorySaver(), chat_model=None, retriever=None, indexer=None
    )


@pytest.mark.usefixtures("no_agent")
async def test_chat_turns_are_limited_per_user(
    limited_app: RateLimiter, client: AsyncClient, make_employee: MakeEmployee
) -> None:
    first, second = await make_employee(), await make_employee()
    # A thread that doesn't exist is a 404, after the limit is checked.
    messages = f"/api/chat/threads/{uuid.uuid4()}/messages"

    await login(client, first)
    for _ in range(limited_app.limits.chat):
        assert (await client.post(messages, json={"message": "Hi"})).status_code == 404

    response = await client.post(messages, json={"message": "Hi"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert response.json()["message"] == (
        "You're sending messages too quickly. Try again in 60 seconds."
    )
    # Confirming and cancelling run the model as well, so they share the limit.
    for verb in ("confirm", "cancel"):
        url = f"/api/chat/actions/{uuid.uuid4()}/{verb}"
        assert (await client.post(url)).status_code == 429
    # Reading isn't a turn.
    assert (await client.get("/api/chat/threads")).status_code == 200

    await login(client, second)
    assert (await client.post(messages, json={"message": "Hi"})).status_code == 404
