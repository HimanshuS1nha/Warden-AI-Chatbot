import io
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from warden.core.exceptions import register_exception_handlers
from warden.core.logging import json_formatter
from warden.core.request_context import RequestContextMiddleware, get_request_context

logger = logging.getLogger("warden.tests.request_id")

app = FastAPI()
register_exception_handlers(app)
app.add_middleware(RequestContextMiddleware)


@app.get("/context")
async def context() -> dict[str, str | None]:
    logger.info("handling request")
    return {"request_id": get_request_context().request_id}


@app.get("/crash")
async def crash() -> None:
    raise RuntimeError("boom")


def client() -> AsyncClient:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


def read_lines(handler_stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in handler_stream.getvalue().splitlines()]


@pytest.fixture
def json_log() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(json_formatter())
    warden = logging.getLogger("warden")
    old_level = warden.level
    warden.addHandler(handler)
    warden.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        warden.removeHandler(handler)
        warden.setLevel(old_level)


async def test_every_response_gets_its_own_request_id() -> None:
    async with client() as c:
        first = await c.get("/context")
        second = await c.get("/context")

    assert first.headers["X-Request-ID"] == first.json()["request_id"]
    assert second.headers["X-Request-ID"] == second.json()["request_id"]
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


async def test_client_supplied_request_id_is_ignored() -> None:
    async with client() as c:
        response = await c.get("/context", headers={"X-Request-ID": "chosen-by-client"})

    assert response.headers["X-Request-ID"] != "chosen-by-client"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


async def test_log_lines_are_json_with_the_request_id(json_log: io.StringIO) -> None:
    async with client() as c:
        response = await c.get("/context")

    [line] = read_lines(json_log)
    assert line["event"] == "handling request"
    assert line["level"] == "info"
    assert line["logger"] == "warden.tests.request_id"
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert "timestamp" in line


async def test_unhandled_errors_keep_the_request_id(json_log: io.StringIO) -> None:
    async with client() as c:
        response = await c.get("/crash")

    assert response.status_code == 500
    request_id = response.headers["X-Request-ID"]
    [line] = read_lines(json_log)
    assert line["event"] == "Unhandled exception on GET /crash"
    assert line["request_id"] == request_id
    assert "RuntimeError: boom" in str(line["exception"])


def test_no_request_id_outside_a_request(json_log: io.StringIO) -> None:
    logger.info("from a script")

    [line] = read_lines(json_log)
    assert "request_id" not in line
