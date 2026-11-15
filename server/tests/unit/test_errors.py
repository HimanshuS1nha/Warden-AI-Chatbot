from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from warden.core.exceptions import register_exception_handlers
from warden.schemas.auth import LoginRequest

app = FastAPI()
register_exception_handlers(app)


@app.post("/login")
async def login(data: LoginRequest) -> None:
    pass


@app.get("/crash")
async def crash() -> None:
    raise RuntimeError("SELECT * FROM employees WHERE password_hash = ...")


def client() -> AsyncClient:
    # raise_app_exceptions=False: Starlette re-raises unhandled errors after
    # sending the 500; we want the response a real client would get.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_unknown_route_uses_the_error_shape() -> None:
    async with client() as c:
        response = await c.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"message": "Not Found"}


async def test_validation_errors_never_echo_the_input() -> None:
    async with client() as c:
        response = await c.post(
            "/login", json={"email": "not-an-email", "password": "hunter2-secret"}
        )

    assert response.status_code == 422
    assert "hunter2-secret" not in response.text
    body = response.json()
    assert body["message"] == "Validation failed"
    assert [(e["location"], e["field"]) for e in body["errors"]] == [("body", "email")]


async def test_unhandled_errors_hide_the_details() -> None:
    async with client() as c:
        response = await c.get("/crash")
    assert response.status_code == 500
    assert response.json() == {"message": "Internal server error"}
