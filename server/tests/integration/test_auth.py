import pytest
from httpx import AsyncClient, Response
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import PASSWORD, MakeEmployee
from warden.models import EmployeeModel


async def login(client: AsyncClient, employee: EmployeeModel) -> Response:
    response = await client.post(
        "/api/auth/login", json={"email": employee.email, "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    return response


async def send_with_cookie(client: AsyncClient, method: str, url: str, cookie: str) -> Response:
    """Send exactly one cookie, ignoring whatever the client has stored."""
    client.cookies.clear()
    return await client.request(method, url, headers={"Cookie": cookie})


async def test_login_sets_http_only_cookies(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    employee = await make_employee(email="asha@warden.demo")

    # Email matching ignores case.
    response = await client.post(
        "/api/auth/login", json={"email": "Asha@Warden.Demo", "password": PASSWORD}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(employee.id)
    assert "password_hash" not in body

    set_cookies = response.headers.get_list("set-cookie")
    access = next(c for c in set_cookies if c.startswith("access_token="))
    refresh = next(c for c in set_cookies if c.startswith("refresh_token="))
    assert "HttpOnly" in access and "Path=/;" in access
    assert "HttpOnly" in refresh and "Path=/api/auth" in refresh

    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["id"] == str(employee.id)


@pytest.mark.parametrize(
    ("email", "password", "is_active"),
    [
        ("employee1@warden.demo", "wrong-password", True),
        ("nobody@warden.demo", PASSWORD, True),
        ("employee1@warden.demo", PASSWORD, False),
    ],
    ids=["wrong-password", "unknown-email", "inactive-employee"],
)
async def test_login_failures_look_the_same(
    client: AsyncClient,
    make_employee: MakeEmployee,
    email: str,
    password: str,
    is_active: bool,
) -> None:
    await make_employee(email="employee1@warden.demo", is_active=is_active)

    response = await client.post("/api/auth/login", json={"email": email, "password": password})

    assert response.status_code == 401
    assert response.json() == {"message": "Invalid email or password"}
    assert "set-cookie" not in response.headers


async def test_me_requires_a_valid_access_token(client: AsyncClient) -> None:
    assert (await client.get("/api/auth/me")).status_code == 401
    response = await send_with_cookie(client, "GET", "/api/auth/me", "access_token=garbage")
    assert response.status_code == 401


async def test_deactivated_employee_is_locked_out_immediately(
    client: AsyncClient, db: AsyncSession, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    await login(client, employee)

    employee.is_active = False
    await db.commit()

    assert (await client.get("/api/auth/me")).status_code == 401
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_refresh_rotates_both_tokens(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    login_response = await login(client, employee)

    response = await client.post("/api/auth/refresh")

    assert response.status_code == 200
    assert response.json()["id"] == str(employee.id)
    for name in ("access_token", "refresh_token"):
        assert response.cookies[name] != login_response.cookies[name]
    assert (await client.get("/api/auth/me")).status_code == 200


async def test_refresh_token_reuse_revokes_every_session(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    employee = await make_employee()
    first = await login(client, employee)
    old_refresh = first.cookies["refresh_token"]
    rotated = await client.post("/api/auth/refresh")
    other_device = await login(client, employee)

    replay = await send_with_cookie(
        client, "POST", "/api/auth/refresh", f"refresh_token={old_refresh}"
    )
    assert replay.status_code == 401

    # Both the rotated session and the employee's other session are now dead.
    for response in (rotated, other_device):
        me = await send_with_cookie(
            client, "GET", "/api/auth/me", f"access_token={response.cookies['access_token']}"
        )
        assert me.status_code == 401
        refresh = await send_with_cookie(
            client,
            "POST",
            "/api/auth/refresh",
            f"refresh_token={response.cookies['refresh_token']}",
        )
        assert refresh.status_code == 401


async def test_access_and_refresh_tokens_are_not_interchangeable(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    tokens = (await login(client, await make_employee())).cookies

    me = await send_with_cookie(
        client, "GET", "/api/auth/me", f"access_token={tokens['refresh_token']}"
    )
    refresh = await send_with_cookie(
        client, "POST", "/api/auth/refresh", f"refresh_token={tokens['access_token']}"
    )

    assert me.status_code == 401
    assert refresh.status_code == 401


async def test_logout_revokes_the_session_and_clears_cookies(
    client: AsyncClient, make_employee: MakeEmployee
) -> None:
    tokens = (await login(client, await make_employee())).cookies

    response = await client.post("/api/auth/logout")

    assert response.status_code == 204
    cleared = response.headers.get_list("set-cookie")
    assert any(c.startswith('access_token="";') and "Max-Age=0" in c for c in cleared)
    assert any(c.startswith('refresh_token="";') and "Max-Age=0" in c for c in cleared)

    # The old tokens stop working right away, not when they expire.
    me = await send_with_cookie(
        client, "GET", "/api/auth/me", f"access_token={tokens['access_token']}"
    )
    refresh = await send_with_cookie(
        client, "POST", "/api/auth/refresh", f"refresh_token={tokens['refresh_token']}"
    )
    assert me.status_code == 401
    assert refresh.status_code == 401
