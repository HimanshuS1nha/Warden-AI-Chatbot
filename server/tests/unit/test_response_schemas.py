"""No response the API can send has a field that could hold a password or a
password hash. Scans the OpenAPI schema, so a new route is covered as soon
as it exists."""

from typing import Any

from warden.main import app

FORBIDDEN_PARTS = ("password", "hash", "secret", "token")


def _response_schema_names(openapi: dict[str, Any]) -> set[str]:
    """Every component schema reachable from a response body."""
    components = openapi["components"]["schemas"]
    pending = [
        media["schema"]
        for path in openapi["paths"].values()
        for operation in path.values()
        for response in operation.get("responses", {}).values()
        for media in response.get("content", {}).values()
    ]
    names: set[str] = set()
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name not in names:
                    names.add(name)
                    pending.append(components[name])
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return names


def _secret_fields(openapi: dict[str, Any]) -> list[str]:
    schemas = openapi["components"]["schemas"]
    return [
        f"{name}.{field}"
        for name in sorted(_response_schema_names(openapi))
        for field in schemas[name].get("properties", {})
        if any(part in field.lower() for part in FORBIDDEN_PARTS)
    ]


def test_no_response_schema_has_a_secret_field() -> None:
    openapi = app.openapi()
    # Sanity check that the scan reaches the schemas it's meant to guard.
    assert {"EmployeeResponse", "ProfileResponse", "TeamLeaveRequestResponse"} <= (
        _response_schema_names(openapi)
    )

    assert _secret_fields(openapi) == []


def test_the_scan_would_catch_one() -> None:
    fake = {
        "paths": {"/x": {"get": {"responses": {"200": {"content": {"application/json": {
            "schema": {"type": "array", "items": {"$ref": "#/components/schemas/Outer"}}
        }}}}}}},
        "components": {"schemas": {
            "Outer": {"properties": {"inner": {"$ref": "#/components/schemas/Inner"}}},
            "Inner": {"properties": {"password_hash": {"type": "string"}}},
        }},
    }  # fmt: skip

    assert _secret_fields(fake) == ["Inner.password_hash"]
