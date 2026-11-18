"""Writes to the append-only audit log.

The log answers "who did what to which record, and was it allowed". It is
written by the backend, never by the agent. Conventions:

- Record anything that changes data, anything a rule or role check refuses,
  and every auth event.
- A successful write is recorded in the same transaction as the change, so a
  rollback takes the audit row with it.
- A refusal is recorded and committed before the error is raised. Otherwise
  the exception discards the row along with the rest of the session.
- Plain reads of your own data and routine token refreshes aren't recorded.
- `details` never holds passwords, tokens, hashes or whole request bodies.
- Responses don't change: a row may say why a login failed, but the client
  still gets the same 401.
"""

import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.request_context import get_request_context
from warden.models import AuditAction, AuditLogModel, AuditOutcome, AuditTarget

MAX_DETAILS_BYTES = 4096

# A key containing any of these is refused, whatever its value. Catching it by
# name is crude, but it fails loudly in tests instead of a secret quietly
# reaching a table that can't be cleaned up.
FORBIDDEN_KEY_PARTS = ("password", "token", "secret", "hash")


def record(
    db: AsyncSession,
    *,
    action: AuditAction,
    outcome: AuditOutcome,
    actor_id: uuid.UUID | None = None,
    target_type: AuditTarget | None = None,
    target_id: uuid.UUID | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLogModel:
    """Add an audit row to `db` without committing.

    The client IP, and the request ID when there is one, come from the
    request context. Raises ValueError when `details` has a forbidden key or
    is larger than MAX_DETAILS_BYTES as JSON.
    """
    context = get_request_context()
    details = dict(details or {})
    if context.request_id is not None:
        details["request_id"] = context.request_id

    entry = AuditLogModel(
        actor_id=actor_id,
        action=action.value,
        outcome=outcome,
        target_type=target_type.value if target_type else None,
        target_id=target_id,
        details=_checked_details(details),
        ip=context.ip,
    )
    db.add(entry)
    return entry


def _checked_details(details: dict[str, Any]) -> dict[str, Any]:
    _refuse_secret_keys(details)
    # default=str turns UUIDs, dates and enums into strings, the same way
    # they'd be written out by hand.
    serialized = json.dumps(details, default=str, separators=(",", ":"))
    size = len(serialized.encode())
    if size > MAX_DETAILS_BYTES:
        raise ValueError(f"audit details are {size} bytes; the limit is {MAX_DETAILS_BYTES}")
    loaded: dict[str, Any] = json.loads(serialized)
    return loaded


def _refuse_secret_keys(value: Any, path: str = "details") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if any(part in name for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"audit {path} can't contain the key {key!r}")
            _refuse_secret_keys(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _refuse_secret_keys(item, f"{path}[{index}]")
