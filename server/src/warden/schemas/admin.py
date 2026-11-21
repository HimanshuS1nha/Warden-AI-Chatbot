import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from warden.models.enums import AuditOutcome, UnansweredReason
from warden.services.admin import AuditEntry


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    actor_id: uuid.UUID | None
    actor_name: str | None
    actor_email: str | None
    action: str
    outcome: AuditOutcome
    target_type: str | None
    target_id: uuid.UUID | None
    details: dict[str, Any]
    ip: str | None

    @classmethod
    def build(cls, entry: AuditEntry) -> "AuditLogResponse":
        log = entry.log
        return cls(
            id=log.id,
            created_at=log.created_at,
            actor_id=log.actor_id,
            actor_name=entry.actor_name,
            actor_email=entry.actor_email,
            action=log.action,
            outcome=log.outcome,
            target_type=log.target_type,
            target_id=log.target_id,
            details=log.details,
            ip=str(log.ip) if log.ip is not None else None,
        )


class AuditLogPage(BaseModel):
    entries: list[AuditLogResponse]
    # Pass as `before` for the next page; null on the last one.
    next: uuid.UUID | None


class UnansweredQueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    query: str
    reason: UnansweredReason


class UnansweredQueryPage(BaseModel):
    queries: list[UnansweredQueryResponse]
    next: uuid.UUID | None
