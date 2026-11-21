"""Read-only HR views: the audit log and questions policy search couldn't
answer. Newest first, paged with the same `before` cursor as chat: IDs are
UUIDv7, so they sort by creation time.

Dates are whole days in the company's timezone, both ends included.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import (
    AuditAction,
    AuditLogModel,
    AuditOutcome,
    EmployeeModel,
    Role,
    UnansweredQueryModel,
    UnansweredReason,
)
from warden.scripts import company as co
from warden.services.access import ensure_role
from warden.services.errors import InvalidInput

MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class AuditEntry:
    log: AuditLogModel
    # The actor as they are now. None for anonymous events (failed logins).
    actor_name: str | None
    actor_email: str | None


@dataclass(frozen=True)
class Page[T]:
    items: Sequence[T]
    # Pass as `before` for the next page. None on the last one.
    next: uuid.UUID | None


@dataclass(frozen=True)
class AuditFilters:
    actor_id: uuid.UUID | None = None
    action: AuditAction | None = None
    outcome: AuditOutcome | None = None
    target_id: uuid.UUID | None = None
    from_date: date | None = None
    to_date: date | None = None


class AdminService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_audit_logs(
        self,
        user: CurrentUser,
        filters: AuditFilters,
        before: uuid.UUID | None = None,
        limit: int = 50,
    ) -> Page[AuditEntry]:
        await ensure_role(self.db, user, Role.HR, operation="list_audit_logs")
        query = select(AuditLogModel, EmployeeModel.name, EmployeeModel.email).outerjoin(
            EmployeeModel, EmployeeModel.id == AuditLogModel.actor_id
        )
        if filters.actor_id is not None:
            query = query.where(AuditLogModel.actor_id == filters.actor_id)
        if filters.action is not None:
            query = query.where(AuditLogModel.action == filters.action.value)
        if filters.outcome is not None:
            query = query.where(AuditLogModel.outcome == filters.outcome)
        if filters.target_id is not None:
            query = query.where(AuditLogModel.target_id == filters.target_id)
        query = _in_dates(query, AuditLogModel.created_at, filters.from_date, filters.to_date)
        if before is not None:
            query = query.where(AuditLogModel.id < before)

        limit = _clamp(limit)
        rows = (
            await self.db.execute(query.order_by(AuditLogModel.id.desc()).limit(limit + 1))
        ).all()
        entries = [AuditEntry(log, name, email) for log, name, email in rows]
        return _page(entries, limit, key=lambda entry: entry.log.id)

    async def list_unanswered_queries(
        self,
        user: CurrentUser,
        *,
        reason: UnansweredReason | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        before: uuid.UUID | None = None,
        limit: int = 50,
    ) -> Page[UnansweredQueryModel]:
        """Questions only, not who asked them: finding gaps in the documents
        doesn't need to know which employee wondered about what."""
        await ensure_role(self.db, user, Role.HR, operation="list_unanswered_queries")
        query = select(UnansweredQueryModel)
        if reason is not None:
            query = query.where(UnansweredQueryModel.reason == reason)
        query = _in_dates(query, UnansweredQueryModel.created_at, from_date, to_date)
        if before is not None:
            query = query.where(UnansweredQueryModel.id < before)

        limit = _clamp(limit)
        rows = (
            await self.db.scalars(query.order_by(UnansweredQueryModel.id.desc()).limit(limit + 1))
        ).all()
        return _page(rows, limit, key=lambda row: row.id)


def _in_dates[Q: Select[*tuple[Any, ...]]](
    query: Q, column: Any, from_date: date | None, to_date: date | None
) -> Q:
    if from_date is not None and to_date is not None and from_date > to_date:
        raise InvalidInput("from_date is after to_date")
    if from_date is not None:
        query = query.where(column >= _start_of(from_date))
    if to_date is not None:
        query = query.where(column < _start_of(to_date + timedelta(days=1)))
    return query


def _start_of(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=co.TIMEZONE)


def _clamp(limit: int) -> int:
    return max(1, min(limit, MAX_PAGE_SIZE))


def _page[T](rows: Sequence[T], limit: int, *, key: Any) -> Page[T]:
    """`rows` was fetched with limit + 1: an extra row means there's more."""
    if len(rows) <= limit:
        return Page(rows, None)
    items = rows[:limit]
    return Page(items, key(items[-1]))
