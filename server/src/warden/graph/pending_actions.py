"""Pending actions: writes the agent proposed and the employee hasn't
confirmed yet.

A write tool never changes anything. It checks the rules, then `create`s a
pending action and returns its summary. The change happens only when the
employee clicks Confirm, which calls an endpoint that loads the action with
`get_for_user`, checks it's still pending and not expired, and runs the
service call. The model has no tool that confirms, so it can't approve its
own proposal however it's prompted.

Every lookup is scoped by the employee, so someone else's action is
NotFound, the same as one that doesn't exist.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import PendingActionModel, PendingActionStatus, PendingActionType
from warden.services.errors import NotFound

# Long enough to read the card, short enough that the balance it shows is
# still close to true. Confirm re-runs the rules anyway.
EXPIRY = timedelta(minutes=10)


def utc_now() -> datetime:
    return datetime.now(UTC)


class PendingActions:
    def __init__(self, db: AsyncSession, *, now: Callable[[], datetime] = utc_now) -> None:
        self.db = db
        # Injectable so tests can move past the expiry without waiting.
        self.now = now

    async def create(
        self,
        user: CurrentUser,
        thread_id: uuid.UUID,
        action_type: PendingActionType,
        params: dict[str, Any],
        summary: str,
    ) -> PendingActionModel:
        """Save a proposed write for `user`, expiring in 10 minutes, and
        commit it, so it outlives the turn that proposed it.

        `thread_id` is the turn's thread, which the chat endpoint has already
        checked belongs to `user`. `params` are what the service call needs on
        confirm; dates and UUIDs are stored as ISO strings.
        """
        action = PendingActionModel(
            employee_id=user.employee_id,
            thread_id=thread_id,
            action_type=action_type.value,
            params=to_jsonable_python(params),
            summary=summary,
            expires_at=self.now() + EXPIRY,
        )
        self.db.add(action)
        await self.db.commit()
        return action

    async def get_for_user(self, action_id: uuid.UUID, user: CurrentUser) -> PendingActionModel:
        """The action if it belongs to `user`, in any status. Raises NotFound
        otherwise. Whether it can still be confirmed is the caller's check:
        see `is_expired`."""
        action = await self.db.scalar(
            select(PendingActionModel).where(
                PendingActionModel.id == action_id,
                PendingActionModel.employee_id == user.employee_id,
            )
        )
        if action is None:
            raise NotFound("Action not found.")
        return action

    def is_expired(self, action: PendingActionModel) -> bool:
        return action.expires_at <= self.now()

    async def mark(
        self, action_id: uuid.UUID, user: CurrentUser, status: PendingActionStatus
    ) -> bool:
        """Move `user`'s action from pending to `status`. Returns False if it
        wasn't pending (or isn't theirs), so of two concurrent confirms only
        one gets True and runs the write.

        Doesn't commit: the caller commits it with the write it guards, so a
        failed write leaves the action pending.
        """
        if status == PendingActionStatus.PENDING:
            raise ValueError("An action can't be marked pending again.")
        result = await self.db.execute(
            update(PendingActionModel)
            .where(
                PendingActionModel.id == action_id,
                PendingActionModel.employee_id == user.employee_id,
                PendingActionModel.status == PendingActionStatus.PENDING,
            )
            .values(status=status)
        )
        # An UPDATE result always has a rowcount; mypy sees the generic Result.
        return bool(result.rowcount)  # type: ignore[attr-defined]
