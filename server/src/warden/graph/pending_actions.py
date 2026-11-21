"""Pending actions: writes the agent proposed and the employee hasn't
confirmed yet.

A write tool never changes anything. It checks the rules, then `create`s a
pending action and returns its summary. The change happens only when the
employee clicks Confirm, which calls an endpoint that runs `confirm`: it
checks the action is theirs, still pending and not expired, then runs the
service call, which checks every rule again. The model has no tool that
confirms, so it can't approve its own proposal however it's prompted.

Statuses: pending until the user clicks; then confirmed (the change was
made), failed (confirmed, but the service refused it), cancelled, or
expired.

Every lookup is scoped by the employee, so someone else's action is
NotFound, the same as one that doesn't exist.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import (
    AuditAction,
    AuditOutcome,
    AuditTarget,
    LeaveRequestModel,
    LeaveType,
    PendingActionModel,
    PendingActionStatus,
    PendingActionType,
)
from warden.services import audit
from warden.services.errors import ActionExpired, InvalidStatus, NotFound, ServiceError
from warden.services.leave import LeaveService

# Long enough to read the card, short enough that the balance it shows is
# still close to true. Confirm re-runs the rules anyway.
EXPIRY = timedelta(minutes=10)


def utc_now() -> datetime:
    return datetime.now(UTC)


def visible_status(action: PendingActionModel, now: datetime) -> PendingActionStatus:
    """The status to show on the card. A pending action past its expiry is
    expired, though the row only says so once someone clicks it (`_claim`
    marks it then)."""
    if action.status == PendingActionStatus.PENDING and action.expires_at <= now:
        return PendingActionStatus.EXPIRED
    return action.status


@dataclass(frozen=True)
class ConfirmOutcome:
    """What happened when the user clicked Confirm."""

    action: PendingActionModel
    # The change was made. False when the service refused it.
    ok: bool
    # For the agent: the request as it is now, or the refusal's code and
    # message.
    result: dict[str, Any]


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

    async def confirm(
        self, user: CurrentUser, action_id: uuid.UUID, leave: LeaveService
    ) -> ConfirmOutcome:
        """Run `user`'s pending action.

        Raises NotFound (not theirs, or no such action), InvalidStatus
        (already confirmed, cancelled, failed or expired) or ActionExpired.
        Otherwise the action is claimed with `mark`, so of two concurrent
        clicks only one runs, and the service call runs with the rules
        checked again: balances may have changed since the card was shown.
        A refusal there isn't raised; it comes back with `ok=False` and the
        action is `failed`, so the agent can explain it.

        Audited as `pending_action.confirm`, on top of the service's own
        audit row for the leave write. A click that lost the race to another
        records nothing: the first one already did.
        """
        audit_action = AuditAction.PENDING_ACTION_CONFIRM
        action = await self._claim(user, action_id, PendingActionStatus.CONFIRMED, audit_action)
        try:
            # Commits the claim along with the change.
            request = await _run(leave, user, action)
        except ServiceError as exc:
            # The service committed its refusal, and the claim with it.
            action.status = PendingActionStatus.FAILED
            result: dict[str, Any] = {"error": exc.code, "message": exc.message} | exc.details
            self._audit(user, action, audit_action, AuditOutcome.DENIED, {"reason": exc.code})
            await self.db.commit()
            return ConfirmOutcome(action, ok=False, result=to_jsonable_python(result))

        result = {
            "leave_request_id": str(request.id),
            "status": request.status.value,
            "days": request.days,
        }
        self._audit(user, action, audit_action, AuditOutcome.ALLOWED, {"result": result})
        await self.db.commit()
        return ConfirmOutcome(action, ok=True, result=result)

    async def cancel(self, user: CurrentUser, action_id: uuid.UUID) -> PendingActionModel:
        """Drop `user`'s pending action without running it. Raises like
        `confirm` when it's not theirs, no longer pending or expired.
        Audited as `pending_action.cancel`."""
        audit_action = AuditAction.PENDING_ACTION_CANCEL
        action = await self._claim(user, action_id, PendingActionStatus.CANCELLED, audit_action)
        self._audit(user, action, audit_action, AuditOutcome.ALLOWED, {})
        await self.db.commit()
        return action

    async def _claim(
        self,
        user: CurrentUser,
        action_id: uuid.UUID,
        status: PendingActionStatus,
        audit_action: AuditAction,
    ) -> PendingActionModel:
        """Move `user`'s action from pending to `status`, without committing,
        or raise. An expired action is marked expired (audited as denied and
        committed) instead. A click that loses the race to another raises
        InvalidStatus and records nothing: the first click already did."""
        action = await self.get_for_user(action_id, user)
        if action.status != PendingActionStatus.PENDING:
            raise InvalidStatus(f"This action is already {action.status.value}.")
        if self.is_expired(action):
            if await self.mark(action.id, user, PendingActionStatus.EXPIRED):
                details = {"reason": ActionExpired.code}
                self._audit(user, action, audit_action, AuditOutcome.DENIED, details)
                await self.db.commit()
            raise ActionExpired("This action expired. Ask again to get a new one.")
        if not await self.mark(action.id, user, status):
            raise InvalidStatus("This action was already handled.")
        return action

    def _audit(
        self,
        user: CurrentUser,
        action: PendingActionModel,
        audit_action: AuditAction,
        outcome: AuditOutcome,
        details: dict[str, Any],
    ) -> None:
        audit.record(
            self.db,
            action=audit_action,
            outcome=outcome,
            actor_id=user.employee_id,
            target_type=AuditTarget.PENDING_ACTION,
            target_id=action.id,
            details={"action_type": action.action_type} | details,
        )


async def _run(
    leave: LeaveService, user: CurrentUser, action: PendingActionModel
) -> LeaveRequestModel:
    """The service call an action stands for, with the params its write tool
    saved. Each call commits."""
    params = action.params
    match PendingActionType(action.action_type):
        case PendingActionType.APPLY_LEAVE:
            return await leave.apply(
                user,
                LeaveType(params["leave_type"]),
                date.fromisoformat(params["start"]),
                date.fromisoformat(params["end"]),
                params.get("reason"),
            )
        case PendingActionType.CANCEL_LEAVE:
            return await leave.cancel(user, uuid.UUID(params["request_id"]))
        case PendingActionType.APPROVE_LEAVE:
            return await leave.approve(user, uuid.UUID(params["request_id"]), params.get("comment"))
        case PendingActionType.REJECT_LEAVE:
            return await leave.reject(user, uuid.UUID(params["request_id"]), params["comment"])
