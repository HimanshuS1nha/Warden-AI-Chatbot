import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import MakeEmployee, as_user
from warden.graph.pending_actions import EXPIRY, PendingActions
from warden.models import (
    ChatThreadModel,
    EmployeeModel,
    PendingActionModel,
    PendingActionStatus,
    PendingActionType,
)
from warden.services.errors import NotFound

NOW = datetime(2026, 11, 20, 9, 30, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def propose(
    db: AsyncSession, actions: PendingActions, owner: EmployeeModel
) -> PendingActionModel:
    thread = ChatThreadModel(employee_id=owner.id)
    db.add(thread)
    await db.flush()
    return await actions.create(
        as_user(owner),
        thread.id,
        PendingActionType.APPLY_LEAVE,
        {"leave_type": "casual", "start": date(2026, 11, 23), "end": date(2026, 11, 24)},
        "Casual leave, Mon 23 Nov to Tue 24 Nov, 2 days. Balance after: 6",
    )


async def test_create_saves_a_pending_action_that_expires_in_ten_minutes(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    actions = PendingActions(db, now=Clock(NOW))

    action_id, user = (await propose(db, actions, me)).id, as_user(me)

    db.expire_all()
    saved = await actions.get_for_user(action_id, user)
    assert saved.status == PendingActionStatus.PENDING
    assert saved.action_type == "apply_leave"
    assert saved.expires_at == NOW + EXPIRY
    assert saved.params == {"leave_type": "casual", "start": "2026-11-23", "end": "2026-11-24"}


async def test_someone_elses_action_is_not_found(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me, other = await make_employee(), await make_employee()
    actions = PendingActions(db)
    action = await propose(db, actions, me)

    with pytest.raises(NotFound):
        await actions.get_for_user(action.id, as_user(other))
    with pytest.raises(NotFound):
        await actions.get_for_user(uuid.uuid4(), as_user(me))
    assert not await actions.mark(action.id, as_user(other), PendingActionStatus.CONFIRMED)


async def test_an_action_expires_after_ten_minutes(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    clock = Clock(NOW)
    actions = PendingActions(db, now=clock)
    action = await propose(db, actions, me)

    clock.now = NOW + EXPIRY - timedelta(seconds=1)
    assert not actions.is_expired(action)
    clock.now = NOW + EXPIRY
    assert actions.is_expired(action)


async def test_mark_moves_a_pending_action_only_once(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    actions = PendingActions(db)
    action = await propose(db, actions, me)

    assert await actions.mark(action.id, as_user(me), PendingActionStatus.CONFIRMED)
    assert not await actions.mark(action.id, as_user(me), PendingActionStatus.CONFIRMED)
    assert not await actions.mark(action.id, as_user(me), PendingActionStatus.CANCELLED)

    action_id, user = action.id, as_user(me)
    db.expire_all()
    saved = await actions.get_for_user(action_id, user)
    assert saved.status == PendingActionStatus.CONFIRMED


async def test_mark_refuses_to_reopen_an_action(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    actions = PendingActions(db)
    action = await propose(db, actions, me)

    with pytest.raises(ValueError, match="pending again"):
        await actions.mark(action.id, as_user(me), PendingActionStatus.PENDING)
