import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from tests.factories import MakeEmployee, as_user
from tests.integration.conftest import wait_for_lock_waiters
from tests.integration.test_leave_service import TODAY, add_balance, add_request, days
from warden.graph.pending_actions import EXPIRY, ConfirmOutcome, PendingActions
from warden.models import (
    AuditLogModel,
    AuditOutcome,
    ChatThreadModel,
    EmployeeModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    PendingActionModel,
    PendingActionStatus,
    PendingActionType,
    Role,
)
from warden.services.errors import InvalidStatus, NotFound
from warden.services.leave import LeaveService

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


# Confirming. TODAY is Monday 1 June 2026.


async def new_action(
    db: AsyncSession,
    owner: EmployeeModel,
    action_type: PendingActionType,
    params: dict[str, Any],
) -> PendingActionModel:
    thread = ChatThreadModel(employee_id=owner.id)
    db.add(thread)
    await db.flush()
    return await PendingActions(db).create(as_user(owner), thread.id, action_type, params, "...")


async def confirm(db: AsyncSession, owner: EmployeeModel, action_id: uuid.UUID) -> ConfirmOutcome:
    leave = LeaveService(db, today=lambda: TODAY)
    return await PendingActions(db).confirm(as_user(owner), action_id, leave)


async def confirm_audits(db: AsyncSession) -> list[tuple[AuditOutcome, Any]]:
    rows = await db.scalars(
        select(AuditLogModel)
        .where(AuditLogModel.action == "pending_action.confirm")
        .order_by(AuditLogModel.id)
    )
    return [(r.outcome, r.details.get("reason")) for r in rows]


async def test_confirm_catches_a_balance_that_changed_since_the_card(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    me = await make_employee()
    balance = await add_balance(db, me, allotted=2)
    action = await new_action(
        db,
        me,
        PendingActionType.APPLY_LEAVE,
        {"leave_type": "casual", "start": TODAY + days(7), "end": TODAY + days(8)},
    )
    # The card said 2 days fit. Then a day was used elsewhere.
    balance.used = 1
    await db.flush()

    outcome = await confirm(db, me, action.id)

    assert not outcome.ok
    assert outcome.result["error"] == "insufficient_balance"
    assert (outcome.result["requested"], outcome.result["available"]) == (2, 1)
    assert outcome.action.status == PendingActionStatus.FAILED
    assert await db.scalar(select(func.count()).select_from(LeaveRequestModel)) == 0
    assert await confirm_audits(db) == [(AuditOutcome.DENIED, "insufficient_balance")]


async def test_confirm_catches_a_reports_balance_that_changed_since_the_card(
    db: AsyncSession, make_employee: MakeEmployee
) -> None:
    manager = await make_employee(role=Role.MANAGER)
    report = await make_employee(manager_id=manager.id)
    balance = await add_balance(db, report, allotted=3)
    request = await add_request(db, report, TODAY + days(7), TODAY + days(8))
    action = await new_action(
        db, manager, PendingActionType.APPROVE_LEAVE, {"request_id": request.id, "comment": None}
    )
    balance.used = 2  # Another approval landed first.
    await db.flush()

    outcome = await confirm(db, manager, action.id)

    assert (outcome.ok, outcome.result["error"]) == (False, "insufficient_balance")
    assert outcome.action.status == PendingActionStatus.FAILED
    await db.refresh(request)
    await db.refresh(balance)
    assert (request.status, balance.used) == (LeaveStatus.PENDING, 2)


async def test_concurrent_confirms_run_the_change_once(
    engine: AsyncEngine, committed_team: tuple[EmployeeModel, EmployeeModel]
) -> None:
    _, me = committed_team
    async with AsyncSession(engine, expire_on_commit=False) as db:
        await add_balance(db, me, allotted=10)
        action = await new_action(
            db,
            me,
            PendingActionType.APPLY_LEAVE,
            {"leave_type": "casual", "start": TODAY + days(7), "end": TODAY + days(8)},
        )

    async def click() -> str:
        async with AsyncSession(engine, expire_on_commit=False) as db:
            try:
                outcome = await confirm(db, me, action.id)
            except InvalidStatus:
                return "already handled"
            return "confirmed" if outcome.ok else "failed"

    # Hold the action's row so both clicks have read it as pending and are
    # waiting on the claim's UPDATE. Releasing it lets them race for real.
    async with engine.connect() as blocker:
        await blocker.begin()
        await blocker.execute(
            select(PendingActionModel).where(PendingActionModel.id == action.id).with_for_update()
        )
        tasks = [asyncio.create_task(click()) for _ in range(2)]
        await wait_for_lock_waiters(engine, 2)
        await blocker.rollback()
    results = await asyncio.gather(*tasks)

    assert sorted(results) == ["already handled", "confirmed"]
    async with AsyncSession(engine) as db:
        requests = await db.scalars(
            select(LeaveRequestModel).where(LeaveRequestModel.employee_id == me.id)
        )
        assert [r.days for r in requests] == [2]
        assert await confirm_audits(db) == [(AuditOutcome.ALLOWED, None)]
        saved = await db.get(PendingActionModel, action.id)
        assert saved is not None and saved.status == PendingActionStatus.CONFIRMED
        used = await db.scalar(
            select(LeaveBalanceModel.used).where(LeaveBalanceModel.employee_id == me.id)
        )
        assert used == 0  # Applying doesn't use days; approval does.
