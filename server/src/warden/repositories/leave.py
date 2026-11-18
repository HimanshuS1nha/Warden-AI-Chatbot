import uuid
from collections.abc import Sequence
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import (
    EmployeeModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
)

# Requests that hold days: pending ones are reserved against the balance,
# approved ones are already in `used`. Both block an overlapping request.
ACTIVE_STATUSES = (LeaveStatus.PENDING, LeaveStatus.APPROVED)

DEFAULT_HISTORY_LIMIT = 20


class LeaveRepository:
    """Leave balances and leave requests."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_balances(self, employee_id: uuid.UUID, year: int) -> Sequence[LeaveBalanceModel]:
        """One row per leave type the employee has for the year."""
        result = await self.db.scalars(
            select(LeaveBalanceModel)
            .where(LeaveBalanceModel.employee_id == employee_id, LeaveBalanceModel.year == year)
            .order_by(LeaveBalanceModel.leave_type)
        )
        return result.all()

    async def get_balance_for_update(
        self, employee_id: uuid.UUID, year: int, leave_type: LeaveType
    ) -> LeaveBalanceModel | None:
        """The balance row, locked until the transaction ends.

        Approvals and cancellations lock it before changing `used`, so two
        concurrent ones apply one after the other instead of both reading
        the same starting value.
        """
        return await self.db.scalar(
            select(LeaveBalanceModel)
            .where(
                LeaveBalanceModel.employee_id == employee_id,
                LeaveBalanceModel.year == year,
                LeaveBalanceModel.leave_type == leave_type,
            )
            .with_for_update()
        )

    async def list_requests(
        self,
        employee_id: uuid.UUID,
        status: LeaveStatus | None = None,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> Sequence[LeaveRequestModel]:
        """The employee's requests, latest start date first."""
        query = (
            select(LeaveRequestModel)
            .where(LeaveRequestModel.employee_id == employee_id)
            .order_by(LeaveRequestModel.start_date.desc(), LeaveRequestModel.id.desc())
            .limit(limit)
        )
        if status is not None:
            query = query.where(LeaveRequestModel.status == status)
        result = await self.db.scalars(query)
        return result.all()

    async def get_request_for_employee(
        self, request_id: uuid.UUID, employee_id: uuid.UUID, *, for_update: bool = False
    ) -> LeaveRequestModel | None:
        """The request if it belongs to the employee, otherwise None.

        Pass `for_update=True` before changing its status, so a second
        cancel running at the same time waits and then sees the new status.
        """
        query = select(LeaveRequestModel).where(
            LeaveRequestModel.id == request_id, LeaveRequestModel.employee_id == employee_id
        )
        if for_update:
            query = query.with_for_update()
        return await self.db.scalar(query)

    async def get_request_for_manager(
        self, request_id: uuid.UUID, manager_id: uuid.UUID, *, for_update: bool = False
    ) -> LeaveRequestModel | None:
        """The request if it was made by one of the manager's direct reports,
        otherwise None.

        The manager's own request isn't in scope (they aren't their own
        report), so this also refuses self-approval. Pass `for_update=True`
        before approving or rejecting; only the request row is locked.
        """
        query = (
            select(LeaveRequestModel)
            .join(EmployeeModel, EmployeeModel.id == LeaveRequestModel.employee_id)
            .where(LeaveRequestModel.id == request_id, EmployeeModel.manager_id == manager_id)
        )
        if for_update:
            query = query.with_for_update(of=LeaveRequestModel)
        return await self.db.scalar(query)

    async def find_overlapping(
        self, employee_id: uuid.UUID, start: date, end: date
    ) -> Sequence[LeaveRequestModel]:
        """Pending or approved requests of the employee that share at least
        one day with `start`..`end` (both included)."""
        result = await self.db.scalars(
            select(LeaveRequestModel)
            .where(
                LeaveRequestModel.employee_id == employee_id,
                LeaveRequestModel.status.in_(ACTIVE_STATUSES),
                LeaveRequestModel.start_date <= end,
                LeaveRequestModel.end_date >= start,
            )
            .order_by(LeaveRequestModel.start_date)
        )
        return result.all()

    async def pending_days(self, employee_id: uuid.UUID, year: int, leave_type: LeaveType) -> int:
        """Days in the employee's pending requests of this type for the year.

        A request never crosses a year boundary, so its start date gives its
        year.
        """
        total = await self.db.scalar(
            select(func.coalesce(func.sum(LeaveRequestModel.days), 0)).where(
                LeaveRequestModel.employee_id == employee_id,
                LeaveRequestModel.leave_type == leave_type,
                LeaveRequestModel.status == LeaveStatus.PENDING,
                LeaveRequestModel.start_date >= date(year, 1, 1),
                LeaveRequestModel.start_date <= date(year, 12, 31),
            )
        )
        return int(total or 0)
