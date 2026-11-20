"""Leave rules: applying, cancelling, approving and rejecting.

The rules (plan.md, "Business rules for leave"; Leave Policy v2):

- Days are working days; weekends and company holidays don't count.
- Leave starts today or later. Sick leave may be backdated up to 7 days.
- A request can't overlap the employee's pending or approved requests.
- Available = allotted - used - days in pending requests. A request needs
  `days <= available`.
- `used` goes up when a manager approves, with the balance row locked.
- Employees cancel their own pending requests, and approved ones that
  haven't started. Cancelling approved leave gives the days back.
- Managers approve or reject pending requests of their direct reports only.
  Nobody approves their own request, and HR can't approve for a manager.
- HR can read anyone's balances and requests; each read is audited as
  `employee.view`.

Every write and every refusal of a write is audited. Writes commit here, in
the same transaction as their audit row.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, NoReturn

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import (
    AuditAction,
    AuditOutcome,
    AuditTarget,
    EmployeeModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
    Role,
)
from warden.repositories import EmployeeRepository, HolidayRepository, LeaveRepository
from warden.scripts import company as co
from warden.services import audit
from warden.services.access import ensure_role
from warden.services.errors import (
    BackdateNotAllowed,
    InsufficientBalance,
    InvalidDates,
    InvalidInput,
    InvalidStatus,
    NotFound,
    Overlap,
    RuleViolation,
    ServiceError,
)
from warden.services.leave_days import count_working_days

MAX_REASON_LENGTH = 1000
MAX_COMMENT_LENGTH = 1000
DEFAULT_SUMMARY_REQUESTS = 10


def company_today() -> date:
    """Today in the company's timezone, which is when a leave day starts."""
    return datetime.now(co.TIMEZONE).date()


@dataclass(frozen=True)
class LeaveQuote:
    """What a valid application would cost. Shown on the confirmation card."""

    leave_type: LeaveType
    start: date
    end: date
    days: int
    available_before: int
    available_after: int


@dataclass(frozen=True)
class CancelQuote:
    """What cancelling a request would give back. Shown on the confirmation
    card."""

    request: LeaveRequestModel
    # Pending days count against the balance too, so cancelling either a
    # pending or an approved request frees its days.
    available_after: int


@dataclass(frozen=True)
class DecisionQuote:
    """What approving or rejecting a report's request would do. Shown on the
    manager's confirmation card."""

    request: LeaveRequestModel
    employee_name: str
    # Trimmed, or None if empty.
    comment: str | None
    # The report's days left of this type once it's approved. None for a
    # rejection, which uses no days.
    available_after: int | None = None


@dataclass(frozen=True)
class BalanceSummary:
    leave_type: LeaveType
    year: int
    allotted: int
    used: int
    pending: int
    available: int


@dataclass(frozen=True)
class EmployeeLeaveSummary:
    """One employee's leave at a glance, for HR."""

    # Loaded with its department.
    employee: EmployeeModel
    year: int
    balances: list[BalanceSummary]
    # Latest start date first, any year and status.
    recent_requests: Sequence[LeaveRequestModel]


def check_dates(leave_type: LeaveType, start: date, end: date, today: date) -> None:
    """The date rules that need no database. Raises InvalidDates or
    BackdateNotAllowed."""
    if end < start:
        raise InvalidDates(f"The end date {end} is before the start date {start}.")
    if start.year != end.year:
        raise InvalidDates(f"{start} to {end} crosses 31 December. Apply for each year separately.")
    if start >= today:
        return
    if leave_type != LeaveType.SICK:
        raise BackdateNotAllowed(
            f"{leave_type.value.capitalize()} leave can't start in the past.", earliest=today
        )
    earliest = today - timedelta(days=co.SICK_LEAVE_BACKDATE_DAYS)
    if start < earliest:
        raise BackdateNotAllowed(
            f"Sick leave can be backdated by at most {co.SICK_LEAVE_BACKDATE_DAYS} days, "
            f"so it can't start before {earliest}.",
            earliest=earliest,
        )


def cancel_refusal(request: LeaveRequestModel, today: date) -> InvalidStatus | None:
    """Why the owner can't cancel this request, or None if they can: only
    pending requests, and approved ones that haven't started."""
    if request.status not in (LeaveStatus.PENDING, LeaveStatus.APPROVED):
        return InvalidStatus(f"This request is already {request.status.value}.")
    if request.status == LeaveStatus.APPROVED and request.start_date <= today:
        return InvalidStatus("Approved leave can only be cancelled before it starts.")
    return None


class LeaveService:
    def __init__(self, db: AsyncSession, *, today: Callable[[], date] = company_today) -> None:
        self.db = db
        self.leave = LeaveRepository(db)
        self.holidays = HolidayRepository(db)
        self.employees = EmployeeRepository(db)
        # Injectable so tests don't depend on the real date.
        self.today = today

    # Reads

    async def get_balances(
        self, user: CurrentUser, year: int | None = None
    ) -> list[BalanceSummary]:
        """The user's balances for `year` (default: this year), one per type."""
        return await self._balances(user.employee_id, year or self.today().year)

    async def _balances(self, employee_id: uuid.UUID, year: int) -> list[BalanceSummary]:
        summaries = []
        for balance in await self.leave.get_balances(employee_id, year):
            pending = await self.leave.pending_days(employee_id, year, balance.leave_type)
            summaries.append(
                BalanceSummary(
                    leave_type=balance.leave_type,
                    year=year,
                    allotted=balance.allotted,
                    used=balance.used,
                    pending=pending,
                    available=balance.allotted - balance.used - pending,
                )
            )
        return summaries

    async def list_my_requests(
        self, user: CurrentUser, status: LeaveStatus | None = None, limit: int = 20
    ) -> Sequence[LeaveRequestModel]:
        return await self.leave.list_requests(user.employee_id, status, _clamp_limit(limit))

    async def get_employee_summary(
        self,
        user: CurrentUser,
        employee_id: uuid.UUID,
        year: int | None = None,
        limit: int = DEFAULT_SUMMARY_REQUESTS,
    ) -> EmployeeLeaveSummary:
        """HR only. Any employee's balances for `year` (default: this year)
        and their `limit` most recent requests. Inactive employees too, for
        questions about people who have left.

        Raises NotFound for an unknown ID. Every successful read is audited as
        `employee.view` with the employee as target, like viewing a profile.
        """
        await ensure_role(self.db, user, Role.HR, operation="leave.employee_summary")
        employee = await self.employees.get(employee_id)
        if employee is None:
            raise NotFound("Employee not found.")
        year = year or self.today().year
        summary = EmployeeLeaveSummary(
            employee=employee,
            year=year,
            balances=await self._balances(employee_id, year),
            recent_requests=await self.leave.list_requests(employee_id, None, _clamp_limit(limit)),
        )
        audit.record(
            self.db,
            action=AuditAction.EMPLOYEE_VIEW,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.EMPLOYEE,
            target_id=employee_id,
            details={"view": "leave_summary", "year": year},
        )
        await self.db.commit()
        return summary

    async def list_team_requests(
        self, user: CurrentUser, status: LeaveStatus | None = None, limit: int = 50
    ) -> Sequence[LeaveRequestModel]:
        """Requests from the manager's direct reports, with `employee` loaded."""
        await ensure_role(self.db, user, Role.MANAGER, operation="leave.list_team")
        return await self.leave.list_team_requests(user.employee_id, status, _clamp_limit(limit))

    # Apply

    async def validate_application(
        self, user: CurrentUser, leave_type: LeaveType, start: date, end: date
    ) -> LeaveQuote:
        """Check an application against every rule without saving anything.

        Returns what it would cost, or raises InvalidDates, BackdateNotAllowed,
        Overlap or InsufficientBalance. The agent calls this before showing
        the confirmation card; `apply` calls it again when the user confirms.
        """
        check_dates(leave_type, start, end, self.today())

        holidays = {h.date for h in await self.holidays.list_for_year(start.year)}
        # check_dates already refused the dates count_working_days rejects.
        days = count_working_days(start, end, holidays)
        if days == 0:
            raise InvalidDates(
                f"{start} to {end} has no working days: it's all weekends or holidays."
            )

        overlapping = await self.leave.find_overlapping(user.employee_id, start, end)
        if overlapping:
            clash = overlapping[0]
            raise Overlap(
                f"This overlaps your {clash.status.value} {clash.leave_type.value} leave "
                f"from {clash.start_date} to {clash.end_date}.",
                request_id=clash.id,
            )

        available = await self._available(user.employee_id, start.year, leave_type)
        if days > available:
            raise InsufficientBalance(
                f"This needs {days} working days of {leave_type.value} leave, "
                f"but only {available} are available.",
                requested=days,
                available=available,
            )
        return LeaveQuote(
            leave_type=leave_type,
            start=start,
            end=end,
            days=days,
            available_before=available,
            available_after=available - days,
        )

    async def apply(
        self,
        user: CurrentUser,
        leave_type: LeaveType,
        start: date,
        end: date,
        reason: str | None = None,
    ) -> LeaveRequestModel:
        """Validate again and save a pending request. Commits."""
        details: dict[str, Any] = {"leave_type": leave_type, "start": start, "end": end}
        reason = (reason or "").strip() or None
        try:
            if reason and len(reason) > MAX_REASON_LENGTH:
                raise InvalidInput(f"The reason can be at most {MAX_REASON_LENGTH} characters.")
            # Held until commit, so a second application from the same person
            # waits and then sees this one in its overlap and balance checks.
            await self.leave.lock_employee(user.employee_id)
            quote = await self.validate_application(user, leave_type, start, end)
        except RuleViolation as exc:
            await self._deny(user, AuditAction.LEAVE_APPLY, exc, details=details)

        request = LeaveRequestModel(
            employee_id=user.employee_id,
            leave_type=leave_type,
            start_date=start,
            end_date=end,
            days=quote.days,
            reason=reason,
            status=LeaveStatus.PENDING,
        )
        self.db.add(request)
        await self.db.flush()
        audit.record(
            self.db,
            action=AuditAction.LEAVE_APPLY,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.LEAVE_REQUEST,
            target_id=request.id,
            details=details | {"days": quote.days},
        )
        await self.db.commit()
        return request

    # Cancel

    async def validate_cancel(self, user: CurrentUser, request_id: uuid.UUID) -> CancelQuote:
        """Check that the user may cancel this request, without saving
        anything. Returns the request and the balance after, or raises
        NotFound or InvalidStatus. The agent calls this before showing the
        confirmation card; `cancel` checks again when the user confirms."""
        request = await self.leave.get_request_for_employee(request_id, user.employee_id)
        if request is None:
            raise _not_found()
        refusal = cancel_refusal(request, self.today())
        if refusal is not None:
            raise refusal
        available = await self._available(
            user.employee_id, request.start_date.year, request.leave_type
        )
        return CancelQuote(request=request, available_after=available + request.days)

    async def cancel(self, user: CurrentUser, request_id: uuid.UUID) -> LeaveRequestModel:
        """Cancel one of the user's own requests. Commits."""
        action = AuditAction.LEAVE_CANCEL
        request = await self.leave.get_request_for_employee(
            request_id, user.employee_id, for_update=True
        )
        if request is None:
            await self._deny(user, action, _not_found(), target_id=request_id)

        details = _request_details(request)
        was_approved = request.status == LeaveStatus.APPROVED
        refusal = cancel_refusal(request, self.today())
        if refusal is not None:
            await self._deny(user, action, refusal, target_id=request.id, details=details)

        if was_approved:
            balance = await self._locked_balance(request)
            # Approval used these days from this row, so it exists unless the
            # data was edited by hand. Cancelling still has to work then.
            if balance is not None:
                details |= _balance_change(balance, -request.days)
                balance.used -= request.days
        request.status = LeaveStatus.CANCELLED
        audit.record(
            self.db,
            action=action,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.LEAVE_REQUEST,
            target_id=request.id,
            details=details | {"previous_status": "approved" if was_approved else "pending"},
        )
        await self.db.commit()
        return request

    # Approve and reject

    async def validate_approve(
        self, user: CurrentUser, request_id: uuid.UUID, comment: str | None = None
    ) -> DecisionQuote:
        """Check that the manager may approve this request, without saving or
        locking anything. Raises Forbidden, InvalidInput, NotFound,
        InvalidStatus or InsufficientBalance, like `approve`, but audits only
        a wrong role. The agent calls this before showing the confirmation
        card; `approve` checks again when the manager confirms."""
        quote = await self._quote_decision(user, request_id, comment, AuditAction.LEAVE_APPROVE)
        request = quote.request
        balances = await self.leave.get_balances(request.employee_id, request.start_date.year)
        balance = next((b for b in balances if b.leave_type == request.leave_type), None)
        if balance is None or request.days > balance.allotted - balance.used:
            raise _over_balance(request, balance)
        return replace(quote, available_after=balance.allotted - balance.used - request.days)

    async def approve(
        self, user: CurrentUser, request_id: uuid.UUID, comment: str | None = None
    ) -> LeaveRequestModel:
        """Approve a direct report's pending request and use the days. Commits."""
        action = AuditAction.LEAVE_APPROVE
        request, comment = await self._load_for_decision(user, request_id, comment, action)
        details = _request_details(request)

        # Locked until commit: a second approval drawing on the same balance
        # waits here, then sees this one's `used`.
        balance = await self._locked_balance(request)
        if balance is None or request.days > balance.allotted - balance.used:
            await self._deny(
                user, action, _over_balance(request, balance), target_id=request.id, details=details
            )

        details |= _balance_change(balance, request.days)
        balance.used += request.days
        self._decide(request, user, LeaveStatus.APPROVED, comment)
        audit.record(
            self.db,
            action=action,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.LEAVE_REQUEST,
            target_id=request.id,
            details=details,
        )
        await self.db.commit()
        return request

    async def validate_reject(
        self, user: CurrentUser, request_id: uuid.UUID, comment: str
    ) -> DecisionQuote:
        """Check that the manager may reject this request, without saving or
        locking anything. Raises like `reject`, but audits only a wrong role.
        The agent calls this before showing the confirmation card; `reject`
        checks again when the manager confirms."""
        if not comment or not comment.strip():
            raise _missing_reason()
        return await self._quote_decision(user, request_id, comment, AuditAction.LEAVE_REJECT)

    async def reject(
        self, user: CurrentUser, request_id: uuid.UUID, comment: str
    ) -> LeaveRequestModel:
        """Reject a direct report's pending request. A comment is required so
        the employee knows why. Commits."""
        action = AuditAction.LEAVE_REJECT
        if not comment or not comment.strip():
            await self._deny(user, action, _missing_reason(), target_id=request_id)
        request, cleaned = await self._load_for_decision(user, request_id, comment, action)

        self._decide(request, user, LeaveStatus.REJECTED, cleaned)
        audit.record(
            self.db,
            action=action,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            target_type=AuditTarget.LEAVE_REQUEST,
            target_id=request.id,
            details=_request_details(request),
        )
        await self.db.commit()
        return request

    # Helpers

    async def _load_for_decision(
        self,
        user: CurrentUser,
        request_id: uuid.UUID,
        comment: str | None,
        action: AuditAction,
    ) -> tuple[LeaveRequestModel, str | None]:
        await ensure_role(self.db, user, Role.MANAGER, operation=action.value)
        comment = _clean_comment(comment)
        if comment_error := _comment_refusal(comment):
            await self._deny(user, action, comment_error, target_id=request_id)

        request = await self._report_request(user, request_id, lock=True)
        if request is None:
            await self._deny(user, action, _not_found(), target_id=request_id)
        if status_error := _decision_refusal(request):
            await self._deny(
                user,
                action,
                status_error,
                target_id=request.id,
                details=_request_details(request),
            )
        return request, comment

    async def _quote_decision(
        self, user: CurrentUser, request_id: uuid.UUID, comment: str | None, action: AuditAction
    ) -> DecisionQuote:
        """The checks approve and reject share, without locking or auditing
        (ensure_role audits a wrong role itself)."""
        await ensure_role(self.db, user, Role.MANAGER, operation=action.value)
        comment = _clean_comment(comment)
        if comment_error := _comment_refusal(comment):
            raise comment_error
        request = await self._report_request(user, request_id, lock=False)
        if request is None:
            raise _not_found()
        if status_error := _decision_refusal(request):
            raise status_error
        employee = await self.employees.get(request.employee_id)
        return DecisionQuote(
            request=request,
            employee_name=employee.name if employee else "Unknown",
            comment=comment,
        )

    async def _report_request(
        self, user: CurrentUser, request_id: uuid.UUID, *, lock: bool
    ) -> LeaveRequestModel | None:
        """A request from one of the manager's direct reports, else None.
        `lock` holds the request row until commit."""
        request = await self.leave.get_request_for_manager(
            request_id, user.employee_id, for_update=lock
        )
        # The scoped query already leaves out the manager's own requests (they
        # aren't their own report). The second check keeps self-approval
        # impossible even if that query changes.
        if request is None or request.employee_id == user.employee_id:
            return None
        return request

    def _decide(
        self,
        request: LeaveRequestModel,
        user: CurrentUser,
        status: LeaveStatus,
        comment: str | None,
    ) -> None:
        request.status = status
        request.approver_id = user.employee_id
        request.decision_comment = comment
        request.decided_at = datetime.now(UTC)

    async def _available(self, employee_id: uuid.UUID, year: int, leave_type: LeaveType) -> int:
        balances = await self.leave.get_balances(employee_id, year)
        balance = next((b for b in balances if b.leave_type == leave_type), None)
        if balance is None:
            return 0
        pending = await self.leave.pending_days(employee_id, year, leave_type)
        return balance.allotted - balance.used - pending

    async def _locked_balance(self, request: LeaveRequestModel) -> LeaveBalanceModel | None:
        return await self.leave.get_balance_for_update(
            request.employee_id, request.start_date.year, request.leave_type
        )

    async def _deny(
        self,
        user: CurrentUser,
        action: AuditAction,
        error: ServiceError,
        *,
        target_id: uuid.UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> NoReturn:
        """Record the refusal, commit it, and raise `error`.

        Refusals are checked before any write, so the commit only saves the
        audit row (and releases the locks taken so far).
        """
        audit.record(
            self.db,
            action=action,
            outcome=AuditOutcome.DENIED,
            actor_id=user.employee_id,
            target_type=AuditTarget.LEAVE_REQUEST if target_id else None,
            target_id=target_id,
            details=(details or {}) | {"reason": error.code},
        )
        await self.db.commit()
        raise error


def _not_found() -> NotFound:
    return NotFound("Leave request not found.")


def _missing_reason() -> InvalidInput:
    return InvalidInput("Say why the request is rejected.")


def _clean_comment(comment: str | None) -> str | None:
    return (comment or "").strip() or None


def _comment_refusal(comment: str | None) -> InvalidInput | None:
    if comment and len(comment) > MAX_COMMENT_LENGTH:
        return InvalidInput(f"The comment can be at most {MAX_COMMENT_LENGTH} characters.")
    return None


def _decision_refusal(request: LeaveRequestModel) -> InvalidStatus | None:
    """Managers decide pending requests only."""
    if request.status != LeaveStatus.PENDING:
        return InvalidStatus(f"This request is already {request.status.value}.")
    return None


def _over_balance(
    request: LeaveRequestModel, balance: LeaveBalanceModel | None
) -> InsufficientBalance:
    """Approving uses days from `allotted - used`. Pending days aren't
    subtracted: this request is one of them."""
    remaining = balance.allotted - balance.used if balance else 0
    return InsufficientBalance(
        f"The request needs {request.days} days but only {remaining} are left.",
        requested=request.days,
        available=remaining,
    )


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, 100))


def _request_details(request: LeaveRequestModel) -> dict[str, Any]:
    return {
        "leave_type": request.leave_type,
        "start": request.start_date,
        "end": request.end_date,
        "days": request.days,
    }


def _balance_change(balance: LeaveBalanceModel, used_delta: int) -> dict[str, Any]:
    """`balance_before` and `balance_after` for the audit row: days left
    unused (allotted - used), before and after this change."""
    before = balance.allotted - balance.used
    return {"balance_before": before, "balance_after": before - used_delta}
