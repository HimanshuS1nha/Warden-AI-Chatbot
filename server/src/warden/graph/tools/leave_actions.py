"""Write tools for leave. Every role gets the ones for their own leave;
managers also get the ones that decide their reports' requests.

None of them changes anything. Each runs the same rule checks as the write
it proposes, then saves a pending action and returns its summary; the write
runs only when the user clicks Confirm on the card. A broken rule raises,
and the tools node hands the error to the model to explain.

The model reads the summary. The pending action's ID goes in the tool
message's artifact, which the model never sees, for the stream to put on
the confirmation card.
"""

import json
import uuid
from datetime import date
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from warden.core.dependencies import CurrentUser
from warden.graph.tools.services import ToolServices
from warden.models import LeaveType, PendingActionModel, PendingActionType
from warden.services.leave import (
    MAX_COMMENT_LENGTH,
    MAX_REASON_LENGTH,
    CancelQuote,
    DecisionQuote,
    LeaveQuote,
)

AWAITING_CONFIRMATION = (
    "Nothing has changed yet. The user sees a card with this summary and must "
    "click Confirm to go ahead. Tell them to check the card and confirm; don't "
    "say it's done."
)


def build_leave_action_tools(
    user: CurrentUser, services: ToolServices, thread_id: uuid.UUID
) -> list[BaseTool]:
    """`thread_id` is the turn's thread, which every proposal is saved under."""

    @tool(response_format="content_and_artifact")
    async def apply_for_leave(
        leave_type: LeaveType,
        start_date: Annotated[date, Field(description="First day of leave, YYYY-MM-DD.")],
        end_date: Annotated[
            date, Field(description="Last day of leave, YYYY-MM-DD. Same as start for one day.")
        ],
        reason: Annotated[
            str | None,
            Field(max_length=MAX_REASON_LENGTH, description="The user's reason, if they gave one."),
        ] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Propose a leave application for the signed-in employee. It checks the
        dates, overlaps and balance, and shows the user a card to confirm; the
        leave is applied only when they click Confirm. Working days and the
        balance are counted for you: never work them out yourself. Ask the user
        for the leave type and dates if they haven't said."""
        quote = await services.leave.validate_application(user, leave_type, start_date, end_date)
        action = await services.pending_actions.create(
            user,
            thread_id,
            PendingActionType.APPLY_LEAVE,
            {
                "leave_type": quote.leave_type,
                "start": quote.start,
                "end": quote.end,
                "reason": (reason or "").strip() or None,
            },
            leave_summary(quote, services.today()),
        )
        return proposal(action, {"days": quote.days, "available_after": quote.available_after})

    @tool(response_format="content_and_artifact")
    async def cancel_my_leave_request(
        request_id: Annotated[
            uuid.UUID,
            Field(description="The request's `id` from get_my_leave_history."),
        ],
    ) -> tuple[str, dict[str, Any]]:
        """Propose cancelling one of the signed-in employee's own leave requests:
        a pending one, or an approved one that hasn't started. Shows the user a
        card to confirm; the request is cancelled only when they click Confirm.
        Find the request's `id` with get_my_leave_history first, and ask which
        one if several could match."""
        quote = await services.leave.validate_cancel(user, request_id)
        action = await services.pending_actions.create(
            user,
            thread_id,
            PendingActionType.CANCEL_LEAVE,
            {"request_id": quote.request.id},
            cancel_summary(quote, services.today()),
        )
        return proposal(
            action, {"days": quote.request.days, "available_after": quote.available_after}
        )

    return [apply_for_leave, cancel_my_leave_request]


def build_team_leave_action_tools(
    user: CurrentUser, services: ToolServices, thread_id: uuid.UUID
) -> list[BaseTool]:
    """Managers only. The service checks the role and that the request is
    from one of the manager's direct reports on every call anyway."""

    @tool(response_format="content_and_artifact")
    async def approve_team_leave(
        request_id: Annotated[
            uuid.UUID,
            Field(description="The request's `id` from get_team_leave_requests."),
        ],
        comment: Annotated[
            str | None,
            Field(
                max_length=MAX_COMMENT_LENGTH,
                description="A note for the employee, if the manager gave one.",
            ),
        ] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Propose approving a pending leave request from one of the signed-in
        manager's direct reports. It checks the request and the report's
        balance, and shows the manager a card to confirm; the leave is approved
        only when they click Confirm. Find the request's `id` with
        get_team_leave_requests first, and ask which one if several could
        match."""
        quote = await services.leave.validate_approve(user, request_id, comment)
        action = await services.pending_actions.create(
            user,
            thread_id,
            PendingActionType.APPROVE_LEAVE,
            {"request_id": quote.request.id, "comment": quote.comment},
            approve_summary(quote, services.today()),
        )
        return proposal(
            action,
            {
                "employee": quote.employee_name,
                "days": quote.request.days,
                "available_after": quote.available_after,
            },
        )

    @tool(response_format="content_and_artifact")
    async def reject_team_leave(
        request_id: Annotated[
            uuid.UUID,
            Field(description="The request's `id` from get_team_leave_requests."),
        ],
        comment: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_COMMENT_LENGTH,
                description="Why it's rejected, shown to the employee. Required: ask the "
                "manager for it if they haven't said. Never make one up.",
            ),
        ],
    ) -> tuple[str, dict[str, Any]]:
        """Propose rejecting a pending leave request from one of the signed-in
        manager's direct reports, with the manager's reason. Shows the manager
        a card to confirm; the leave is rejected only when they click Confirm.
        Find the request's `id` with get_team_leave_requests first, and ask
        which one if several could match."""
        quote = await services.leave.validate_reject(user, request_id, comment)
        action = await services.pending_actions.create(
            user,
            thread_id,
            PendingActionType.REJECT_LEAVE,
            {"request_id": quote.request.id, "comment": quote.comment},
            reject_summary(quote, services.today()),
        )
        return proposal(action, {"employee": quote.employee_name, "days": quote.request.days})

    return [approve_team_leave, reject_team_leave]


def leave_summary(quote: LeaveQuote, today: date) -> str:
    """The card for an application, e.g. "Casual leave, Mon 23 Nov to Tue
    24 Nov, 2 days. Balance after: 6"."""
    leave = _leave(quote.leave_type, quote.start, quote.end, quote.days, today)
    return f"{leave[0].upper()}{leave[1:]}. Balance after: {quote.available_after}"


def cancel_summary(quote: CancelQuote, today: date) -> str:
    """The card for a cancellation, e.g. "Cancel approved casual leave, Mon
    23 Nov to Tue 24 Nov, 2 days. Balance after: 8"."""
    r = quote.request
    leave = _leave(r.leave_type, r.start_date, r.end_date, r.days, today)
    return f"Cancel {r.status.value} {leave}. Balance after: {quote.available_after}"


def approve_summary(quote: DecisionQuote, today: date) -> str:
    """The card for an approval, e.g. "Approve Asha Rao's casual leave, Mon
    23 Nov to Tue 24 Nov, 2 days. Their balance after: 4. Comment: Enjoy"."""
    r = quote.request
    leave = _leave(r.leave_type, r.start_date, r.end_date, r.days, today)
    summary = (
        f"Approve {quote.employee_name}'s {leave}. Their balance after: {quote.available_after}"
    )
    return summary + (f". Comment: {quote.comment}" if quote.comment else "")


def reject_summary(quote: DecisionQuote, today: date) -> str:
    """The card for a rejection, e.g. "Reject Asha Rao's casual leave, Mon
    23 Nov to Tue 24 Nov, 2 days. Reason: Release week"."""
    r = quote.request
    leave = _leave(r.leave_type, r.start_date, r.end_date, r.days, today)
    return f"Reject {quote.employee_name}'s {leave}. Reason: {quote.comment}"


def _leave(leave_type: LeaveType, start: date, end: date, days: int, today: date) -> str:
    """E.g. "casual leave, Mon 23 Nov to Tue 24 Nov, 2 days". Dates outside
    this year show it."""
    span = _day(start, today) + (f" to {_day(end, today)}" if end != start else "")
    return f"{leave_type.value} leave, {span}, {days} day" + ("" if days == 1 else "s")


def _day(day: date, today: date) -> str:
    return f"{day:%a %-d %b}" + (f" {day.year}" if day.year != today.year else "")


def proposal(action: PendingActionModel, facts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A write tool's result: what the model reads, and the artifact the
    stream turns into a confirmation card."""
    content = {"status": "awaiting_confirmation", "summary": action.summary} | facts
    content["note"] = AWAITING_CONFIRMATION
    artifact = {
        "pending_action_id": str(action.id),
        "action_type": action.action_type,
        "summary": action.summary,
    }
    return json.dumps(content), artifact
