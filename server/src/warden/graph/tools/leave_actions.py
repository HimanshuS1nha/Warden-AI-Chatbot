"""Write tools for the signed-in employee's own leave. Every role gets them.

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
from warden.services.leave import MAX_REASON_LENGTH, LeaveQuote

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

    return [apply_for_leave]


def leave_summary(quote: LeaveQuote, today: date) -> str:
    """The confirmation card's text, e.g. "Casual leave, Mon 23 Nov to Tue
    24 Nov, 2 days. Balance after: 6". Dates outside this year show it."""
    span = _day(quote.start, today)
    if quote.end != quote.start:
        span += f" to {_day(quote.end, today)}"
    days = f"{quote.days} day" + ("" if quote.days == 1 else "s")
    return (
        f"{quote.leave_type.value.capitalize()} leave, {span}, {days}. "
        f"Balance after: {quote.available_after}"
    )


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
