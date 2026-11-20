"""Argument types and formatting shared by the tool modules."""

from datetime import datetime
from typing import Annotated

from pydantic import Field

from warden.scripts import company as co

# Tool metadata key: the tool doesn't use the request's database session, so
# it may run at the same time as other calls in one reply. See ToolCallGuard.
PARALLEL_SAFE = "parallel_safe"
# Tool metadata key: the tool proposes a write (a pending action). One
# proposal per turn, since a reply carries one confirmation card. See
# ToolCallGuard.
PROPOSES_WRITE = "proposes_write"

Year = Annotated[
    int | None,
    Field(ge=2000, le=2100, description="Calendar year, e.g. 2026. Leave out for this year."),
]
Month = Annotated[
    int | None,
    Field(ge=1, le=12, description="Month number, 1 to 12. Leave out for this month."),
]


def company_time(moment: datetime | None) -> str | None:
    """A stored UTC time in the company's timezone, so the model says 09:00
    for a 09:00 check-in instead of the UTC 03:30."""
    return moment.astimezone(co.TIMEZONE).isoformat(timespec="minutes") if moment else None
