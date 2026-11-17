"""Working-day arithmetic for leave.

Leave Policy v2, section 4: leave is counted in working days (Monday to
Friday), weekends and company public holidays inside the dates are not
deducted, and a single request must fall within one leave year.
"""

from collections.abc import Container
from datetime import date, timedelta

SATURDAY = 5


class CrossesYearBoundary(ValueError):
    """A request from December into January. The policy asks for two requests."""


def is_working_day(day: date, holidays: Container[date]) -> bool:
    return day.weekday() < SATURDAY and day not in holidays


def count_working_days(start: date, end: date, holidays: Container[date]) -> int:
    """Working days from `start` to `end`, both included.

    Raises ValueError when `end` is before `start`, and CrossesYearBoundary
    when the dates are in different years.
    """
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    if start.year != end.year:
        raise CrossesYearBoundary(
            f"{start} to {end} crosses 31 December. Apply for each year separately."
        )
    return sum(
        is_working_day(start + timedelta(days=offset), holidays)
        for offset in range((end - start).days + 1)
    )
