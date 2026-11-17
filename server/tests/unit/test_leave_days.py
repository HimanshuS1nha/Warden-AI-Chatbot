from datetime import date

import pytest
from warden.services.leave_days import CrossesYearBoundary, count_working_days

NO_HOLIDAYS: frozenset[date] = frozenset()

# 2026-11-19 is a Thursday.
THURSDAY = date(2026, 11, 19)
FRIDAY = date(2026, 11, 20)
SATURDAY = date(2026, 11, 21)
SUNDAY = date(2026, 11, 22)
MONDAY = date(2026, 11, 23)
TUESDAY = date(2026, 11, 24)


def test_single_working_day() -> None:
    assert count_working_days(THURSDAY, THURSDAY, NO_HOLIDAYS) == 1


def test_single_weekend_day_counts_zero() -> None:
    assert count_working_days(SATURDAY, SATURDAY, NO_HOLIDAYS) == 0
    assert count_working_days(SATURDAY, SUNDAY, NO_HOLIDAYS) == 0


def test_range_skips_weekend() -> None:
    # The policy's own example: Thursday to the following Tuesday is 4 days.
    assert count_working_days(THURSDAY, TUESDAY, NO_HOLIDAYS) == 4


def test_range_skips_holidays() -> None:
    # ...and 3 days when the Monday is a public holiday.
    assert count_working_days(THURSDAY, TUESDAY, {MONDAY}) == 3


def test_holiday_on_a_weekend_is_not_counted_twice() -> None:
    assert count_working_days(THURSDAY, TUESDAY, {SATURDAY}) == 4


def test_single_holiday_counts_zero() -> None:
    assert count_working_days(FRIDAY, FRIDAY, {FRIDAY}) == 0


def test_full_week() -> None:
    assert count_working_days(date(2026, 11, 16), date(2026, 11, 22), NO_HOLIDAYS) == 5


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="before start"):
        count_working_days(FRIDAY, THURSDAY, NO_HOLIDAYS)


def test_range_across_year_boundary_is_rejected() -> None:
    with pytest.raises(CrossesYearBoundary, match="31 December"):
        count_working_days(date(2026, 12, 30), date(2027, 1, 2), NO_HOLIDAYS)


def test_last_and_first_day_of_year_are_fine_separately() -> None:
    # 2026-12-31 is a Thursday, 2027-01-01 a Friday.
    assert count_working_days(date(2026, 12, 28), date(2026, 12, 31), NO_HOLIDAYS) == 4
    assert count_working_days(date(2027, 1, 1), date(2027, 1, 1), NO_HOLIDAYS) == 1
