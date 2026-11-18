from datetime import date

import pytest

from warden.models import LeaveType
from warden.scripts import company as co


def test_entitlements_match_leave_policy_v2() -> None:
    assert co.LEAVE_ENTITLEMENT == {
        LeaveType.CASUAL: 12,
        LeaveType.SICK: 10,
        LeaveType.EARNED: 18,
    }
    assert co.EARNED_LEAVE_CARRY_FORWARD_LIMIT == 8
    assert co.SICK_LEAVE_BACKDATE_DAYS == 7


def test_prorating_matches_the_policy_example() -> None:
    # Leave Policy 3.1: joining on 1 July gives 6 casual, 5 sick, 9 earned.
    joined = date(2026, 7, 1)
    assert [co.prorated_entitlement(t, joined, 2026) for t in LeaveType] == [6, 5, 9]


@pytest.mark.parametrize(
    ("joined", "full_months_left"),
    [
        (date(2026, 1, 1), 12),
        (date(2026, 1, 2), 11),
        (date(2026, 7, 15), 5),
        (date(2026, 12, 1), 1),
        (date(2026, 12, 2), 0),
    ],
)
def test_prorating_counts_full_months_left(joined: date, full_months_left: int) -> None:
    assert co.prorated_entitlement(LeaveType.CASUAL, joined, 2026) == full_months_left


def test_no_prorating_after_the_joining_year() -> None:
    assert co.prorated_entitlement(LeaveType.EARNED, date(2025, 9, 9), 2026) == 18
    assert co.prorated_entitlement(LeaveType.EARNED, date(2027, 1, 1), 2026) == 0


@pytest.mark.parametrize("year", range(2024, 2031))
def test_ten_public_holidays_a_year(year: int) -> None:
    holidays = co.public_holidays(year)
    assert len(holidays) == co.PUBLIC_HOLIDAYS_PER_YEAR
    assert all(day.year == year for day, _ in holidays)


def test_good_friday() -> None:
    holidays = {name: day for day, name in co.public_holidays(2026)}
    assert holidays["Good Friday"] == date(2026, 4, 3)
    assert {n: d for d, n in co.public_holidays(2028)}["Good Friday"] == date(
        2028, 4, 14
    )


def test_unknown_year_fails_loudly() -> None:
    with pytest.raises(ValueError, match="2031"):
        co.public_holidays(2031)
