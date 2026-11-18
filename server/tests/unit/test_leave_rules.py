from datetime import date, timedelta

import pytest

from warden.models import LeaveType
from warden.services.errors import BackdateNotAllowed, InvalidDates
from warden.services.leave import check_dates

TODAY = date(2026, 6, 10)


@pytest.mark.parametrize("leave_type", list(LeaveType))
@pytest.mark.parametrize(
    ("start", "end"),
    [(TODAY, TODAY), (TODAY, TODAY + timedelta(days=4)), (date(2026, 12, 31), date(2026, 12, 31))],
)
def test_today_or_later_is_fine_for_every_type(
    leave_type: LeaveType, start: date, end: date
) -> None:
    check_dates(leave_type, start, end, TODAY)


@pytest.mark.parametrize("leave_type", list(LeaveType))
def test_end_before_start_is_refused(leave_type: LeaveType) -> None:
    with pytest.raises(InvalidDates, match="before the start date"):
        check_dates(leave_type, TODAY + timedelta(days=2), TODAY + timedelta(days=1), TODAY)


def test_a_range_across_31_december_is_refused() -> None:
    with pytest.raises(InvalidDates, match="crosses 31 December"):
        check_dates(LeaveType.EARNED, date(2026, 12, 30), date(2027, 1, 2), TODAY)


@pytest.mark.parametrize("leave_type", [LeaveType.CASUAL, LeaveType.EARNED])
def test_only_sick_leave_can_start_in_the_past(leave_type: LeaveType) -> None:
    yesterday = TODAY - timedelta(days=1)

    with pytest.raises(BackdateNotAllowed) as exc_info:
        check_dates(leave_type, yesterday, TODAY, TODAY)
    assert exc_info.value.details == {"earliest": TODAY}

    check_dates(LeaveType.SICK, yesterday, TODAY, TODAY)


def test_sick_leave_can_be_backdated_by_up_to_7_days() -> None:
    check_dates(LeaveType.SICK, TODAY - timedelta(days=7), TODAY - timedelta(days=5), TODAY)

    with pytest.raises(BackdateNotAllowed, match="at most 7 days") as exc_info:
        check_dates(LeaveType.SICK, TODAY - timedelta(days=8), TODAY, TODAY)
    assert exc_info.value.details == {"earliest": TODAY - timedelta(days=7)}
