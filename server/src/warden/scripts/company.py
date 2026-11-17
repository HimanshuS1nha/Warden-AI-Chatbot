"""Facts about the fictional company, taken from the policy PDFs in
`server/documents/`.

The PDFs are fixed. When a number here disagrees with them, this file is
wrong. Sources:

- Employee Handbook (WD-HR-001), section 3: the five departments.
- Leave Policy v2 (WD-HR-002): entitlements (section 3), pro-rating for
  joiners (3.1), sick leave backdating (6), carry-forward (9), 10 public
  holidays (10).
- Attendance Policy (WD-HR-004): Monday to Friday, core hours 11:00 to 16:00
  IST.
- Work From Home Policy (WD-HR-005): office only for the first 30 days.
"""

from datetime import date, time, timedelta
from zoneinfo import ZoneInfo

from warden.models.enums import LeaveType, Role

COMPANY_NAME = "Warden"
EMAIL_DOMAIN = "warden.demo"
TIMEZONE = ZoneInfo("Asia/Kolkata")

ENGINEERING = "Engineering"
PRODUCT = "Product"
SALES = "Sales"
FINANCE = "Finance"
HUMAN_RESOURCES = "Human Resources"
DEPARTMENTS = (ENGINEERING, PRODUCT, SALES, FINANCE, HUMAN_RESOURCES)

# Working days per leave year (1 January to 31 December).
LEAVE_ENTITLEMENT: dict[LeaveType, int] = {
    LeaveType.CASUAL: 12,
    LeaveType.SICK: 10,
    LeaveType.EARNED: 18,
}
# Unused earned leave above this lapses on 31 December. Casual and sick
# leave don't carry forward at all.
EARNED_LEAVE_CARRY_FORWARD_LIMIT = 8
# Sick leave may start up to this many calendar days before the request.
SICK_LEAVE_BACKDATE_DAYS = 7
PUBLIC_HOLIDAYS_PER_YEAR = 10

CORE_HOURS_START = time(11, 0)
CORE_HOURS_END = time(16, 0)
OFFICE_ONLY_DAYS_AFTER_JOINING = 30

# Fixed accounts for the demo buttons on the login page.
DEMO_ACCOUNTS: dict[Role, str] = {
    Role.EMPLOYEE: f"employee@{EMAIL_DOMAIN}",
    Role.MANAGER: f"manager@{EMAIL_DOMAIN}",
    Role.HR: f"hr@{EMAIL_DOMAIN}",
}


def prorated_entitlement(leave_type: LeaveType, joining_date: date, year: int) -> int:
    """A year's entitlement for someone who joined on `joining_date`.

    Leave Policy 3.1: joiners get the entitlement pro-rated by the full
    calendar months left in the year, rounded down. Joining on 1 July gives
    6 casual, 5 sick and 9 earned days.
    """
    full = LEAVE_ENTITLEMENT[leave_type]
    if joining_date.year < year:
        return full
    if joining_date.year > year:
        return 0
    full_months_left = 12 - joining_date.month + (1 if joining_date.day == 1 else 0)
    return full * full_months_left // 12


# Holi and Diwali follow the lunar calendar, so their dates are listed.
_HOLI = {
    2024: date(2024, 3, 25),
    2025: date(2025, 3, 14),
    2026: date(2026, 3, 4),
    2027: date(2027, 3, 22),
    2028: date(2028, 3, 11),
    2029: date(2029, 3, 1),
    2030: date(2030, 3, 20),
}
_DIWALI = {
    2024: date(2024, 10, 31),
    2025: date(2025, 10, 20),
    2026: date(2026, 11, 8),
    2027: date(2027, 10, 29),
    2028: date(2028, 10, 17),
    2029: date(2029, 11, 5),
    2030: date(2030, 10, 26),
}


def _easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7 
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def public_holidays(year: int) -> list[tuple[date, str]]:
    """The 10 paid public holidays for `year`, in date order.

    A holiday that falls on a weekend isn't moved (Leave Policy section 10).
    """
    if year not in _HOLI or year not in _DIWALI:
        raise ValueError(f"No Holi and Diwali dates for {year}. Add them to {__name__}.")
    holidays = [
        (date(year, 1, 1), "New Year's Day"),
        (date(year, 1, 26), "Republic Day"),
        (_HOLI[year], "Holi"),
        (_easter_sunday(year) - timedelta(days=2), "Good Friday"),
        (date(year, 5, 1), "May Day"),
        (date(year, 8, 15), "Independence Day"),
        (date(year, 10, 2), "Gandhi Jayanti"),
        (_DIWALI[year], "Diwali"),
        (_DIWALI[year] + timedelta(days=1), "Day after Diwali"),
        (date(year, 12, 25), "Christmas Day"),
    ]
    if len({day for day, _ in holidays}) != PUBLIC_HOLIDAYS_PER_YEAR:
        raise ValueError(f"Two holidays share a date in {year}: {holidays}")
    return sorted(holidays)
