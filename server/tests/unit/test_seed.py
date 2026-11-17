"""The seed data has to follow the same leave rules as the backend, or the
assistant's answers and the balances it shows won't add up."""

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from functools import cache

import pytest
from warden.models import AttendanceStatus, LeaveStatus, LeaveType, Role
from warden.scripts import company as co
from warden.scripts.seed import CompanyData, build_company
from warden.services.leave_days import count_working_days, is_working_day

NOW = datetime(2026, 11, 17, 10, 30, tzinfo=co.TIMEZONE)
# Mid-year, the first working days of a year, and the last days of one.
DATES = [
    NOW,
    datetime(2026, 6, 15, 9, 0, tzinfo=co.TIMEZONE),
    datetime(2027, 1, 2, 2, 0, tzinfo=co.TIMEZONE),
    datetime(2026, 12, 30, 23, 0, tzinfo=co.TIMEZONE),
]


@cache
def _company(now: datetime) -> CompanyData:
    return build_company(now)


@pytest.fixture(params=DATES, ids=lambda d: d.date().isoformat())
def now(request: pytest.FixtureRequest) -> datetime:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def data(now: datetime) -> CompanyData:
    return _company(now)


def _holidays(data: CompanyData) -> set[date]:
    return {h["date"] for h in data.holidays}


def _by_email(data: CompanyData) -> dict[str, dict]:  # type: ignore[type-arg]
    return {e["email"]: e for e in data.employees}


def test_headcount(data: CompanyData) -> None:
    roles = Counter(e["role"] for e in data.employees)
    assert roles == {Role.HR: 3, Role.MANAGER: 5, Role.EMPLOYEE: 45}
    assert {d["name"] for d in data.departments} == set(co.DEPARTMENTS)

    managers = [e for e in data.employees if e["role"] is Role.MANAGER]
    assert len({m["department_id"] for m in managers}) == 5
    assert len({e["email"] for e in data.employees}) == len(data.employees)
    assert len({e["employee_code"] for e in data.employees}) == len(data.employees)


def test_every_employee_reports_to_a_manager_in_their_department(
    data: CompanyData,
) -> None:
    people = {e["id"]: e for e in data.employees}
    for e in data.employees:
        if e["role"] is Role.EMPLOYEE:
            manager = people[e["manager_id"]]
            assert manager["role"] is Role.MANAGER
            assert manager["department_id"] == e["department_id"]


def test_demo_accounts(data: CompanyData, now: datetime) -> None:
    people = _by_email(data)
    employee = people[co.DEMO_ACCOUNTS[Role.EMPLOYEE]]
    manager = people[co.DEMO_ACCOUNTS[Role.MANAGER]]
    hr = people[co.DEMO_ACCOUNTS[Role.HR]]
    assert (employee["role"], manager["role"], hr["role"]) == (
        Role.EMPLOYEE,
        Role.MANAGER,
        Role.HR,
    )
    assert employee["manager_id"] == manager["id"]

    today = now.date()
    mine = [r for r in data.leave_requests if r["employee_id"] == employee["id"]]
    assert any(
        r["status"] is LeaveStatus.PENDING and r["start_date"] > today for r in mine
    ), "a pending request for the demo manager to approve"
    assert any(
        r["status"] is LeaveStatus.APPROVED and r["start_date"] > today for r in mine
    ), "approved leave that can still be cancelled"


def test_every_status_is_seeded(data: CompanyData) -> None:
    assert {r["status"] for r in data.leave_requests} == set(LeaveStatus)


def test_request_days_are_working_days(data: CompanyData) -> None:
    holidays = _holidays(data)
    for r in data.leave_requests:
        assert (
            r["days"]
            == count_working_days(r["start_date"], r["end_date"], holidays)
            > 0
        )
        assert is_working_day(r["start_date"], holidays)


def test_requests_never_overlap(data: CompanyData) -> None:
    by_employee = defaultdict(list)
    for r in data.leave_requests:
        by_employee[r["employee_id"]].append((r["start_date"], r["end_date"]))
    for spans in by_employee.values():
        spans.sort()
        for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert end < next_start


def test_balances_match_requests(data: CompanyData) -> None:
    used: Counter[tuple] = Counter()  # type: ignore[type-arg]
    pending: Counter[tuple] = Counter()  # type: ignore[type-arg]
    for r in data.leave_requests:
        key = (r["employee_id"], r["start_date"].year, r["leave_type"])
        if r["status"] is LeaveStatus.APPROVED:
            used[key] += r["days"]
        elif r["status"] is LeaveStatus.PENDING:
            pending[key] += r["days"]

    balances = {
        (b["employee_id"], b["year"], b["leave_type"]): b for b in data.leave_balances
    }
    assert set(used) | set(pending) <= set(balances)
    for key, b in balances.items():
        assert b["used"] == used[key]
        assert b["used"] + pending[key] <= b["allotted"]


def test_everyone_has_this_years_balances(data: CompanyData, now: datetime) -> None:
    keys = {(b["employee_id"], b["year"], b["leave_type"]) for b in data.leave_balances}
    for e in data.employees:
        for leave_type in LeaveType:
            assert (e["id"], now.year, leave_type) in keys


def test_earned_leave_carry_forward_is_capped(data: CompanyData, now: datetime) -> None:
    full = co.LEAVE_ENTITLEMENT[LeaveType.EARNED]
    for b in data.leave_balances:
        if b["leave_type"] is LeaveType.EARNED and b["year"] == now.year:
            assert b["allotted"] <= full + co.EARNED_LEAVE_CARRY_FORWARD_LIMIT
        elif b["year"] == now.year:
            assert b["allotted"] <= co.LEAVE_ENTITLEMENT[b["leave_type"]]


def test_pending_requests_follow_the_start_date_rule(
    data: CompanyData, now: datetime
) -> None:
    today = now.date()
    for r in data.leave_requests:
        if r["status"] is not LeaveStatus.PENDING or r["start_date"] >= today:
            continue
        assert r["leave_type"] is LeaveType.SICK
        assert r["start_date"] >= r["created_at"].date() - timedelta(
            days=co.SICK_LEAVE_BACKDATE_DAYS
        )


def test_pending_requests_have_a_manager_who_can_approve(data: CompanyData) -> None:
    people = {e["id"]: e for e in data.employees}
    for r in data.leave_requests:
        if r["status"] is LeaveStatus.PENDING:
            manager = people[people[r["employee_id"]]["manager_id"]]
            assert manager["role"] is Role.MANAGER


def test_decisions(data: CompanyData, now: datetime) -> None:
    people = {e["id"]: e for e in data.employees}
    for r in data.leave_requests:
        assert r["created_at"] <= now
        assert r["created_at"].date() >= people[r["employee_id"]]["joining_date"]
        if r["status"] in (LeaveStatus.APPROVED, LeaveStatus.REJECTED):
            assert (
                r["approver_id"]
                == people[r["employee_id"]]["manager_id"]
                != r["employee_id"]
            )
            assert r["created_at"] <= r["decided_at"] <= now
        if r["status"] is LeaveStatus.REJECTED:
            assert r["decision_comment"]
        if r["status"] is LeaveStatus.PENDING:
            assert r["approver_id"] is None and r["decided_at"] is None


def test_attendance(data: CompanyData, now: datetime) -> None:
    today = now.date()
    holidays = _holidays(data)
    approved_days = {
        (r["employee_id"], r["start_date"] + timedelta(days=n))
        for r in data.leave_requests
        if r["status"] is LeaveStatus.APPROVED
        for n in range((r["end_date"] - r["start_date"]).days + 1)
    }
    joined = {e["id"]: e["joining_date"] for e in data.employees}

    for row in data.attendance:
        day = row["date"]
        assert today - timedelta(days=90) <= day < today
        assert day.weekday() < 5
        assert day >= joined[row["employee_id"]]
        if day in holidays:
            assert row["status"] is AttendanceStatus.HOLIDAY
        elif (row["employee_id"], day) in approved_days:
            assert row["status"] is AttendanceStatus.ON_LEAVE
        else:
            assert row["status"] is not AttendanceStatus.ON_LEAVE

        has_times = row["status"] in (AttendanceStatus.PRESENT, AttendanceStatus.WFH)
        assert (row["check_in"] is not None) == has_times
        if has_times:
            assert row["check_in"].date() == day
            assert row["check_in"] < row["check_out"]
            assert row["check_out"].date() == day

    statuses = {row["status"] for row in data.attendance}
    assert statuses >= {
        AttendanceStatus.PRESENT,
        AttendanceStatus.WFH,
        AttendanceStatus.ABSENT,
    }


def test_payslips(data: CompanyData, now: datetime) -> None:
    per_employee = Counter(p["employee_id"] for p in data.payslips)
    demo = _by_email(data)[co.DEMO_ACCOUNTS[Role.EMPLOYEE]]
    assert per_employee[demo["id"]] == 12
    assert max(per_employee.values()) == 12
    for p in data.payslips:
        assert (p["year"], p["month"]) < (now.year, now.month)
        assert p["net_pay"] > 0


def test_holidays_cover_the_years_in_use(data: CompanyData, now: datetime) -> None:
    years = Counter(h["date"].year for h in data.holidays)
    assert years == {now.year - 1: 10, now.year: 10, now.year + 1: 10}


def test_same_day_gives_the_same_company() -> None:
    first, second = build_company(NOW), build_company(NOW)

    def shape(d: CompanyData) -> list[object]:
        return [
            [(e["name"], e["email"], e["job_title"]) for e in d.employees],
            [(r["start_date"], r["end_date"], r["status"]) for r in d.leave_requests],
            [(a["date"], a["status"]) for a in d.attendance],
            [p["net_pay"] for p in d.payslips],
        ]

    assert shape(first) == shape(second)
