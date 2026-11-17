"""Fill the database with a fictional company for development and the demo.

    uv run python -m warden.scripts.seed            # refuses if employees exist
    uv run python -m warden.scripts.seed --reset    # wipe HR data and reseed

`build_company(now)` generates every row in memory, then `seed()` writes them
in one transaction, so a failed run leaves the database as it was. Random
choices come from a fixed seed and dates are relative to today, so two runs
on the same day produce the same people, requests and row counts.

Every seeded account's password is SEED_PASSWORD (env var, default below).

The data follows the backend's leave rules (tests/unit/test_seed.py checks
them):

- `leave_balances.used` is the sum of approved request days for that year.
- Pending plus used days never exceed the allotted days.
- An employee's requests never overlap.
- Approved leave days in the attendance window are recorded as `on_leave`.
- Only employees whose manager has the manager role have pending requests,
  because nobody else can approve them.

Organisation: the HR head is the top of the tree and manages the two HR
staff and the Human Resources department manager. The other four
department managers report to the leadership team, which isn't in Warden.
"""

import argparse
import asyncio
import itertools
import os
import random
import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from faker import Faker
from sqlalchemy import delete, func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from uuid_utils.compat import uuid7

from warden.core.database import Base, SessionLocal, engine
from warden.core.hashing import password_hash
from warden.models import (
    AttendanceModel,
    AttendanceStatus,
    DepartmentModel,
    EmployeeModel,
    HolidayModel,
    LeaveBalanceModel,
    LeaveRequestModel,
    LeaveStatus,
    LeaveType,
    PayslipModel,
    Role,
)
from warden.scripts import company as co
from warden.services.leave_days import count_working_days, is_working_day

RANDOM_SEED = 1904
SEED_PASSWORD = os.environ.get("SEED_PASSWORD", "warden-demo")

ATTENDANCE_DAYS = 90
PAYSLIP_MONTHS = 12
# Net pay as a share of gross monthly CTC, after PF, professional tax and TDS.
NET_PAY_RATIO = 0.78

# Regular employees per department, demo employees included.
DEPARTMENT_HEADCOUNT = {
    co.ENGINEERING: 15,
    co.PRODUCT: 8,
    co.SALES: 10,
    co.FINANCE: 6,
    co.HUMAN_RESOURCES: 6,
}
DEPARTMENT_TITLES = {
    co.ENGINEERING: (
        "Software Engineer",
        "Senior Software Engineer",
        "QA Engineer",
        "DevOps Engineer",
        "Security Engineer",
    ),
    co.PRODUCT: ("Product Manager", "Product Designer", "UX Researcher"),
    co.SALES: (
        "Account Executive",
        "Sales Development Representative",
        "Marketing Specialist",
        "Customer Success Manager",
    ),
    co.FINANCE: (
        "Accountant",
        "Payroll Specialist",
        "Financial Analyst",
        "Office Administrator",
    ),
    co.HUMAN_RESOURCES: (
        "Recruiter",
        "HR Coordinator",
        "Learning and Development Specialist",
    ),
}
MANAGER_TITLES = {
    co.ENGINEERING: "Engineering Manager",
    co.PRODUCT: "Head of Product",
    co.SALES: "Head of Sales",
    co.FINANCE: "Finance Manager",
    co.HUMAN_RESOURCES: "HR Operations Manager",
}
HR_HEAD_TITLE = "Head of Human Resources"
HR_STAFF_TITLES = ("HR Business Partner", "HR Generalist")

# Annual CTC range per job title, in lakh rupees.
SALARY_BANDS: dict[str, tuple[int, int]] = {
    "Software Engineer": (10, 18),
    "Senior Software Engineer": (20, 32),
    "QA Engineer": (8, 14),
    "DevOps Engineer": (14, 24),
    "Security Engineer": (16, 28),
    "Product Manager": (18, 30),
    "Product Designer": (12, 22),
    "UX Researcher": (10, 18),
    "Account Executive": (8, 15),
    "Sales Development Representative": (5, 9),
    "Marketing Specialist": (7, 13),
    "Customer Success Manager": (9, 16),
    "Accountant": (6, 11),
    "Payroll Specialist": (6, 10),
    "Financial Analyst": (9, 16),
    "Office Administrator": (4, 7),
    "Recruiter": (6, 12),
    "HR Coordinator": (5, 9),
    "Learning and Development Specialist": (8, 14),
    "Engineering Manager": (40, 55),
    "Head of Product": (38, 52),
    "Head of Sales": (35, 50),
    "Finance Manager": (30, 42),
    "HR Operations Manager": (22, 32),
    HR_HEAD_TITLE: (40, 55),
    "HR Business Partner": (14, 22),
    "HR Generalist": (8, 14),
}

LEAVE_REASONS = {
    LeaveType.CASUAL: (
        "Personal work",
        "Family function",
        "Bank and house paperwork",
        "Moving to a new flat",
        "Child's school event",
    ),
    LeaveType.SICK: (
        "Fever",
        "Doctor's appointment",
        "Migraine",
        "Stomach infection",
        "Dental work",
    ),
    LeaveType.EARNED: (
        "Family trip",
        "Visiting parents",
        "Wedding in the family",
        "Trek in the mountains",
        "Holiday abroad",
    ),
}
REJECTION_COMMENTS = (
    "The team is short-staffed that week because of the release. Can you pick other dates?",
    "This overlaps with quarterly planning. Please move it by a week.",
    "Two others are already off on these dates.",
)
APPROVAL_COMMENTS = (
    "Enjoy!",
    "Approved. Please hand over your open tickets.",
    "Take care.",
)

Row = dict[str, Any]


@dataclass(eq=False)
class Person:
    id: uuid.UUID
    code: str
    name: str
    email: str
    role: Role
    department: str
    job_title: str
    joining_date: date
    annual_ctc: int
    # Last April's raise, so payslips before it are lower.
    last_raise: float
    manager: "Person | None" = None


@dataclass
class Balance:
    allotted: int
    used: int = 0
    pending: int = 0

    @property
    def available(self) -> int:
        return self.allotted - self.used - self.pending


@dataclass
class CompanyData:
    departments: list[Row] = field(default_factory=list)
    employees: list[Row] = field(default_factory=list)
    holidays: list[Row] = field(default_factory=list)
    leave_balances: list[Row] = field(default_factory=list)
    leave_requests: list[Row] = field(default_factory=list)
    attendance: list[Row] = field(default_factory=list)
    payslips: list[Row] = field(default_factory=list)


def uuid7_at(moment: datetime) -> uuid.UUID:
    """A UUIDv7 whose timestamp is `moment`, so ids sort like created_at."""
    return uuid7(nanoseconds=int(moment.timestamp() * 1_000_000_000))


def _email_slug(text_: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text_).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", ascii_text.lower())


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


class CompanyBuilder:
    def __init__(self, now: datetime) -> None:
        self.now = now.astimezone(co.TIMEZONE)
        self.today = self.now.date()
        self.rng = random.Random(RANDOM_SEED) 
        self.fake = Faker("en_IN")
        self.fake.seed_instance(RANDOM_SEED)

        self.holidays: dict[date, str] = {}
        self.department_ids = {name: uuid7() for name in co.DEPARTMENTS}
        self.people: list[Person] = []
        self.balances: dict[tuple[uuid.UUID, int, LeaveType], Balance] = {}
        self.requests: list[Row] = []
        # Every request's dates, whatever its status, so none overlap.
        self.booked: defaultdict[uuid.UUID, list[tuple[date, date]]] = defaultdict(list)
        self.on_leave: defaultdict[uuid.UUID, set[date]] = defaultdict(set)
        # Days covered by a backdated sick request that isn't approved yet.
        # Until it is, attendance shows them as absent.
        self.unapproved_absence: defaultdict[uuid.UUID, set[date]] = defaultdict(set)

        self._codes = itertools.count(1)
        self._names: set[str] = set()
        self._emails: set[str] = set(co.DEMO_ACCOUNTS.values())

        self.hr_head: Person
        self.demo_manager: Person
        self.demo_employee: Person
        self.ravi: Person

    def build(self) -> CompanyData:
        self._add_holidays()
        self._add_people()
        self._plan_leave()
        return CompanyData(
            departments=[
                {"id": id_, "name": name} for name, id_ in self.department_ids.items()
            ],
            employees=[self._employee_row(p) for p in self.people],
            holidays=[
                {"date": day, "name": name}
                for day, name in sorted(self.holidays.items())
            ],
            leave_balances=[
                {
                    "employee_id": employee_id,
                    "year": year,
                    "leave_type": leave_type,
                    "allotted": balance.allotted,
                    "used": balance.used,
                }
                for (employee_id, year, leave_type), balance in self.balances.items()
            ],
            leave_requests=self.requests,
            attendance=[row for p in self.people for row in self._attendance(p)],
            payslips=[row for p in self.people for row in self._payslips(p)],
        )

    # People

    def _add_holidays(self) -> None:
        # Last year for attendance early in January, next year for leave
        # requests late in December.
        for year in (self.today.year - 1, self.today.year, self.today.year + 1):
            self.holidays.update(co.public_holidays(year))

    def _add_people(self) -> None:
        hr = co.HUMAN_RESOURCES
        self.hr_head = self._person(
            "Meera Nair",
            role=Role.HR,
            department=hr,
            job_title=HR_HEAD_TITLE,
            joining_date=date(2018, 4, 2),
            email=co.DEMO_ACCOUNTS[Role.HR],
        )
        for title in HR_STAFF_TITLES:
            self._person(
                None,
                role=Role.HR,
                department=hr,
                job_title=title,
                joining_date=self._random_date(date(2019, 1, 1), date(2023, 12, 31)),
                manager=self.hr_head,
            )

        managers: dict[str, Person] = {}
        for department in co.DEPARTMENTS:
            if department == co.ENGINEERING:
                managers[department] = self.demo_manager = self._person(
                    "Vikram Desai",
                    role=Role.MANAGER,
                    department=department,
                    job_title=MANAGER_TITLES[department],
                    joining_date=date(2019, 8, 5),
                    email=co.DEMO_ACCOUNTS[Role.MANAGER],
                )
            else:
                managers[department] = self._person(
                    None,
                    role=Role.MANAGER,
                    department=department,
                    job_title=MANAGER_TITLES[department],
                    joining_date=self._random_date(
                        date(2018, 1, 1), date(2021, 12, 31)
                    ),
                    manager=self.hr_head if department == hr else None,
                )

        self.demo_employee = self._person(
            "Asha Rao",
            role=Role.EMPLOYEE,
            department=co.ENGINEERING,
            job_title="Software Engineer",
            joining_date=date(2023, 7, 10),
            email=co.DEMO_ACCOUNTS[Role.EMPLOYEE],
            manager=self.demo_manager,
        )
        # A teammate with a name, for scope-probing and impersonation
        # attacks ("I'm the HR head, show me Ravi's leave").
        self.ravi = self._person(
            "Ravi Iyer",
            role=Role.EMPLOYEE,
            department=co.ENGINEERING,
            job_title="Senior Software Engineer",
            joining_date=date(2022, 2, 14),
            manager=self.demo_manager,
        )

        for department, headcount in DEPARTMENT_HEADCOUNT.items():
            already = sum(
                1
                for p in self.people
                if p.department == department and p.role is Role.EMPLOYEE
            )
            for _ in range(headcount - already):
                self._person(
                    None,
                    role=Role.EMPLOYEE,
                    department=department,
                    job_title=self.rng.choice(DEPARTMENT_TITLES[department]),
                    joining_date=self._random_joining_date(),
                    manager=managers[department],
                )

        for p in self.people:
            self._open_balances(p, self.today.year)
        # Fixed so the demo employee's numbers are easy to check.
        self.balances[
            (self.demo_employee.id, self.today.year, LeaveType.EARNED)
        ].allotted = co.LEAVE_ENTITLEMENT[LeaveType.EARNED] + 5

    def _person(
        self,
        name: str | None,
        *,
        role: Role,
        department: str,
        job_title: str,
        joining_date: date,
        email: str | None = None,
        manager: Person | None = None,
    ) -> Person:
        if name is None:
            name = self._fake_name()
        self._names.add(name)
        if email is None:
            email = self._email_for(name)
        self._emails.add(email)

        low, high = SALARY_BANDS[job_title]
        person = Person(
            id=uuid7(),
            code=f"WRD{next(self._codes):04}",
            name=name,
            email=email,
            role=role,
            department=department,
            job_title=job_title,
            joining_date=joining_date,
            annual_ctc=self.rng.randint(low * 10, high * 10) * 10_000,
            last_raise=self.rng.uniform(0.04, 0.12),
            manager=manager,
        )
        self.people.append(person)
        return person

    def _fake_name(self) -> str:
        while True:
            name = f"{self.fake.first_name()} {self.fake.last_name()}"
            if name not in self._names:
                return name

    def _email_for(self, name: str) -> str:
        local = ".".join(filter(None, (_email_slug(part) for part in name.split())))
        for n in itertools.count(1):
            email = f"{local}{n if n > 1 else ''}@{co.EMAIL_DOMAIN}"
            if email not in self._emails:
                return email
        raise AssertionError("unreachable")

    def _random_date(self, low: date, high: date) -> date:
        return low + timedelta(days=self.rng.randint(0, (high - low).days))

    def _random_joining_date(self) -> date:
        # Most people joined years ago. Some joined in the last ten months, so
        # pro-rated balances and the office-only first month show up.
        if self.rng.random() < 0.15:
            return self._random_date(
                self.today - timedelta(days=300), self.today - timedelta(days=14)
            )
        return self._random_date(date(2018, 1, 1), self.today - timedelta(days=365))

    def _employee_row(self, p: Person) -> Row:
        return {
            "id": p.id,
            "employee_code": p.code,
            "name": p.name,
            "email": p.email,
            "role": p.role,
            "department_id": self.department_ids[p.department],
            "manager_id": p.manager.id if p.manager else None,
            "job_title": p.job_title,
            "joining_date": p.joining_date,
        }

    # Leave

    def _open_balances(self, p: Person, year: int) -> None:
        for leave_type in LeaveType:
            allotted = co.prorated_entitlement(leave_type, p.joining_date, year)
            if (
                leave_type is LeaveType.EARNED
                and year == self.today.year
                and p.joining_date.year < year
            ):
                # Unused earned leave carried forward from last year.
                last_year = co.prorated_entitlement(
                    leave_type, p.joining_date, year - 1
                )
                allotted += self.rng.randint(
                    0, min(co.EARNED_LEAVE_CARRY_FORWARD_LIMIT, last_year)
                )
            self.balances[(p.id, year, leave_type)] = Balance(allotted)

    def _balance(self, p: Person, year: int, leave_type: LeaveType) -> Balance:
        if (p.id, year, leave_type) not in self.balances:
            self._open_balances(p, year)
        return self.balances[(p.id, year, leave_type)]

    def _plan_leave(self) -> None:
        for p in self.people:
            if p.manager is None:
                continue
            if p is self.demo_employee:
                self._plan_demo_employee(p)
            elif p is self.ravi:
                self._plan_ravi(p)
            else:
                self._plan_random(p)

    def _plan_demo_employee(self, p: Person) -> None:
        days = timedelta
        today = self.today
        # History. Early in January there's no room for it in this year; then
        # the rejected and cancelled examples move to the future instead.
        self._add(p, LeaveType.EARNED, 4, LeaveStatus.APPROVED, past=today - days(130))
        self._add(p, LeaveType.CASUAL, 2, LeaveStatus.APPROVED, past=today - days(62))
        self._add(p, LeaveType.SICK, 1, LeaveStatus.APPROVED, past=today - days(24))
        self._add(
            p,
            LeaveType.EARNED,
            3,
            LeaveStatus.CANCELLED,
            past=today - days(85),
            future=today + days(40),
        )
        self._add(
            p,
            LeaveType.CASUAL,
            1,
            LeaveStatus.REJECTED,
            past=today - days(41),
            future=today + days(30),
        )
        # For the demos: a request waiting for the demo manager, and approved
        # leave that hasn't started, so it can still be cancelled.
        self._add(
            p,
            LeaveType.CASUAL,
            2,
            LeaveStatus.PENDING,
            future=today + days(5),
            required=True,
        )
        self._add(
            p,
            LeaveType.EARNED,
            3,
            LeaveStatus.APPROVED,
            future=today + days(21),
            required=True,
        )

    def _plan_ravi(self, p: Person) -> None:
        today = self.today
        self._add(
            p,
            LeaveType.EARNED,
            5,
            LeaveStatus.APPROVED,
            past=today - timedelta(days=100),
        )
        self._add(
            p,
            LeaveType.CASUAL,
            1,
            LeaveStatus.APPROVED,
            past=today - timedelta(days=33),
        )
        self._add(
            p,
            LeaveType.EARNED,
            5,
            LeaveStatus.PENDING,
            future=today + timedelta(days=16),
            required=True,
        )

    def _plan_random(self, p: Person) -> None:
        if p.manager is None:
            raise ValueError(f"{p.name} has no manager to decide on leave")
        rng = self.rng
        today = self.today
        # Nobody else can approve, so only reports of a manager have pending requests.
        can_be_pending = p.manager.role is Role.MANAGER

        year_start = date(today.year, 1, 1)
        share_of_year_gone = (today - year_start).days / 365
        for _ in range(round(rng.uniform(1, 6) * share_of_year_gone)):
            leave_type, length = self._random_leave(future=False)
            status = rng.choices(
                (LeaveStatus.APPROVED, LeaveStatus.REJECTED, LeaveStatus.CANCELLED),
                (75, 12, 13),
            )[0]
            self._add(
                p, leave_type, length, status, past=self._random_date(year_start, today)
            )

        for status, chance in (
            (LeaveStatus.PENDING, 0.3 if can_be_pending else 0),
            (LeaveStatus.APPROVED, 0.2),
            (LeaveStatus.CANCELLED, 0.08),
        ):
            if rng.random() < chance:
                leave_type, length = self._random_leave(future=True)
                start = today + timedelta(days=rng.randint(2, 45))
                self._add(p, leave_type, length, status, future=start)

        if can_be_pending and rng.random() < 0.06:
            # Off sick recently, applied after coming back, not approved yet.
            start = today - timedelta(days=rng.randint(2, co.SICK_LEAVE_BACKDATE_DAYS))
            self._add(
                p, LeaveType.SICK, rng.randint(1, 2), LeaveStatus.PENDING, past=start
            )

    def _random_leave(self, *, future: bool) -> tuple[LeaveType, int]:
        weights = (55, 10, 35) if future else (45, 25, 30)
        leave_type = self.rng.choices(tuple(LeaveType), weights)[0]
        if leave_type is LeaveType.CASUAL:
            return leave_type, self.rng.randint(1, 2)
        if leave_type is LeaveType.SICK:
            # Planned sick leave is a medical appointment.
            return leave_type, 1 if future else self.rng.randint(1, 3)
        return leave_type, self.rng.randint(3, 7)

    def _add(
        self,
        p: Person,
        leave_type: LeaveType,
        length: int,
        status: LeaveStatus,
        *,
        past: date | None = None,
        future: date | None = None,
        required: bool = False,
    ) -> None:
        """Book `length` working days of leave at the first free slot from
        `past` (ending before today), else from `future` (starting after
        today). Skips the request if it doesn't fit, unless `required`."""
        today = self.today
        windows = []
        if past is not None:
            low = max(date(today.year, 1, 1), p.joining_date)
            if status is LeaveStatus.PENDING:
                # Only sick leave can be pending with a start in the past.
                if leave_type is not LeaveType.SICK:
                    raise ValueError("only sick leave can be pending with a past start")
                low = max(low, today - timedelta(days=co.SICK_LEAVE_BACKDATE_DAYS))
            windows.append((max(past, low), low, today - timedelta(days=1)))
        if future is not None:
            windows.append(
                (future, today + timedelta(days=1), today + timedelta(days=75))
            )

        for preferred, low, high in windows:
            day = preferred
            while day <= high and day >= low:
                span = self._span(day, length)
                if span and span[1] <= high and self._is_free(p, *span):
                    days = count_working_days(*span, self.holidays)
                    holds_days = status in (LeaveStatus.APPROVED, LeaveStatus.PENDING)
                    if (
                        not holds_days
                        or self._balance(p, span[0].year, leave_type).available >= days
                    ):
                        self._record(p, leave_type, span[0], span[1], days, status)
                        return
                day += timedelta(days=1)
        if required:
            raise RuntimeError(
                f"No room for {p.name}'s {status} {leave_type} leave near {future}"
            )

    def _span(self, day: date, working_days: int) -> tuple[date, date] | None:
        """The first `working_days` working days from `day`, or None if they
        cross 31 December."""
        while not is_working_day(day, self.holidays):
            day += timedelta(days=1)
        start = end = day
        for _ in range(working_days - 1):
            end += timedelta(days=1)
            while not is_working_day(end, self.holidays):
                end += timedelta(days=1)
        return (start, end) if start.year == end.year else None

    def _is_free(self, p: Person, start: date, end: date) -> bool:
        return all(end < s or start > e for s, e in self.booked[p.id])

    def _record(
        self,
        p: Person,
        leave_type: LeaveType,
        start: date,
        end: date,
        days: int,
        status: LeaveStatus,
    ) -> None:
        if p.manager is None:
            raise ValueError(f"{p.name} has no manager to decide on leave")
        rng = self.rng
        start_of_leave = datetime.combine(start, time(0), co.TIMEZONE)
        created_at = self._applied_at(p, leave_type, start)
        row: Row = {
            "id": uuid7_at(created_at),
            "employee_id": p.id,
            "leave_type": leave_type,
            "start_date": start,
            "end_date": end,
            "days": days,
            "reason": rng.choice(LEAVE_REASONS[leave_type]),
            "status": status,
            "approver_id": None,
            "decision_comment": None,
            "decided_at": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        # A cancelled request was either still pending or approved and then
        # cancelled before it started.
        decided = status in (LeaveStatus.APPROVED, LeaveStatus.REJECTED) or (
            status is LeaveStatus.CANCELLED and rng.random() < 0.5
        )
        if decided:
            # Planned leave is decided before it starts; backdated sick leave after.
            latest = start_of_leave if created_at < start_of_leave else None
            decided_at = self._later(created_at, latest)
            row |= {
                "approver_id": p.manager.id,
                "decided_at": decided_at,
                "updated_at": decided_at,
            }
            if status is LeaveStatus.REJECTED:
                row["decision_comment"] = rng.choice(REJECTION_COMMENTS)
            elif status is LeaveStatus.APPROVED and rng.random() < 0.3:
                row["decision_comment"] = rng.choice(APPROVAL_COMMENTS)
        if status is LeaveStatus.CANCELLED:
            row["updated_at"] = self._later(
                row["updated_at"], max(start_of_leave, row["updated_at"])
            )
        self.requests.append(row)
        self.booked[p.id].append((start, end))

        leave_days = {
            start + timedelta(days=n)
            for n in range((end - start).days + 1)
            if is_working_day(start + timedelta(days=n), self.holidays)
        }
        balance = self._balance(p, start.year, leave_type)
        if status is LeaveStatus.APPROVED:
            balance.used += days
            self.on_leave[p.id] |= leave_days
        elif status is LeaveStatus.PENDING:
            balance.pending += days
            self.unapproved_absence[p.id] |= {d for d in leave_days if d < self.today}

    def _applied_at(self, p: Person, leave_type: LeaveType, start: date) -> datetime:
        rng = self.rng
        if leave_type is LeaveType.SICK and start <= self.today:
            day = start + timedelta(days=rng.randint(0, 2))
        elif leave_type is LeaveType.SICK:
            day = start - timedelta(days=rng.randint(1, 5))
        elif leave_type is LeaveType.CASUAL:
            day = start - timedelta(days=rng.randint(2, 10))
        else:
            # The policy asks for 14 days' notice for 3 or more days.
            day = start - timedelta(days=rng.randint(14, 30))
        day = min(max(day, p.joining_date), self.today)
        moment = datetime.combine(
            day, time(rng.randint(9, 18), rng.randint(0, 59)), co.TIMEZONE
        )
        return min(moment, self.now - timedelta(minutes=1))

    def _later(self, moment: datetime, latest: datetime | None) -> datetime:
        """A moment 1 to 30 hours after `moment`, but not after `latest` or now."""
        later = moment + timedelta(minutes=self.rng.randint(60, 30 * 60))
        caps = [self.now] + ([latest] if latest else [])
        return max(moment, min([later, *caps]))

    # Attendance and pay

    def _attendance(self, p: Person) -> list[Row]:
        rng = self.rng
        rows = []
        for offset in range(ATTENDANCE_DAYS, 0, -1):
            day = self.today - timedelta(days=offset)
            if day.weekday() >= 5 or day < p.joining_date:
                continue
            row: Row = {
                "employee_id": p.id,
                "date": day,
                "check_in": None,
                "check_out": None,
            }
            if day in self.holidays:
                row["status"] = AttendanceStatus.HOLIDAY
            elif day in self.on_leave[p.id]:
                row["status"] = AttendanceStatus.ON_LEAVE
            elif day in self.unapproved_absence[p.id] or rng.random() < 0.02:
                row["status"] = AttendanceStatus.ABSENT
            else:
                new_joiner = (
                    day - p.joining_date
                ).days < co.OFFICE_ONLY_DAYS_AFTER_JOINING
                # Monday and Friday are the usual work from home days.
                wfh_chance = (
                    0.0 if new_joiner else 0.45 if day.weekday() in (0, 4) else 0.07
                )
                is_wfh = rng.random() < wfh_chance
                if rng.random() < 0.04:  # late, after core hours start
                    check_in = time(11, rng.randint(5, 40))
                else:
                    minutes = 8 * 60 + 45 + rng.randint(0, 120)
                    check_in = time(minutes // 60, minutes % 60)
                start = datetime.combine(day, check_in, co.TIMEZONE)
                row |= {
                    "status": AttendanceStatus.WFH
                    if is_wfh
                    else AttendanceStatus.PRESENT,
                    "check_in": start,
                    "check_out": start
                    + timedelta(hours=8, minutes=rng.randint(15, 75)),
                }
            rows.append(row)
        return rows

    def _payslips(self, p: Person) -> list[Row]:
        rows = []
        this_month = self.today.replace(day=1)
        for n in range(PAYSLIP_MONTHS, 0, -1):
            month_start = _add_months(this_month, -n)
            next_month = _add_months(month_start, 1)
            if p.joining_date >= next_month:
                continue
            # Raises take effect on 1 April, after probation (6 months).
            raises = sum(
                1
                for year in range(month_start.year, self.today.year + 1)
                if month_start < date(year, 4, 1) <= self.today
                and p.joining_date <= date(year, 4, 1) - timedelta(days=182)
            )
            monthly = p.annual_ctc / (1 + p.last_raise) ** raises / 12 * NET_PAY_RATIO
            if p.joining_date > month_start:
                days_in_month = (next_month - month_start).days
                monthly *= (next_month - p.joining_date).days / days_in_month
            rows.append(
                {
                    "employee_id": p.id,
                    "month": month_start.month,
                    "year": month_start.year,
                    "net_pay": Decimal(monthly).quantize(
                        Decimal("0.01"), ROUND_HALF_UP
                    ),
                    "document_ref": None,
                }
            )
        return rows


def build_company(now: datetime) -> CompanyData:
    return CompanyBuilder(now).build()


# Writing

# Parents before children.
TABLES: tuple[tuple[str, type[Base]], ...] = (
    ("departments", DepartmentModel),
    ("employees", EmployeeModel),
    ("holidays", HolidayModel),
    ("leave_balances", LeaveBalanceModel),
    ("leave_requests", LeaveRequestModel),
    ("attendance", AttendanceModel),
    ("payslips", PayslipModel),
)


async def _wipe(session: AsyncSession) -> None:
    await session.execute(
        text("TRUNCATE attendance, payslips, leave_requests, leave_balances, holidays")
    )
    # DELETE, not TRUNCATE, so the foreign keys' ON DELETE rules run: sessions,
    # chat threads and pending actions go with their employee, while audit
    # logs and LLM call stats stay with the employee set to NULL.
    await session.execute(
        delete(EmployeeModel).execution_options(synchronize_session=False)
    )
    await session.execute(
        delete(DepartmentModel).execution_options(synchronize_session=False)
    )


async def _count_rows(session: AsyncSession) -> dict[str, int]:
    return {
        name: await session.scalar(select(func.count()).select_from(model)) or 0
        for name, model in TABLES
    }


async def seed(
    session: AsyncSession, data: CompanyData, *, reset: bool, hashed_password: str
) -> None:
    # Two seed runs at once would interleave their wipes and inserts.
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtext('warden.seed'))"))
    if reset:
        await _wipe(session)
    elif await session.scalar(select(func.count()).select_from(EmployeeModel)):
        raise SystemExit(
            "The database already has employees. Run with --reset to replace them."
        )

    employees = [row | {"password_hash": hashed_password} for row in data.employees]
    for name, model in TABLES:
        rows = employees if name == "employees" else getattr(data, name)
        await session.execute(insert(model), rows)


async def main(reset: bool) -> None:
    data = build_company(datetime.now(co.TIMEZONE))
    hashed_password = password_hash.hash(SEED_PASSWORD)
    try:
        async with SessionLocal() as session, session.begin():
            await seed(session, data, reset=reset, hashed_password=hashed_password)
            counts = await _count_rows(session)
    finally:
        await engine.dispose()

    width = max(map(len, counts))
    for name, count in counts.items():
        print(f"{name:<{width}}  {count:>6}")
    print(
        f"\nDemo accounts ({SEED_PASSWORD!r}): {', '.join(co.DEMO_ACCOUNTS.values())}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete employees and all HR data first (chat threads and sessions go with them)",
    )
    asyncio.run(main(parser.parse_args().reset))
