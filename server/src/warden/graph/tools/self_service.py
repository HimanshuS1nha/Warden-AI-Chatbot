"""Tools that read the signed-in employee's own data. Every role gets them.

Results are JSON strings. Internal UUIDs are left out because they mean
nothing to the user, except leave request IDs, which the model needs to
cancel a request later.
"""

import json
from collections import Counter
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from warden.core.dependencies import CurrentUser
from warden.graph.tools.common import Month, Year, company_time
from warden.graph.tools.services import ToolServices
from warden.models import AttendanceStatus, LeaveStatus
from warden.schemas.employee import EmployeeSummaryResponse, ProfileResponse
from warden.schemas.holiday import HolidayResponse
from warden.schemas.leave import LeaveBalanceResponse, LeaveRequestResponse
from warden.schemas.payslip import PayslipResponse


def build_self_service_tools(user: CurrentUser, services: ToolServices) -> list[BaseTool]:
    @tool
    async def get_my_profile() -> str:
        """Get the signed-in employee's own profile: name, employee code, email,
        job title, role, department, joining date and whether they're active.
        Use it for any question about who the user is or their job details."""
        profile = ProfileResponse.model_validate(await services.employees.get_profile(user))
        data = profile.model_dump(mode="json", exclude={"id", "department_id", "manager_id"})
        data["department"] = profile.department.name
        return json.dumps(data)

    @tool
    async def get_my_manager() -> str:
        """Get the signed-in employee's manager: name, employee code, email and
        job title. Use it for questions like "who is my manager" or "who
        approves my leave"."""
        manager = await services.employees.get_manager(user)
        if manager is None:
            return json.dumps({"manager": None, "note": "The user has no manager."})
        summary = EmployeeSummaryResponse.model_validate(manager)
        return json.dumps({"manager": summary.model_dump(mode="json", exclude={"id"})})

    @tool
    async def get_my_leave_balance(year: Year = None) -> str:
        """Get the signed-in employee's leave balances for a year, one per leave
        type (casual, sick, earned): days allotted, used, pending approval and
        available. Available is what a new request can use."""
        year = year or services.today().year
        balances = await services.leave.get_balances(user, year)
        rows = [
            LeaveBalanceResponse.model_validate(b).model_dump(mode="json", exclude={"year"})
            for b in balances
        ]
        result: dict[str, object] = {"year": year, "balances": rows}
        if not rows:
            result["note"] = "No leave balances on record for this year."
        return json.dumps(result)

    @tool
    async def get_my_leave_history(
        status: Annotated[
            LeaveStatus | None,
            Field(description="Only requests with this status. Leave out for all."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="How many, newest first.")] = 10,
    ) -> str:
        """List the signed-in employee's own leave requests, newest first: type,
        dates, working days, reason, status, the manager's decision comment, and
        when it was made and decided (company local time). Each has an `id`,
        needed to cancel that request."""
        rows = []
        for request in await services.leave.list_my_requests(user, status, limit):
            row = LeaveRequestResponse.model_validate(request).model_dump(
                mode="json", exclude={"approver_id"}
            )
            row["created_at"] = company_time(request.created_at)
            row["decided_at"] = company_time(request.decided_at)
            rows.append(row)
        return json.dumps({"requests": rows})

    @tool
    async def get_my_attendance(
        month: Month = None,
        year: Year = None,
        to_month: Annotated[
            int | None,
            Field(
                ge=1,
                le=12,
                description="For a range in the same year, the last month, e.g. month=1, "
                "to_month=10 for January to October. Leave out for one month.",
            ),
        ] = None,
    ) -> str:
        """Get the signed-in employee's attendance (statuses: present, absent,
        wfh, on_leave, holiday).

        For one month: each recorded day with check-in and check-out times in
        company local time, plus a count of days per status.
        For a range of months (set to_month): a count of days per status for
        each month, and the date and status of every day that wasn't
        `present`. Use one call for a range instead of one per month."""
        today = services.today()
        month, year = month or today.month, year or today.year
        records = await services.attendance.get_mine(user, month, year, to_month)

        result: dict[str, object]
        if not to_month or to_month == month:
            days = [
                {
                    "date": a.date.isoformat(),
                    "status": a.status.value,
                    "check_in": company_time(a.check_in),
                    "check_out": company_time(a.check_out),
                }
                for a in records
            ]
            result = {
                "month": month,
                "year": year,
                "days_by_status": Counter(d["status"] for d in days),
                "days": days,
            }
        else:
            result = {
                "year": year,
                "from_month": month,
                "to_month": to_month,
                "months": [
                    {
                        "month": m,
                        "days_by_status": Counter(
                            a.status.value for a in records if a.date.month == m
                        ),
                    }
                    for m in range(month, to_month + 1)
                ],
                "days_not_present": [
                    {"date": a.date.isoformat(), "status": a.status.value}
                    for a in records
                    if a.status != AttendanceStatus.PRESENT
                ],
            }
        if not records:
            result["note"] = "No attendance recorded for this period."
        return json.dumps(result)

    @tool
    async def get_my_payslips(year: Year = None) -> str:
        """List the signed-in employee's payslips, newest month first, with the
        net pay for each month. Leave out the year for every year on record."""
        payslips = [
            PayslipResponse.model_validate(p).model_dump(mode="json", exclude={"id"})
            for p in await services.payslips.get_mine(user, year)
        ]
        result: dict[str, object] = {"payslips": payslips}
        if not payslips:
            result["note"] = "No payslips on record" + (f" for {year}." if year else ".")
        return json.dumps(result)

    @tool
    async def get_company_holidays(year: Year = None) -> str:
        """List the company's public holidays for a year, in date order, with
        each holiday's name. They are the same for every employee and don't
        count as leave days."""
        year = year or services.today().year
        holidays = [
            HolidayResponse.model_validate(h).model_dump(mode="json")
            for h in await services.holidays.list(year)
        ]
        result: dict[str, object] = {"year": year, "holidays": holidays}
        if not holidays:
            result["note"] = "No holidays on record for this year."
        return json.dumps(result)

    return [
        get_my_profile,
        get_my_manager,
        get_my_leave_balance,
        get_my_leave_history,
        get_my_attendance,
        get_my_payslips,
        get_company_holidays,
    ]
