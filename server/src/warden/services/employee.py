"""Employee profiles, the reporting line, and HR search.

Who sees what:

- Everyone: their own profile and their manager.
- Managers: their active direct reports.
- HR: anyone. Every HR read of someone else is audited as `employee.view`,
  because that's what an auditor will ask about.

A manager asking for someone outside their team gets NotFound, the same as
for an ID that doesn't exist, and the attempt is audited as denied so
probing shows up in the log.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import AuditAction, AuditOutcome, AuditTarget, EmployeeModel, Role
from warden.repositories import EmployeeRepository
from warden.services import audit
from warden.services.access import ensure_role
from warden.services.errors import NotFound

MAX_SEARCH_LIMIT = 50


class EmployeeService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.employees = EmployeeRepository(db)

    async def get_profile(self, user: CurrentUser) -> EmployeeModel:
        employee = await self.employees.get(user.employee_id)
        if employee is None:  # The session check loaded this row moments ago.
            raise NotFound("Employee not found.")
        return employee

    async def get_manager(self, user: CurrentUser) -> EmployeeModel | None:
        """None at the top of the hierarchy."""
        return await self.employees.get_manager(user.employee_id)

    async def list_team(self, user: CurrentUser) -> Sequence[EmployeeModel]:
        await ensure_role(self.db, user, Role.MANAGER, operation="employee.list_team")
        return await self.employees.list_reports(user.employee_id)

    async def search(
        self,
        user: CurrentUser,
        name: str | None = None,
        department: str | None = None,
        limit: int = MAX_SEARCH_LIMIT,
    ) -> Sequence[EmployeeModel]:
        """HR only. Active employees by part of the name and exact department
        name, ignoring case."""
        await ensure_role(self.db, user, Role.HR, operation="employee.search")
        results = await self.employees.search(
            name, department, limit=max(1, min(limit, MAX_SEARCH_LIMIT))
        )
        audit.record(
            self.db,
            action=AuditAction.EMPLOYEE_VIEW,
            outcome=AuditOutcome.ALLOWED,
            actor_id=user.employee_id,
            details={"search": {"name": name, "department": department}, "results": len(results)},
        )
        await self.db.commit()
        return results

    async def get_employee(self, user: CurrentUser, employee_id: uuid.UUID) -> EmployeeModel:
        """Another employee's profile: HR sees anyone, a manager sees their
        direct reports. Anyone may ask for themselves."""
        if employee_id == user.employee_id:
            return await self.get_profile(user)
        await ensure_role(
            self.db, user, Role.HR, Role.MANAGER, operation=AuditAction.EMPLOYEE_VIEW.value
        )

        employee = await self.employees.get(employee_id)
        if user.role == Role.HR:
            if employee is None:
                raise NotFound("Employee not found.")
            await self._record_view(user, employee_id, AuditOutcome.ALLOWED)
            return employee

        if employee is None or employee.manager_id != user.employee_id:
            await self._record_view(user, employee_id, AuditOutcome.DENIED, reason="not_found")
            raise NotFound("Employee not found.")
        return employee

    async def _record_view(
        self,
        user: CurrentUser,
        employee_id: uuid.UUID,
        outcome: AuditOutcome,
        reason: str | None = None,
    ) -> None:
        audit.record(
            self.db,
            action=AuditAction.EMPLOYEE_VIEW,
            outcome=outcome,
            actor_id=user.employee_id,
            target_type=AuditTarget.EMPLOYEE,
            target_id=employee_id,
            details={"reason": reason} if reason else None,
        )
        await self.db.commit()
