import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, contains_eager, joinedload

from warden.models import DepartmentModel, EmployeeModel

DEFAULT_SEARCH_LIMIT = 50


def escape_like(value: str) -> str:
    """Escape LIKE wildcards so user input matches literally. Use with
    `escape="\\\\"`."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class EmployeeRepository:
    """Employees, always loaded with their department."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, employee_id: uuid.UUID) -> EmployeeModel | None:
        """Any employee by ID, active or not. Who may see them is the
        service's decision."""
        return await self.db.scalar(
            select(EmployeeModel)
            .options(joinedload(EmployeeModel.department))
            .where(EmployeeModel.id == employee_id)
        )

    async def get_manager(self, employee_id: uuid.UUID) -> EmployeeModel | None:
        """The employee's manager, or None at the top of the hierarchy."""
        report = aliased(EmployeeModel)
        return await self.db.scalar(
            select(EmployeeModel)
            .join(report, report.manager_id == EmployeeModel.id)
            .options(joinedload(EmployeeModel.department))
            .where(report.id == employee_id)
        )

    async def list_reports(self, manager_id: uuid.UUID) -> Sequence[EmployeeModel]:
        """Active direct reports of the manager, by name."""
        result = await self.db.scalars(
            select(EmployeeModel)
            .options(joinedload(EmployeeModel.department))
            .where(EmployeeModel.manager_id == manager_id, EmployeeModel.is_active.is_(True))
            .order_by(EmployeeModel.name, EmployeeModel.id)
        )
        return result.all()

    async def search(
        self,
        name: str | None = None,
        department: str | None = None,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> Sequence[EmployeeModel]:
        """Active employees whose name contains `name` and whose department is
        named `department`, both case-insensitive. Either filter may be left
        out. Ordered by name, at most `limit` rows."""
        query = (
            select(EmployeeModel)
            .join(EmployeeModel.department)
            .options(contains_eager(EmployeeModel.department))
            .where(EmployeeModel.is_active.is_(True))
            .order_by(EmployeeModel.name, EmployeeModel.id)
            .limit(limit)
        )
        if name and name.strip():
            query = query.where(
                EmployeeModel.name.ilike(f"%{escape_like(name.strip())}%", escape="\\")
            )
        if department and department.strip():
            query = query.where(
                DepartmentModel.name.ilike(escape_like(department.strip()), escape="\\")
            )
        result = await self.db.scalars(query)
        return result.all()
