import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import PayslipModel


class PayslipRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_for_employee(
        self, employee_id: uuid.UUID, year: int | None = None
    ) -> Sequence[PayslipModel]:
        """The employee's payslips, newest month first. All years unless
        `year` is given."""
        query = (
            select(PayslipModel)
            .where(PayslipModel.employee_id == employee_id)
            .order_by(PayslipModel.year.desc(), PayslipModel.month.desc())
        )
        if year is not None:
            query = query.where(PayslipModel.year == year)
        result = await self.db.scalars(query)
        return result.all()
