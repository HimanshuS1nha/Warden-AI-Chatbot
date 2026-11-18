from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import PayslipModel
from warden.repositories import PayslipRepository


class PayslipService:
    """Payslips are only ever the caller's own. There is no way to ask for
    someone else's, not even for HR."""

    def __init__(self, db: AsyncSession) -> None:
        self.payslips = PayslipRepository(db)

    async def get_mine(self, user: CurrentUser, year: int | None = None) -> Sequence[PayslipModel]:
        """Newest month first; every year unless `year` is given."""
        return await self.payslips.list_for_employee(user.employee_id, year)
