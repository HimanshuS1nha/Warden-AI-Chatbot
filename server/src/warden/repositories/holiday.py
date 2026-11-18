from collections.abc import Sequence
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import HolidayModel


class HolidayRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_for_year(self, year: int) -> Sequence[HolidayModel]:
        """The year's public holidays, in date order."""
        result = await self.db.scalars(
            select(HolidayModel)
            .where(HolidayModel.date >= date(year, 1, 1), HolidayModel.date <= date(year, 12, 31))
            .order_by(HolidayModel.date)
        )
        return result.all()
