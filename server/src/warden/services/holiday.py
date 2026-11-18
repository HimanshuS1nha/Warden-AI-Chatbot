from collections.abc import Callable, Sequence
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import HolidayModel
from warden.repositories import HolidayRepository
from warden.services.leave import company_today


class HolidayService:
    """Company holidays are public to every employee, so there's no user."""

    def __init__(self, db: AsyncSession, *, today: Callable[[], date] = company_today) -> None:
        self.holidays = HolidayRepository(db)
        self.today = today

    async def list(self, year: int | None = None) -> Sequence[HolidayModel]:
        """The year's holidays in date order (default: this year)."""
        return await self.holidays.list_for_year(year or self.today().year)
