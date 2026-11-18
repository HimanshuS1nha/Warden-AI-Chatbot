from typing import Annotated

from fastapi import APIRouter, Query

from warden.core.dependencies import CurrentUserDep
from warden.routers.services import HolidayServiceDep
from warden.schemas.holiday import HolidayResponse

router = APIRouter(prefix="/api/holidays", tags=["holidays"])


@router.get("")
async def list_holidays(
    _user: CurrentUserDep,
    holidays: HolidayServiceDep,
    year: Annotated[int | None, Query(ge=2000, le=2100)] = None,
) -> list[HolidayResponse]:
    """The company's public holidays for `year` (default: this year). Any
    logged-in employee can read them."""
    return [HolidayResponse.model_validate(h) for h in await holidays.list(year)]
