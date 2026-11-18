from datetime import date

from pydantic import BaseModel, ConfigDict


class HolidayResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    name: str
