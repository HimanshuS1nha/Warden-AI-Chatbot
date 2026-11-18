import uuid
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class PayslipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    year: int
    month: int
    net_pay: Decimal
