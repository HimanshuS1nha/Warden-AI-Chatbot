from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from warden.schemas.employee import EmployeeResponse


class LoginRequest(BaseModel):
    email: EmailStr
    # Upper bound stops someone from sending a huge string for argon2 to hash.
    password: str = Field(min_length=1, max_length=128)


# Returned by the login service to the router. The tokens go into cookies;
# only `employee` is sent in the response body.
class LoginResult(BaseModel):
    employee: EmployeeResponse
    access_token: str
    refresh_token: str
    refresh_expires_at: datetime
