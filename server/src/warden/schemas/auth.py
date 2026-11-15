from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, StringConstraints

from warden.schemas.user import UserResponse


class RegisterRequest(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
    email: EmailStr
    # Upper bound stops someone from sending a huge string for argon2 to hash.
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


# Returned by the login service to the router. The tokens go into cookies;
# only `user` is sent in the response body.
class LoginResult(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    refresh_expires_at: datetime
