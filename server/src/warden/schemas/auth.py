from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, StringConstraints


class RegisterRequest(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
    email: EmailStr
    # Upper bound stops someone from sending a huge string for argon2 to hash.
    password: str = Field(min_length=8, max_length=128)
