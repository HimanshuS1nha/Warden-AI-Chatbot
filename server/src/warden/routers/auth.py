from fastapi import APIRouter, status

from warden.core.dependencies import DbSession
from warden.schemas.auth import RegisterRequest
from warden.schemas.user import UserResponse
from warden.services.auth import register_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, db: DbSession) -> UserResponse:
    user = await register_user(db, data)
    return UserResponse.model_validate(user)
