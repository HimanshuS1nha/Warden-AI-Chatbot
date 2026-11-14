from fastapi import APIRouter

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/ping")
async def ping() -> dict[str, str]:
    return {"message": "auth router is up"}
