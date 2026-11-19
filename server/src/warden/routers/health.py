from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check for container health checks and uptime monitors. Needs no
    login and doesn't touch the database."""
    return {"status": "ok"}


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from Warden"}
