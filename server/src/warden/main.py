from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from warden.core.config import get_settings
from warden.core.exceptions import register_exception_handlers
from warden.core.logging import configure_logging
from warden.routers import auth

settings = get_settings()

configure_logging(settings.log_level)

app = FastAPI(title="Warden")

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from Warden"}
