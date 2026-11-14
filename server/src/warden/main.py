from fastapi import FastAPI

from warden.routers import auth

app = FastAPI(title="Warden")

app.include_router(auth.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from Warden"}
