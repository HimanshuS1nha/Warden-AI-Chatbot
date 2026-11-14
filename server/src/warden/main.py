from fastapi import FastAPI

app = FastAPI(title="Warden")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from Warden"}
