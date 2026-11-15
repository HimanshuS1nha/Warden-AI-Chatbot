from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Every error response has the shape {"message": str}. Validation errors add
# an "errors" list with one entry per invalid field.


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Registered for Starlette's HTTPException so it also catches the 404 and
    # 405 Starlette raises for unknown routes and methods. FastAPI's
    # HTTPException is a subclass, so ours land here too.
    message = (
        exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": message},
        # Keeps headers like WWW-Authenticate on 401s.
        headers=exc.headers,
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        location, *path = error["loc"]
        message: str = error["msg"]
        # Pydantic prefixes messages from custom validators with this.
        message = message.removeprefix("Value error, ")
        errors.append(
            {
                "location": location,  # body, query, path, header or cookie
                # For malformed JSON the path is a character offset, not a field.
                "field": None
                if error["type"] == "json_invalid"
                else ".".join(str(part) for part in path) or None,
                "message": message,
            }
        )
    # Pydantic's raw errors include the submitted value ("input"), which
    # can be a password, so only location, field and message are returned.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=jsonable_encoder({"message": "Validation failed", "errors": errors}),
    )


async def unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
    # Never send internal details to the client. Starlette re-raises the
    # exception after this response is sent, so the server still logs the
    # full traceback; logging here as well would print it twice.
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Internal server error"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    # Covers everything else, including pydantic ValidationError raised by our
    # own code and invalid response models: those are server bugs, not bad input.
    app.add_exception_handler(Exception, unhandled_exception_handler)
