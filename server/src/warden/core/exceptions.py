import logging
from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from warden.core.request_context import REQUEST_ID_HEADER, context_for, use_request_context
from warden.services.errors import Forbidden, NotFound, RuleViolation, ServiceError

# Every error response has the shape {"message": str}. Validation errors add
# an "errors" list with one entry per invalid field. Service errors add a
# "code", and rule violations their "details" (e.g. the available balance).

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Registered for Starlette's HTTPException so it also catches the 404 and
    # 405 Starlette raises for unknown routes and methods. FastAPI's
    # HTTPException is a subclass, so ours land here too.
    message = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    # 4xx responses are client mistakes (bad token, unknown route) and happen
    # all the time, so they're INFO. A 5xx raised on purpose is our problem.
    level = logging.ERROR if exc.status_code >= 500 else logging.INFO
    logger.log(
        level,
        "%s %s -> %d %s",
        request.method,
        request.url.path,
        exc.status_code,
        message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": message},
        # Keeps headers like WWW-Authenticate on 401s.
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
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
    # can be a password, so only location, field and message are returned
    # and logged.
    logger.info("%s %s -> 422 Validation failed: %s", request.method, request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=jsonable_encoder({"message": "Validation failed", "errors": errors}),
    )


def _service_error_status(exc: ServiceError) -> int:
    # Out of scope and missing are both NotFound by the time they get here,
    # so a 404 never tells the caller whether the record exists.
    if isinstance(exc, NotFound):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, Forbidden):
        return status.HTTP_403_FORBIDDEN
    if isinstance(exc, RuleViolation):
        return status.HTTP_422_UNPROCESSABLE_CONTENT
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    status_code = _service_error_status(exc)
    level = logging.ERROR if status_code >= 500 else logging.INFO
    logger.log(level, "%s %s -> %d %s", request.method, request.url.path, status_code, exc.code)
    content: dict[str, object] = {"message": exc.message, "code": exc.code}
    # Details are facts about the caller's own request (their balance, the
    # date that clashed). Not-found and forbidden errors never carry any.
    if isinstance(exc, RuleViolation) and exc.details:
        content["details"] = exc.details
    return JSONResponse(status_code=status_code, content=jsonable_encoder(content))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # The traceback goes to the log, never to the client. Starlette re-raises
    # the exception after this response is sent, so uvicorn logs it a second
    # time as "Exception in ASGI application"; this entry is the one that
    # names the request.
    #
    # This handler runs outside RequestContextMiddleware, so it restores the
    # request's context for the log line and adds the request ID header itself.
    context = context_for(request.scope)
    with use_request_context(context):
        logger.error(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Internal server error"},
        headers={REQUEST_ID_HEADER: context.request_id} if context.request_id else None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ServiceError, service_error_handler)  # type: ignore[arg-type]
    # Covers everything else, including pydantic ValidationError raised by our
    # own code and invalid response models: those are server bugs, not bad input.
    app.add_exception_handler(Exception, unhandled_exception_handler)
