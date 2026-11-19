"""JSON logs, one object per line, for every logger in the process.

Our modules keep using `logging.getLogger(__name__)`. structlog's
ProcessorFormatter renders their records, and uvicorn's, as JSON with a
timestamp, level, logger name and the current request ID.
"""

import logging

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from warden.core.request_context import get_request_context

HANDLER_NAME = "warden-json"

# Uvicorn gives these loggers their own handlers and stops propagation.
# configure_logging sends them through the root handler instead.
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def add_request_id(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    request_id = get_request_context().request_id
    if request_id is not None:
        event_dict.setdefault("request_id", request_id)
    return event_dict


_SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    add_request_id,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def json_formatter() -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        # Runs on records from stdlib loggers. Records from structlog loggers
        # already went through the same processors.
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # The traceback as one string. dict_tracebacks can include local
            # variables, which may hold passwords or tokens.
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )


def configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(json_formatter())

    root = logging.getLogger()
    # Replace our handler on repeat calls (create_app in tests) but leave
    # handlers added by others, such as pytest's log capture, alone.
    for existing in [h for h in root.handlers if h.get_name() == HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    # Only our own loggers get the configured level, so third-party libraries
    # stay at WARNING.
    logging.getLogger("warden").setLevel(level.upper())

    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
