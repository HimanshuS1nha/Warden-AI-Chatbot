import logging

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str) -> None:
    # Uvicorn only configures its own loggers ("uvicorn", "uvicorn.access",
    # "uvicorn.error"). Without a root handler, INFO logs from our modules are
    # dropped and WARNING and above print with no timestamp or level.
    logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
    # Only our own loggers get the configured level, so third-party libraries
    # stay at WARNING.
    logging.getLogger("warden").setLevel(level.upper())
