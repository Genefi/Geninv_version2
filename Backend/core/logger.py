import logging
import sys

from core.environment import settings


def configure_uvicorn_logger():
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").handlers.clear()
    get_logger("uvicorn")
    get_logger("uvicorn.access")


def get_logger(name: str = "geninv_app"):
    logger = logging.getLogger(name)
    log_level = (
        logging.INFO
        if settings.ENVIRONMENT.lower() in ["production", "staging"]
        else logging.DEBUG
    )
    logger.setLevel(log_level)
    stream_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s] [%(funcName)s]: %(message)s"
    )
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    return logger


app_logger = get_logger()


__all__ = ["app_logger", "configure_uvicorn_logger"]
