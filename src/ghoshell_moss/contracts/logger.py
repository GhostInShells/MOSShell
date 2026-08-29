"""Logging contract — logger utilities and formatter helpers."""

from ghoshell_common.contracts import LoggerItf, config_logger_from_yaml
import logging

__all__ = [
    "LoggerItf", 'config_logger_from_yaml', 'get_console_logger',
    "get_moss_logger", "default_logger_formatter",
]


def get_moss_logger() -> LoggerItf:
    return logging.getLogger('moss')


def default_logger_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
    )


def get_console_logger(level=logging.ERROR, name: str = "ghost"):
    """
    quickly get console logger for debugging purposes
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s  - %(filename)s:%(lineno)d ")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
