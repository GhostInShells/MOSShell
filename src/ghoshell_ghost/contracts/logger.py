from ghoshell_common.contracts import LoggerItf, config_logger_from_yaml
import logging

__all__ = ["LoggerItf", 'config_logger_from_yaml', 'get_console_logger']


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
