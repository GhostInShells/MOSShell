import logging

__all__ = ["get_console_logger"]


def get_console_logger(level=logging.ERROR):
    logger = logging.getLogger("moss")
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s  - %(filename)s:%(lineno)d ")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
