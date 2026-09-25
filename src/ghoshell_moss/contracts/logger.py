"""Logging contract — logger utilities and formatter helpers."""

from ghoshell_common.contracts import LoggerItf, config_logger_from_yaml
import logging

__all__ = [
    "LoggerItf", 'config_logger_from_yaml', 'get_console_logger',
    "get_moss_logger", "default_logger_formatter",
    "MOSS_FILE_HANDLER_NAME", "bind_moss_file_handler",
]

MOSS_FILE_HANDLER_NAME = 'moss_file_handler'


def get_moss_logger() -> LoggerItf:
    return logging.getLogger('moss')


def default_logger_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
    )


def bind_moss_file_handler(
        logger: logging.Logger,
        log_file,
        *,
        handler_name: str = MOSS_FILE_HANDLER_NAME,
) -> None:
    """把运行时文件 handler 绑到 logger 上 — 幂等 (按 handler name 去重).

    logging.yml 只配格式/等级, 文件路径与轮换策略是运行时确定的, 由这里统一:
    TimedRotatingFileHandler(when='midnight', backupCount=5) → log_file.

    Project (首选装配点) 与 MatrixLoggerProvider (兜底) 都走这个函数, 共享同一
    handler name 做幂等去重, 避免第二份真源 (曾因此漏改 when='d').
    """
    from logging.handlers import TimedRotatingFileHandler

    for h in logger.handlers:
        if h.get_name() == handler_name:
            return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        # 'midnight' 按自然日边界算轮换点; 'd' 取的是 (文件 mtime + 24h),
        # 而每个 moss 命令都是新建 handler 的短命进程 — 每天都用反而永远轮不到.
        when='midnight',
        interval=1,
        backupCount=5,
    )
    handler.set_name(handler_name)
    handler.setLevel(logging.INFO)
    handler.setFormatter(default_logger_formatter())
    logger.addHandler(handler)


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
