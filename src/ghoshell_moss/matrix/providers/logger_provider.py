"""MatrixLoggerProvider — matrix baseline default (§ZZ-6).

matrix 层承接 LoggerItf 的接线, 不再走 host layer.

注: 按 §ZZ-6, matrix 的 logger 反绑是 pull 模式 (Matrix.__aenter__ 里
container.get(LoggerItf) 覆写 self._logger). 本 provider 是 IoC 侧的
默认兜底 — 如果没有它, container.get(LoggerItf) 返回 None, matrix 沿用
自己 __init__ 时的 `moss.cell.{address}` logger.

职责边界:
- logging.yml 的 dictConfig 全局加载是 project.bootstrap() 的唯一职责
- 本 provider 只做 handler 幂等兜底: 确保 moss logger 上有 TimedRotatingFileHandler
- project.bootstrap 先跑先挂 handler, 本 provider 后续发现已有同名 handler 则直接返回
"""

import logging
from typing import Type, Iterable

from ghoshell_container import Provider, IoCContainer, INSTANCE
from logging.handlers import TimedRotatingFileHandler

from ghoshell_moss.contracts.logger import LoggerItf, default_logger_formatter
from ghoshell_moss.contracts.workspace import Workspace

__all__ = ["MatrixLoggerProvider"]


class MatrixLoggerProvider(Provider[LoggerItf]):

    def __init__(
            self,
            *,
            handler_name: str = 'moss_file_handler',
            log_handler: logging.Handler | None = None,
            log_file_name: str = 'moss.log',
    ):
        self._handler_name = handler_name
        self._log_handler = log_handler
        self._log_file_name = log_file_name

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LoggerItf]:
        return LoggerItf

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield logging.Logger

    def factory(self, con: IoCContainer) -> LoggerItf:
        ws = con.get(Workspace)
        if ws is None:
            return logging.getLogger('moss')

        moss_logger = logging.getLogger('moss')

        # 幂等: 已有同名 handler 则不重复添加 (§ZZ-6 python logging 坑规避)
        for h in moss_logger.handlers:
            if h.get_name() == self._handler_name:
                return moss_logger

        # 挂 TimedRotatingFileHandler → {workspace}/runtime/logs/moss.log
        handler = self._log_handler
        if handler is None:
            log_dir = ws.runtime().sub_storage('logs').abspath()
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = log_dir.joinpath(self._log_file_name)
            handler = TimedRotatingFileHandler(
                filename=str(filename),
                when='d',
                interval=1,
                backupCount=5,
                encoding='utf-8',
            )
            handler.set_name(self._handler_name)
            handler.setLevel(logging.INFO)
            handler.setFormatter(default_logger_formatter())

        moss_logger.addHandler(handler)
        return moss_logger
