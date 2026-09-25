"""MatrixLoggerProvider — matrix baseline default (§ZZ-6).

matrix 层承接 LoggerItf 的接线, 不再走 host layer.

定位: **兜底, 不是首选**. 首选装配点是 `Project._ensure_log_file_handler`
(project.bootstrap 先跑, 已按 per-cell 命名挂好 handler); 本 provider 复用同一个
`bind_moss_file_handler` + 同一个 handler name 做幂等去重, 只在 Project 未装配的
退化路径上兜住, 不再持有第二份轮换/命名真源.
"""

import logging
from typing import Type, Iterable

from ghoshell_container import Provider, IoCContainer, INSTANCE

from ghoshell_moss.contracts.logger import LoggerItf, bind_moss_file_handler
from ghoshell_moss.contracts.workspace import Workspace

__all__ = ["MatrixLoggerProvider"]


class MatrixLoggerProvider(Provider[LoggerItf]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LoggerItf]:
        return LoggerItf

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield logging.Logger

    def factory(self, con: IoCContainer) -> LoggerItf:
        moss_logger = logging.getLogger('moss')

        ws = con.get(Workspace)
        if ws is not None:
            log_dir = ws.runtime().sub_storage('logs').abspath()
            bind_moss_file_handler(moss_logger, log_dir.joinpath('moss.log'))

        return moss_logger
