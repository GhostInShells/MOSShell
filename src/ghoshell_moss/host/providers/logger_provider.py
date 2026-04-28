import logging
from typing import Type

from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.logger import LoggerItf, config_logger_from_yaml, default_logger_formatter
from ghoshell_container import Provider, IoCContainer
from logging.handlers import TimedRotatingFileHandler
from ghoshell_moss.core.blueprint.matrix import Matrix

__all__ = [
    'WorkspaceLoggerProvider',
]


class WorkspaceLoggerProvider(Provider[LoggerItf]):

    def __init__(
            self,
            logger_name: str = '',
            *,
            logger_config_file: str = 'logging.yml',
            moss_file_handler_name: str = 'moss_file_handler',
            log_handler: logging.Handler | None = None,
    ):
        self._logger_name = logger_name
        self._logger_config_file = logger_config_file
        self._moss_file_handler_name = moss_file_handler_name
        self._log_handler = log_handler

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LoggerItf]:
        return LoggerItf

    def factory(self, con: IoCContainer) -> LoggerItf:
        # 强行依赖 workspace.
        ws = con.force_fetch(Workspace)

        logger_name = self._logger_name
        if not logger_name:
            matrix = con.force_fetch(Matrix)
            logger_name = matrix.this.log_name

        if not logger_name.startswith('moss'):
            return logging.getLogger(logger_name)

        # 初始化 moss.
        moss_root_logger = logging.getLogger('moss')
        # 如果有 logging 日志配置, 从配置文件中读取.
        if len(moss_root_logger.handlers) == 0:
            expect_config_file = ws.configs().abspath().joinpath(self._logger_config_file)
            if expect_config_file.exists():
                config_logger_from_yaml(str(expect_config_file))

        has_moss_file_handler = False
        for handler in moss_root_logger.handlers:
            if handler.get_name() == self._moss_file_handler_name:
                has_moss_file_handler = True
                break

        #  注册默认的文件 handler.
        if not has_moss_file_handler:
            handler = self._log_handler
            # default handler
            if handler is None:
                logger_file_name = 'moss.log'
                # 约定的日志存储路径在 workspace/runtime/logs/moss-app-name.log 这样的路径下.
                filename = ws.runtime().sub_storage('logs').abspath().joinpath(logger_file_name)
                handler = TimedRotatingFileHandler(
                    filename=str(filename),
                    when='d',
                    interval=1,
                    backupCount=5,
                )
                handler.set_name(self._moss_file_handler_name)
                handler.setLevel(logging.INFO)
                handler.setFormatter(default_logger_formatter())
            moss_root_logger.addHandler(handler)
        return logging.getLogger(logger_name)
