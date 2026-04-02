from ghoshell_common.contracts import LoggerItf, config_logger_from_yaml
from ghoshell_container import Provider, IoCContainer
from .workspace import Workspace
from logging import handlers
import logging

__all__ = ["LoggerItf", 'config_logger_from_yaml', 'get_console_logger', 'WorkspaceLoggerProvider']


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


class WorkspaceLoggerProvider(Provider[LoggerItf]):

    def __init__(
            self,
            *,
            name: str = 'moss',
            handler_name: str = 'runtime_logger',
            log_config_file='logging.yaml',
            runtime_log_dir: str = 'logs',
            log_file_name: str = 'moss.log',
            log_when: str = 'd',
            log_interval: int = 1,
            backup_count: int = 5,
    ):
        self.name = name
        self.handler_name = handler_name
        self.runtime_log_dir = runtime_log_dir
        self.log_config_file = log_config_file
        self.log_file_name = log_file_name
        self.log_when = log_when
        self.log_interval = log_interval
        self.backup_count = backup_count

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> LoggerItf:
        workspace = con.get(Workspace)
        if workspace is None:
            # 容错, 如果 workspace 不存在, 则退回到通过 logging 返回日志.
            return logging.getLogger(self.name)

        # 1. 尝试从 YAML 加载全局配置
        config_file = workspace.configs().abspath().joinpath(self.log_config_file)
        if config_file.exists():
            # 注意：config_logger_from_yaml 最好设置 disable_existing_loggers=False
            config_logger_from_yaml(str(config_file.absolute()))

        # 2. 获取 Logger 实例
        logger = logging.getLogger(self.name)

        # 3. 防止重复添加 Handler (关键修复)
        # 检查是否已经有名为 'moss_file_handler' 的处理器，避免多次初始化容器导致日志翻倍
        handler_name = self.handler_name
        if not any(getattr(h, 'name', None) == handler_name for h in logger.handlers):
            # 4. 确定日志文件路径并确保目录存在
            log_dir_storage = workspace.runtime().sub_storage(self.runtime_log_dir)
            log_dir_path = log_dir_storage.abspath()
            log_dir_path.mkdir(parents=True, exist_ok=True)  # 兜底创建

            filename_path = log_dir_path.joinpath(self.log_file_name)

            # 5. 创建并配置 Handler
            file_handler = handlers.TimedRotatingFileHandler(
                filename=str(filename_path),
                when=self.log_when,
                interval=self.log_interval,
                backupCount=self.backup_count,
                encoding='utf-8',  # 建议显式指定编码，防止 Windows 下乱码
            )
            file_handler.name = handler_name  # 给 handler 命名以便检查

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
            )
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
        return logger
