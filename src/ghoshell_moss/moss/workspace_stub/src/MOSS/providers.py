from ghoshell_moss.contracts.logger import WorkspaceLoggerProvider
from ghoshell_moss.contracts.configs import WorkspaceYamlConfigStoreProvider

"""
本文件存放 MOSS 指定模式的进程级别
"""

# default logger
logger_provider = WorkspaceLoggerProvider(
    name='moss',
    default_handler_name='runtime_log',
    log_config_file='logging.yaml',
    log_file_name='moss.log',
    log_when='d',
    log_interval=1,
    backup_count=5,
)


# 配置文件的读取模块.
# 默认从 [workspace]/configs  下读取 yaml 类型的配置文件.
config_store_provider = WorkspaceYamlConfigStoreProvider()