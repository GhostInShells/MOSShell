# IoC Provider manifest — declare "this interface is produced by this factory".
#
# Define module-level Provider instances.  Matrix scans via isinstance(obj, Provider),
# deduplicates by id(), then registers each into the IoC container.
#
# Mode extends by: from MOSS.manifests.providers import *
#
# --
# IoC Provider 声明 — 声明依赖注入工厂。
# 定义模块级 Provider 实例，Matrix 扫描自动发现并注入 IoC 容器。
# Mode 通过 from MOSS.manifests.providers import * 继承全局。

from ghoshell_moss.matrix.providers import (
    ProjectZenohSessionProvider,
    ZenohTopicServiceProvider,
    EnvConfigStoreProvider,
    MatrixLoggerProvider,
)

from ghoshell_moss.core.resources.memory_registry import InMemoryResourceRegistryProvider

# zenoh session provider
moss_session_provider = ProjectZenohSessionProvider()

# file-based config store
config_store_provider = EnvConfigStoreProvider()

# zenoh topic system
topic_service_provider = ZenohTopicServiceProvider()

# workspace logger — returns moss root logger with TimedRotatingFileHandler
logger_provider = MatrixLoggerProvider()

# in-memory resource registry
resources_provider = InMemoryResourceRegistryProvider()
