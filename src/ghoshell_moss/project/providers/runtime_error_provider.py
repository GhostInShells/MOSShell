"""RuntimeErrorLogProvider — fallback provider for the runtime error collector.

兜底: cell 未走 LocalProject 的早设路径时, 懒建一个 collector 并挂到容器里的 logger.
LocalProject 路径会 eager set 同名 contract, 本 provider 不会触发.
"""
from typing import Type

from ghoshell_container import Provider, IoCContainer, INSTANCE

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.runtime_error import RuntimeErrorLog
from ghoshell_moss.core.runtime_error import RuntimeErrorLogImpl

__all__ = ["RuntimeErrorLogProvider"]


class RuntimeErrorLogProvider(Provider[RuntimeErrorLog]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[RuntimeErrorLog]:
        return RuntimeErrorLog

    def factory(self, con: IoCContainer) -> RuntimeErrorLog:
        impl = RuntimeErrorLogImpl()
        logger = con.get(LoggerItf)
        if logger is not None:
            impl.attach(logger)
        return impl
