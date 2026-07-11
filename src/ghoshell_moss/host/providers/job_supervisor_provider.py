"""JobSupervisor IoC Provider — host 默认接线.

singleton=True: 一个"根实例"住 IoC. 消费者拿到后调用 ``.new()`` 派生隔离 peer
(内部依赖引用共享, 状态独立), owner 自负 async with. 这条路径避免了
``container.make(kwargs=)`` 的自解释性丢失.

底层依赖 Subprocesses (由 HostSubprocessesProvider 提供), factory 内 fetch 组合.
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.job_supervisor import JobSupervisor
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.core.job_supervisor._impl import JobSupervisorImpl

__all__ = ["HostJobSupervisorProvider"]


class HostJobSupervisorProvider(Provider[JobSupervisor]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[JobSupervisor]:
        return JobSupervisor

    def factory(self, con: IoCContainer) -> JobSupervisor:
        sp = con.force_fetch(Subprocesses)
        logger = con.get(LoggerItf)
        return JobSupervisorImpl(subprocesses=sp, logger=logger)
