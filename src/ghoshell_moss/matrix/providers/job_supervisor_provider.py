"""MatrixJobSupervisorProvider — matrix baseline default (§ZZ-2).

matrix 层承接 JobSupervisor 的接线, 不再走 host layer. singleton=True: 一个
根实例住 IoC. 消费者拿到后调用 ``.new()`` 派生隔离 peer (§XX-4 判决).

底层依赖 Subprocesses (由 MatrixSubprocessesProvider 提供), factory 内 fetch 组合.
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.job_supervisor import JobSupervisor
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.core.job_supervisor._impl import JobSupervisorImpl

__all__ = ["MatrixJobSupervisorProvider"]


class MatrixJobSupervisorProvider(Provider[JobSupervisor]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[JobSupervisor]:
        return JobSupervisor

    def factory(self, con: IoCContainer) -> JobSupervisor:
        sp = con.force_fetch(Subprocesses)
        logger = con.get(LoggerItf)
        return JobSupervisorImpl(subprocesses=sp, logger=logger)
