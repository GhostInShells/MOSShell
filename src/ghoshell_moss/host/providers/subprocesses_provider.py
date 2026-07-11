"""Subprocesses IoC Provider — host 默认接线.

singleton=True: Matrix 一份, 消费者从 IoC fetch. cwd/output_dir 从 Workspace 派生
(TT-6 边界做成环境). Matrix 通过 lifecycle_level_contracts 接入 async lifecycle.
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider, INSTANCE

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl

__all__ = ["HostSubprocessesProvider"]


class HostSubprocessesProvider(Provider[Subprocesses]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[Subprocesses]:
        return Subprocesses

    def factory(self, con: IoCContainer) -> Subprocesses:
        ws = con.get(Workspace)
        logger = con.get(LoggerItf)
        if ws is not None:
            runtime = ws.runtime().sub_storage("subprocesses").abspath()
            cwd = ws.root().abspath()
            output_dir = runtime
        else:
            cwd = None
            output_dir = None
        return SubprocessesImpl(cwd=cwd, output_dir=output_dir, logger=logger)
