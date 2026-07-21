"""MatrixSubprocessesProvider — matrix baseline default (§ZZ-2).

matrix 层承接 Subprocesses 的接线, 不再走 host layer. 语义 = per-Matrix singleton,
cwd 从 Workspace 派生 (TT-6 边界做成环境).

workspace 用户在 MatrixManifest.providers 里显式覆写即可覆盖 default.
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl

__all__ = ["MatrixSubprocessesProvider"]


class MatrixSubprocessesProvider(Provider[Subprocesses]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[Subprocesses]:
        return Subprocesses

    def factory(self, con: IoCContainer) -> Subprocesses:
        ws = con.get(Workspace)
        logger = con.get(LoggerItf)
        cwd = ws.root().abspath() if ws is not None else None
        return SubprocessesImpl(cwd=cwd, logger=logger)
