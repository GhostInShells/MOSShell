"""ProjectSubprocessesProvider — project baseline default.

project 层承接 Subprocesses 的接线. 语义 = per-project singleton,
cwd 从 Workspace 派生.

workspace 用户在 ProjectManifest.providers 里显式覆写即可覆盖 default.
"""

from typing import Iterable, Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.subprocesses import Subprocesses, SubprocessFacade
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.core.subprocesses import SubprocessesImpl

__all__ = ["ProjectSubprocessesProvider"]


class ProjectSubprocessesProvider(Provider[Subprocesses]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[Subprocesses]:
        return Subprocesses

    def aliases(self) -> Iterable[Type[Subprocesses]]:
        # 纯 spawn 消费面绑定同一单例: decorator 等消费者只依赖 SubprocessFacade,
        # 生命周期方法在类型层不可见, 结构性杜绝误关共享实例.
        return [SubprocessFacade]

    def factory(self, con: IoCContainer) -> Subprocesses:
        ws = con.get(Workspace)
        logger = con.get(LoggerItf)
        cwd = ws.root().abspath() if ws is not None else None
        return SubprocessesImpl(cwd=cwd, logger=logger)
