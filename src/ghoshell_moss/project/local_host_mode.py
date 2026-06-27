from pathlib import Path
import logging

from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.blueprint.cell import CellRegistry
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import HostMode, HostModeMeta, ModeManifests
from ghoshell_moss.contracts.workspace import LocalWorkspace
from ghoshell_moss.project.cell_registry import ProjectCellRegistry
from ghoshell_moss.project.manifests.mode_impl import ScannedModeManifests

__all__ = ['LocalHostMode']


class LocalHostMode(HostMode):

    def __init__(
            self,
            env: Environment,
            meta: HostModeMeta,
            workspace_dir: Path,
            *,
            logger: logging.Logger | None = None,
    ):
        self._env = env
        self._meta = meta
        self._workspace_dir = workspace_dir
        self._workspace = LocalWorkspace(workspace_dir)
        self._logger = logger

        # 懒加载: 仅当 bootstrap() 调用后, manifests/cells 才可用.
        self._manifests: ModeManifests | None = None
        self._cells: CellRegistry | None = None

    # -- 基础属性 -- #

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def meta(self) -> HostModeMeta:
        return self._meta

    @property
    def workspace_dir(self) -> Path:
        return self._workspace_dir

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def logger(self) -> logging.Logger:
        if self._logger is not None:
            return self._logger
        return logging.getLogger('moss')

    # -- bootstrap 守卫 -- #

    def _check_bootstrapped(self) -> None:
        """bootstrap() 未调用时拒绝对外提供 manifests / cells, 避免静默失效."""
        # bootstrap() 通过 setattr(self, '_bootstrapped', True) 标记
        if not hasattr(self, '_bootstrapped') or not getattr(self, '_bootstrapped', False):
            raise RuntimeError(
                f"Mode '{self.name}' is not bootstrapped. Call bootstrap() first."
            )

    # -- 能力声明 -- #

    def manifests(self) -> ModeManifests:
        self._check_bootstrapped()
        if self._manifests is None:
            self._manifests = ScannedModeManifests()
        return self._manifests

    # -- cell 发现 -- #

    def cells(self) -> CellRegistry:
        self._check_bootstrapped()
        if self._cells is None:
            self._cells = ProjectCellRegistry(
                self._env,
                self.cells_discover_paths(),
            )
        return self._cells
