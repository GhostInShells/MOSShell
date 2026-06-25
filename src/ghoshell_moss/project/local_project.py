from pathlib import Path
from typing import Iterable

from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.blueprint.cell import CellRegistry
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.project import Project, HostMode, Manifest, ModeMeta
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.contracts.workspace import LocalWorkspace

__all__ = ['LocalProject']


class LocalProject(Project):

    def __init__(self, env: Environment):
        self._env = env
        self._workspace = LocalWorkspace(self._env.workspace_path)

    @property
    def env(self) -> Environment:
        return self._env

    def ghosts(self) -> Iterable[tuple[Path, GhostMeta | Exception]]:
        pass

    def get_ghost(self, name: str) -> GhostMeta:
        pass

    def list_modes(self) -> Iterable[tuple[Path, Manifest[ModeMeta]]]:
        pass

    def get_mode(self, mode_name: str) -> HostMode:
        pass

    @property
    def cells(self) -> CellRegistry:
        pass

    @property
    def workspace(self) -> Workspace:
        return self._workspace
