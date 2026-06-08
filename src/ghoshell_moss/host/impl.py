from pathlib import Path
from typing_extensions import Self

from ghoshell_moss.core.blueprint.host import (
    MossHost, Mode, MossRuntime, GhostRuntime,
)
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.manifests import Manifests
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.codex.discover import ScanError, ModuleManifest
from ghoshell_moss.contracts.workspace import LocalWorkspace, Workspace
from ghoshell_moss.core.blueprint.environment import Environment, CellMeta
from ghoshell_moss.host.manifests import PackageManifests
from ghoshell_moss.host.app_store import HostAppStore
from ghoshell_moss.host.modes import list_modes_from_root_package, new_mode
from ghoshell_moss.host.ghosts import list_ghosts_from_root_package
from ghoshell_moss.host.matrix import MatrixImpl
from ghoshell_moss.host.moss_runtime import MossRuntimeImpl
from ghoshell_moss.host.ghost_runtime import GhostRuntimeImpl
import importlib
import pathlib

__all__ = ['Host']

_host_instance = None


class Host(MossHost):

    def __init__(
            self,
            *,
            env: Environment | None = None,
            mode: Mode | str | None = None,
            session_scope: str | None = None,
    ):
        self._scan_ghost_errors: list[ScanError] = []
        self._scan_manifest_errors: list[ScanError] = []

        self._env = env or Environment.discover()
        if mode is not None:
            self._env.set_mode(mode if isinstance(mode, str) else mode.name)
        if session_scope is not None:
            self._env.set_session_scope(session_scope)

        self._env.bootstrap()
        self._workspace = LocalWorkspace(self.env.workspace_path)
        if not self._workspace.root_path().exists():
            raise RuntimeError()

        self._env_modes: dict[str, Mode] | None = None
        self._ghosts: dict[str, tuple[GhostMeta, ModuleManifest]] | None = None
        moss_mode = mode
        if moss_mode is None:
            moss_mode = self.env.moss_mode_name
        if isinstance(moss_mode, str):
            moss_mode_name = moss_mode
            moss_mode = self.all_modes().get(moss_mode_name)
            if moss_mode is None:
                raise RuntimeError(f"Unknown mode: {moss_mode}")
        self._moss_mode: Mode = moss_mode
        # manifest 直接取 mode 的——mode 的每个 manifest 文件显式继承全局
        # (from MOSS.manifests.xxx import * + 扩展)，无需运行时合并。
        self._manifest = self._moss_mode.manifest
        # 获取一个用来做环境发现的 apps.
        # 创建 container, 但是先不启动它.
        self._app_store = HostAppStore(
            env=self.env,
            workspace=self._workspace,
            namespace="MOSS/app_store",
            runnable=False,
            bringup=self._moss_mode.bringup_apps,
            include=self._moss_mode.apps,
        )
        self._matrix = MatrixImpl(
            mode=self._moss_mode,
            env=self.env,
            manifest=self._manifest,
            app_store=self._app_store,
            workspace=self._workspace,
        )

    def name(self) -> str:
        return self._env.meta_config.name

    def description(self) -> str:
        return self._env.meta_config.description

    @classmethod
    def discover(cls, env: Environment | None = None) -> Self:
        global _host_instance
        if _host_instance is None:
            _host_instance = Host(env=env)
        return _host_instance

    def reboot(self) -> Self:
        global _host_instance
        _host_instance = None
        new_host = Host(env=self._env)
        _host_instance = new_host
        return new_host

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def manifests(self) -> Manifests:
        return self._manifest

    @property
    def mode(self) -> Mode:
        return self._moss_mode

    @property
    def scan_errors(self) -> list[ScanError]:
        """Aggregated scan errors from manifests, modes, and ghosts discovery."""
        return self._scan_manifest_errors + self._scan_ghost_errors

    def all_modes(self) -> dict[str, Mode]:
        if self._env_modes is None:
            self._env_modes = {
                mode.name: mode for mode in list_modes_from_root_package(
                    strict=False, errors=self._scan_ghost_errors,
                )
            }
        return self._env_modes

    def all_ghost_manifests(self) -> dict[str, tuple[GhostMeta, ModuleManifest]]:
        if self._ghosts is None:
            self._ghosts = list_ghosts_from_root_package(
                strict=False, errors=self._scan_ghost_errors,
            )
        return self._ghosts

    def all_ghosts(self) -> dict[str, GhostMeta]:
        return {name: value[0] for name, value in self.all_ghost_manifests().items()}

    def new_mode(self, name: str, apps: list[str], bringup_apps: list[str], description: str = "") -> Path:
        """
        create new mode follow convertion
        """
        if name in self.all_modes():
            raise NameError(f"Mode {name} already exists")
        return new_mode(name=name, apps=apps, bring_up_apps=bringup_apps, description=description)

    def apps(self) -> HostAppStore:
        return self._app_store

    def matrix(self) -> Matrix:
        return self._matrix

    def list_scope_cells(self, alive_only: bool = True) -> list[CellMeta]:
        return self._env.list_scope_cells(alive_only)

    def kill_cell(self, address: str) -> bool:
        return self._env.kill_cell(address)

    def kill_all_cells(self) -> list[str]:
        return self._env.kill_all_cells()

    def scan_manifest_errors(self) -> list[ScanError]:
        return self._scan_manifest_errors

    def scan_ghost_errors(self) -> list[ScanError]:
        return self._scan_ghost_errors

    def run(
            self,
            *,
            run_shell: bool = True,
            name: str | None = None,
            description: str | None = None,
    ) -> MossRuntime:
        return MossRuntimeImpl(
            env=self.env,
            workspace=self._workspace,
            mode=self._moss_mode,
            matrix=self._matrix,
            run_shell_on_start=run_shell,
        )

    def run_ghost(
            self,
            ghost: str | GhostMeta,
            *,
            run_shell: bool = True,
    ) -> GhostRuntime:
        """创建 GhostRuntime — 编排 MossRuntime + Ghost 生命周期.

        Args:
            ghost: ghost 名称 (从 all_ghosts() 查找) 或 GhostMeta 实例.
                   传入实例时环境无关，可用于测试.
            run_shell: 传递给 MossRuntime.
        """
        if isinstance(ghost, str):
            ghost_meta, _manifest = self.all_ghost_manifests().get(ghost)
            if ghost_meta is None:
                raise KeyError(f"Ghost '{ghost}' not found in workspace")
            manifest: ModuleManifest = _manifest
            module = importlib.import_module(manifest.module_path)
            source_path = pathlib.Path(module.__file__).parent.absolute()
        elif isinstance(ghost, GhostMeta):
            ghost_meta = ghost
            source_path = None
        else:
            raise ValueError(f'invalid ghost argument type {type(ghost)}')

        # env.ghost_name 必须在 self.run() 之前设置: Matrix.ghost_home
        # 在 MossRuntime.__aenter__ 期间从 env.ghost_name 解析,
        # GhostWorkspace.home → matrix.ghost_home → ghosts/{name}/.
        # 不设则解析为 ghosts/None/, soul 加载失败.
        self.env.set_ghost_name(ghost_meta.name())
        moss_runtime = self.run(run_shell=run_shell)
        return GhostRuntimeImpl(
            moss_runtime=moss_runtime,
            ghost_meta=ghost_meta,
            source_path=source_path,
        )
