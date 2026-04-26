from typing_extensions import Self

from ghoshell_moss.host.abcd import ToolSet
from ghoshell_moss.host.abcd.host_interface import (
    MossHost, MossMode, MossRuntime,
)
from ghoshell_moss.host.abcd.manifests import Manifests
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.contracts.workspace import LocalWorkspace, Workspace
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.host.abcd.environment import Environment
from ghoshell_moss.host.manifests import PackageManifests, MergedManifests
from ghoshell_moss.host.app_store import HostAppStore
from ghoshell_moss.host.modes import list_modes_from_root_package, new_mode
from ghoshell_moss.host.matrix import HostMatrix
from ghoshell_moss.host.toolset import HostAsToolSet
import logging

__all__ = ['Host']

_host_instance = None


class Host(MossHost):

    def __init__(
            self,
            *,
            env: Environment | None = None,
            mode: MossMode | str | None = None,
            logger: logging.Logger | None = None,
    ):
        self.env = env or Environment.discover()
        self.env.bootstrap()
        self._workspace = LocalWorkspace(self.env.workspace_path)
        if not self._workspace.root_path().exists():
            raise RuntimeError()
        self._env_manifest = PackageManifests.from_environment(self.env)
        self._logger: LoggerItf | None = logger

        self._env_modes = {mode.name: mode for mode in list_modes_from_root_package()}
        moss_mode = mode
        if moss_mode is None:
            moss_mode = self.env.moss_mode
        if isinstance(moss_mode, str):
            moss_mode_name = moss_mode
            moss_mode = self._env_modes.get(moss_mode_name)
            if moss_mode is None:
                raise RuntimeError(f"Unknown mode: {moss_mode}")
        self._moss_mode: MossMode = moss_mode
        self._manifest = MergedManifests([self._env_manifest, self._moss_mode.manifest])
        # 获取一个用来做环境发现的 apps.
        # 创建 container, 但是先不启动它.
        self._app_store = HostAppStore(
            env=self.env,
            workspace=self._workspace,
            namespace="MOSS/app_store/toolset",
            runnable=False,
        )
        self._matrix = HostMatrix(
            mode=self._moss_mode,
            env=self.env,
            manifest=self._manifest,
            app_store=self._app_store,
            workspace=self._workspace,
            logger=self._logger,
        )

    @classmethod
    def discover(cls) -> Self:
        global _host_instance
        if _host_instance is None:
            _host_instance = Host()
        return _host_instance

    @property
    def manifests(self) -> Manifests:
        return self._manifest

    @property
    def mode(self) -> MossMode:
        return self._moss_mode

    def all_modes(self) -> dict[str, MossMode]:
        """
        map all the modes in the environment.
        """
        return self._env_modes

    def new_mode(self, name: str, apps: list[str], bring_up_apps: list[str], description: str = "") -> None:
        """
        create new mode follow convertion
        """
        if name in self._env_modes:
            raise NameError(f"Mode {name} already exists")
        new_mode(name=name, apps=apps, bring_up_apps=bring_up_apps, description=description)

    @property
    def apps(self) -> HostAppStore:
        return self._app_store

    def matrix(self) -> Matrix:
        return self._matrix

    def run_as_toolset(self) -> ToolSet:
        return HostAsToolSet(
            env=self.env,
            workspace=self._workspace,
            mode=self._moss_mode,
            matrix=self._matrix,
        )
