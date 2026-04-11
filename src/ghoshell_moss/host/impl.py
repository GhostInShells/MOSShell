from ghoshell_moss.host.abcd.host_interface import (
    MossHost, MossMode, MossRuntime,
)
from ghoshell_moss.host.abcd.manifests import Manifest
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.contracts.workspace import LocalWorkspace, Workspace
from ghoshell_container import Container
from .environment import Environment
from .manifests import PackageManifest, MergedManifest
from .app_store import HostAppStore
from .modes import list_modes_from_root_package, new_mode


class Host(MossHost):

    def __init__(
            self,
            *,
            env: Environment | None = None,
            mode: MossMode | str | None = None,
    ):
        self.env = env or Environment.discover()
        self.env.bootstrap()
        self._workspace = LocalWorkspace(self.env.workspace_path)
        if not self._workspace.root_path().exists():
            raise RuntimeError()
        self._env_manifest = PackageManifest.from_environment(self.env)

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
        self._manifest = MergedManifest([self._env_manifest, self._moss_mode.manifest])
        self._container = self._prepare_container()
        # 获取一个用来做环境发现的 apps.
        # 创建 container, 但是先不启动它.
        self._app_store = HostAppStore(
            env=self.env,
            workspace=self._workspace,
            namespace="MOSS/apps",
        )

    def _prepare_container(self) -> Container:
        container = Container(name="MOSS/host")
        container.set(MossHost, self)
        container.set(Host, self)
        container.set(Environment, self.env)
        container.set(LocalWorkspace, self._workspace)
        container.set(Workspace, self._workspace)

        for contract in self._env_manifest.contracts():
            # register provider from manifest.contracts.
            container.register(contract.provider)
        return container

    @property
    def manifest(self) -> Manifest:
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
        pass

    def run(self, *, mode: MossMode | str = 'default', session_id: str = 'default') -> MossRuntime:
        pass
