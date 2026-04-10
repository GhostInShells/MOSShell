from ghoshell_moss.host.abcd.host_interface import (
    IHost, Mode, IRuntime,
)
from ghoshell_moss.host.abcd.manifests import Manifest
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.contracts.workspace import LocalWorkspace
from ghoshell_container import IoCContainer, Container
from .environment import Environment
from .manifests import PackageManifest


class Host(IHost):

    def __init__(
            self,
            *,
            env: Environment | None = None,
            mode: Mode | str = 'default',
    ):
        self._env = env or Environment.discover()
        self._env.bootstrap()
        self._mode = mode
        self._workspace = LocalWorkspace(self._env.workspace_path)
        if not self._workspace.root_path().exists():
            raise RuntimeError()
        self._container = Container(name="MOSS/host")
        self._env_manifest = PackageManifest.from_environment(self._env)

    @property
    def manifest(self) -> Manifest:
        return self._env_manifest

    def list_modes(self) -> dict[str, Mode]:
        pass

    def matrix(self) -> Matrix:
        pass

    def run(self, *, mode: Mode | str = 'default', session_id: str = 'default') -> IRuntime:
        pass
