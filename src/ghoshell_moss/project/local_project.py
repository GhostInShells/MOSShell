import signal
from pathlib import Path
from typing import Iterable, Iterator

from ghoshell_container import IoCContainer, Container, Provider

from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.blueprint.cell import CellRuntimeInfo, NodeManager
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.project import (
    Project, HostMode, Manifest, ProjectManifest, HostModeMeta, HOST_MODE_FILE,
)
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.contracts.workspace import LocalWorkspace
from ghoshell_moss.core.subprocesses import killpg
from ghoshell_moss.project.node_manager import ProjectNodeManager
from ghoshell_moss.project.local_host_mode import LocalHostMode
from ghoshell_moss.project.manifests.ghosts import search_ghost_manifests
from ghoshell_moss.project.manifests.impl import ScannedProjectManifest
from ghoshell_moss.project.manifests.base import ScannedManifest
from ghoshell_moss.contracts import ConfigInstanceRegisterBootstrapper
from ghoshell_moss.contracts.resource import ResourceStorageFactoryBootstrapper

__all__ = ['LocalProject']


class LocalProject(Project):

    def __init__(self, env: Environment):
        self._env = env
        self._workspace = LocalWorkspace(self._env.workspace_path)

        self._nodes: NodeManager | None = None
        self._ghosts_cache: dict[str, tuple[Path, GhostMeta | Exception]] | None = None
        self._modes_cache: dict[str, tuple[Path, Manifest[HostModeMeta]]] | None = None
        self._project_manifests: ProjectManifest | None = None
        self._container: IoCContainer | None = None

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def container(self) -> IoCContainer:
        if self._container is not None:
            return self._container
        container = Container(name=f"moss/project/{self.env.project_name}/{self.env.project_id}")
        self._container = container
        container.set(Environment, self._env)
        container.set(Project, self)
        container.set(Workspace, self.workspace)

        project_manifests = self.project_manifests()
        for manifest in project_manifests.providers():
            if manifest.is_error():
                raise RuntimeError(f"Project provider manifest error: {manifest.error()} at {manifest.found_at()}")
            provider = manifest.value()
            if not isinstance(provider, Provider):
                raise RuntimeError(f"Project provider manifest `{provider}` is not a Provider at {manifest.found_at()}")
            # override default providers
            container.register(provider)

        for provider in self._default_providers():
            contract = provider.contract()
            if not container.bound(contract):
                container.register(provider)

        # -- configs -- #
        configs = []
        for config_manifest in project_manifests.configs():
            if config_manifest.is_error():
                raise RuntimeError(
                    "Config Manifest error: %s at %s" % (config_manifest.error(), config_manifest.found_at())
                )
            configs.append(config_manifest.value())
        if len(configs) > 0:
            bootstrapper = ConfigInstanceRegisterBootstrapper(*configs)
            container.add_bootstrapper(bootstrapper)

        # -- resources (project_manifests.resources → bootstrapper) -- #
        for resource_manifest in project_manifests.resources():
            if resource_manifest.is_error():
                raise RuntimeError(
                    "Resource Manifest error: %s at %s" % (resource_manifest.error(), resource_manifest.found_at()),
                )
            storage_factory = resource_manifest.value()
            bootstrapper = ResourceStorageFactoryBootstrapper(storage_factory)
            container.add_bootstrapper(bootstrapper)

        return container

    def _default_providers(self) -> Iterable[Provider]:
        """
        project 层的 default 接线 (default 兜底).

        workspace 用户在 ProjectManifest.providers 里显式覆写即可覆盖.
        driver-specific 的 default (topic/session/zenoh.Session) 归 adapter.
        """
        from ghoshell_moss.project.providers.configs_provider import EnvConfigStoreProvider
        from ghoshell_moss.project.providers.subprocesses_provider import ProjectSubprocessesProvider
        from ghoshell_moss.project.providers.job_supervisor_provider import ProjectJobSupervisorProvider
        from ghoshell_moss.project.providers.llms_provider import ProjectLLMFuncsProvider
        from ghoshell_moss.core.resources.memory_registry import InMemoryResourceRegistryProvider

        yield ProjectSubprocessesProvider()
        yield ProjectJobSupervisorProvider()
        yield EnvConfigStoreProvider()
        yield InMemoryResourceRegistryProvider()
        yield ProjectLLMFuncsProvider()

    # -- ghosts -- #

    def ghosts(self) -> Iterable[tuple[Path, GhostMeta | Exception]]:
        if self._ghosts_cache is None:
            self._scan_ghosts()
        for path, meta in self._ghosts_cache.values():
            yield path, meta

    def get_ghost(self, name: str) -> GhostMeta:
        if self._ghosts_cache is None:
            self._scan_ghosts()
        if name not in self._ghosts_cache:
            raise LookupError(f"Ghost '{name}' not found in project")
        _, meta = self._ghosts_cache[name]
        if isinstance(meta, Exception):
            raise meta
        return meta

    def _scan_ghosts(self):
        self._ghosts_cache = {}
        for manifest in search_ghost_manifests():
            if manifest.is_error():
                self._ghosts_cache[manifest.name()] = (manifest.found_at(), manifest.error())
            else:
                self._ghosts_cache[manifest.name()] = (manifest.found_at(), manifest.value())

    # -- modes -- #

    def list_modes(self) -> Iterable[tuple[Path, Manifest[HostModeMeta]]]:
        if self._modes_cache is None:
            self._scan_modes()
        yield from self._modes_cache.values()

    def get_mode(self, mode_name: str) -> HostMode:
        if self._modes_cache is None:
            self._scan_modes()
        if mode_name not in self._modes_cache:
            raise LookupError(f"Mode '{mode_name}' not found in project")
        _, manifest = self._modes_cache[mode_name]
        if manifest.is_error():
            raise manifest.error() or LookupError(f"Mode '{mode_name}' scan error")
        meta = manifest.value()
        return LocalHostMode(
            env=self._env,
            meta=meta,
            workspace_dir=Path(meta.file).parent,
        )

    def _scan_modes(self):
        self._modes_cache = {}
        modes_dir = self.modes_home
        if not modes_dir.is_dir():
            return
        for entry in sorted(modes_dir.iterdir()):
            if not entry.is_dir():
                continue
            host_md = entry / HOST_MODE_FILE
            if not host_md.is_file():
                continue
            try:
                meta = HostModeMeta.from_file(host_md, name=entry.name)
            except Exception as e:
                manifest = ScannedManifest(
                    name=entry.name,
                    found_at=host_md,
                    error=e,
                )
            else:
                manifest = ScannedManifest(
                    name=meta.name,
                    value=meta,
                    found_at=host_md,
                    description=meta.description,
                )
            self._modes_cache[manifest.name()] = (host_md, manifest)

    # -- nodes -- #

    @property
    def nodes(self) -> NodeManager:
        if self._nodes is None:
            try:
                mode = self.current_mode()
                node_dirs = mode.nodes_discover_paths() if mode else self._env.node_dirs()
            except Exception as e:
                self.logger.exception("Failed to discover nodes: %s", e)
                node_dirs = self._env.node_dirs()
            self._nodes = ProjectNodeManager(self._env, node_dirs=node_dirs)
        return self._nodes

    def cell_runtimes(self) -> Iterator[CellRuntimeInfo]:
        # 直接读文件系统, 不做活性核对 — 活性判断由调用方按 info.is_alive() 自负.
        # 目录不存在时 (从未 spawn 过 cell) 返回空迭代, 不引 FileNotFoundError.
        runtime_dir = self._env.cell_runtimes_dir
        if not runtime_dir.is_dir():
            return
        yield from CellRuntimeInfo.iter_runtime_info(runtime_dir)

    def kill_cell(self, address: str) -> bool:
        # 孤儿清理: 尝试对本 project ledger 里的 cell 进程发 signal + 清账本.
        # ledger 里没有 = 不属本地治理域, 无操作 (契约 False).
        # ledger 里有 = 属本地, 无论进程还活着与否都要清账本 (契约 True).
        runtime_dir = self._env.cell_runtimes_dir
        info = CellRuntimeInfo.read_from_runtime_dir(runtime_dir, address)
        if info is None:
            return False
        if info.is_alive() and info.pgid:
            # 对进程组发 SIGTERM — 孤儿场景没有 owner 走优雅退出, SIGTERM 一发即杀.
            # killpg 内部吞 ProcessLookupError, 我们不 care 是否真正落地.
            killpg(info.pgid, signal.SIGTERM)
        info.delete_invalid(runtime_dir)
        return True

    # -- matrix manifests -- #

    def project_manifests(self) -> ProjectManifest:
        # 单例缓存 — scanner 是 lazy generator, 重复构造无副作用, 但缓存避免
        # 每次 fetch 都重新扫包.
        if self._project_manifests is None:
            self._project_manifests = ScannedProjectManifest(
                self._env.moss_meta.matrix_manifest_package,
            )
        return self._project_manifests
