import signal
from pathlib import Path
from typing import Iterable, Iterator

from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.blueprint.cell import CellRuntimeInfo, NodeManager
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.project import (
    Project, HostMode, Manifest, MatrixManifest, HostModeMeta, HOST_MODE_FILE,
)
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.contracts.workspace import LocalWorkspace
from ghoshell_moss.core.subprocesses._utils import killpg
from ghoshell_moss.project.node_manager import ProjectNodeManager
from ghoshell_moss.project.local_host_mode import LocalHostMode
from ghoshell_moss.project.manifests.ghosts import search_ghost_manifests
from ghoshell_moss.project.manifests.impl import ScannedMatrixManifest
from ghoshell_moss.project.manifests.base import ScannedManifest

__all__ = ['LocalProject']


class LocalProject(Project):

    def __init__(self, env: Environment):
        self._env = env
        self._workspace = LocalWorkspace(self._env.workspace_path)

        self._cells: NodeManager | None = None
        self._ghosts_cache: dict[str, tuple[Path, GhostMeta]] | None = None
        self._modes_cache: dict[str, tuple[Path, Manifest[HostModeMeta]]] | None = None
        self._matrix_manifests: MatrixManifest | None = None

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def workspace(self) -> Workspace:
        return self._workspace

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

    # -- cells -- #

    @property
    def nodes(self) -> NodeManager:
        if self._cells is None:
            try:
                mode = self.current_mode()
                cell_dirs = mode.cells_discover_paths() if mode else self._env.cell_dirs()
            except Exception:
                cell_dirs = self._env.cell_dirs()
            self._cells = ProjectNodeManager(self._env, cell_dirs=cell_dirs)
        return self._cells

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

    def matrix_manifests(self) -> MatrixManifest:
        # 单例缓存 — scanner 是 lazy generator, 重复构造无副作用, 但缓存避免
        # 每次 fetch 都重新扫包.
        if self._matrix_manifests is None:
            self._matrix_manifests = ScannedMatrixManifest(
                self._env.moss_meta.matrix_manifest_package,
            )
        return self._matrix_manifests
