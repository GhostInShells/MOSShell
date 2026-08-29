"""
ProjectNodeManager — NodeManager 的只读 inventory 实现.

扫描治理域领地内的 NODE.md 声明, 只回答 "领地里可以拉起什么".
不拉起、不杀灭、不持有任何运行时状态.
"""
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from ghoshell_moss.core.blueprint.cell import (
    CellRuntimeInfo, ExecSpec, MatchPattern, NodeLauncher, NodeManager,
    NodeManifest, NodeProbeError, ProjectRelativePath,
)
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.contracts.subprocesses import (
    CaptureSpec, ManagedProcess, ProcessMeta, SubprocessFacade,
)

__all__ = ['ProjectNodeManager']

_KILL_GRACE_SECONDS = 3.0
"""SIGTERM → wait → SIGKILL 的宽限窗口, kill/prune 共用."""


class ProjectNodeManager(NodeManager):
    """扫描指定目录集合, 生成 project-relative 路径到 NodeManifest 的字典视图."""

    # 扫描时跳过的目录名 (常见非 cell 目录, 减少无谓 IO).
    _SKIP_DIRS: set[str] = {'__pycache__', 'node_modules'}

    def __init__(
            self,
            env: Environment,
            node_dirs: list[Path],
            subprocesses: SubprocessFacade | None = None,
            logger: logging.Logger | None = None,
    ):
        self._env = env
        # 默认扫描根目录集合. list_nodes(paths=...) 可 override.
        self._node_dirs = node_dirs
        self._cache: dict[ProjectRelativePath, NodeManifest] | None = None
        self._subprocesses = subprocesses
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

    @property
    def subprocesses(self) -> SubprocessFacade:
        from ghoshell_moss.core.subprocesses import SubprocessesImpl
        if self._subprocesses is None:
            self._subprocesses = SubprocessesImpl(cwd=self._env.project_path)
        return self._subprocesses

    def list_nodes(
            self,
            refresh: bool = True,
            *,
            paths: list[Path] | None = None,
            installed: bool | None = None,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[ProjectRelativePath, NodeManifest]:
        # paths 显式给定 → override, 不写缓存 (缓存只服务默认根目录集合).
        # 默认路径下的 refresh 才刷缓存.
        if paths is not None:
            scanned = self._scan(paths)
        else:
            if refresh or self._cache is None:
                self._cache = self._scan(self._node_dirs)
            scanned = self._cache.copy()

        # installed=None → 全部返回 (含未安装, 让 CLI 能提示 INSTALL.md 路径, WW-5 故事 3).
        if installed is not None:
            scanned = {k: v for k, v in scanned.items() if v.installed is installed}
        if include or exclude:
            scanned = dict(self.match_nodes(scanned, include=include, exclude=exclude))
        return scanned

    def get_node(self, relative_path: str | Path) -> NodeManifest | None:
        relative_path = str(relative_path)
        path = self._env.project_path / relative_path
        if path.is_dir():
            return NodeManifest.read_from_directory(path)
        return None

    def get_node_launcher(self, relative_path: str | Path) -> NodeLauncher | None:
        # 薄组合: manifest 反查 → 装 launcher (build_node + dump_cell_env + argv 解析
        # 全部在 NodeLauncher.from_manifest 内). launcher 消费者 (Subprocesses.execute)
        # 负责起进程 + 回填 pid/pgid.
        manifest = self.get_node(relative_path)
        if manifest is None:
            return None
        return NodeLauncher.from_manifest(self._env, manifest)

    def _scan(self, roots: list[Path]) -> dict[ProjectRelativePath, NodeManifest]:
        result: dict[ProjectRelativePath, NodeManifest] = {}
        project_root = self._env.project_path
        for root in roots:
            if not root.is_dir():
                continue
            self._walk(root, project_root, result)
        return result

    def _walk(
            self,
            directory: Path,
            project_root: Path,
            result: dict[ProjectRelativePath, NodeManifest],
    ) -> None:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('.') or entry.name in self._SKIP_DIRS:
                continue
            node_md = entry / NodeManifest.MANIFEST_FILENAME
            if node_md.is_file():
                try:
                    manifest = NodeManifest.read_from_file(node_md)
                except Exception:
                    manifest = None
                if manifest is not None:
                    relative = str(entry.relative_to(project_root))
                    result[relative] = manifest
                    continue  # node 目录内部不再递归 (一个目录一个身份锚)
            self._walk(entry, project_root, result)

    async def spawn_node(
            self,
            manifest: NodeManifest,
            *,
            extra_env: dict[str, str] | None = None,
            capture: CaptureSpec | None = None,
    ) -> ManagedProcess:
        """
        拉起一个 node cell — spawn 咽喉 (唯一入口).

        只做: installed 校验 → NodeLauncher 打包 → probe 闸门 → Subprocesses.execute 拉起.
        不做: singleton 锁 / ledger 写入与清理 / pid·pgid 回填 — 那些归 enter_cell_lifecycle
        (cell 自身宣告) 或 matrix 治理层. spawner 只负责把进程生出来.

        probe (manifest.check) 失败时抛 NodeProbeError (携带 broken reason), 不拉起主脚本.
        """
        if not manifest.installed:
            install_path = Path(manifest.file).parent / NodeManifest.INSTALL_FILENAME
            raise RuntimeError(
                f"node {manifest.name!r} not installed. See {install_path} for install steps."
            )

        launcher = NodeLauncher.from_manifest(self._env, manifest)

        broken = await self._run_probe(manifest, launcher)
        if broken is not None:
            raise NodeProbeError(broken)

        child_env = dict(launcher.env)
        if manifest.exec.env:
            child_env.update(manifest.exec.env)
        if extra_env:
            child_env.update(extra_env)

        managed = await self.subprocesses.execute(
            *launcher.run,
            name=f'cell:{manifest.name}',
            description=manifest.description or f'node cell {manifest.name}',
            cwd=Path(launcher.runtime.cell.home),
            extra_env=child_env,
            with_os_env=False,  # launcher.env 已包含必要 env
            capture=capture,
            on_exit=self._on_node_exit(launcher.runtime),
        )
        return managed

    async def _run_probe(
            self,
            manifest: NodeManifest,
            launcher: NodeLauncher,
    ) -> str | None:
        """跑 manifest.check 探针 — 返回 broken reason (str) 或 None (通过).

        探针是独立进程, 语言无关, 目标零配合 (不逼对方走到 Matrix.__aenter__).
        只用 exit code: 0 → 通过; nonzero → 携带 stderr (或 stdout) 作为 broken reason.
        不发明 ready 状态机, 不加 CellRuntimeInfo 字段 (§ FEATURE node-lifecycle probe).
        """
        check = manifest.check
        if check is None:
            return None

        argv = self._probe_argv(check)
        probe_env = dict(launcher.env)
        if check.env:
            probe_env.update(check.env)

        managed = await self.subprocesses.execute(
            *argv,
            name=f'probe:{manifest.name}',
            description=f'pre-launch probe for {manifest.name}',
            cwd=manifest.cwd,
            extra_env=probe_env,
            with_os_env=False,
            capture=CaptureSpec(buffer_lines=200),
        )
        meta = await self._await_exit(managed)
        if meta.exit_code == 0:
            return None
        stderr = stdout = ''
        if managed.output is not None:
            stderr = (managed.output.stderr() or '').strip()
            stdout = (managed.output.stdout() or '').strip()
        reason = stderr or stdout
        return reason or f'probe exited with code {meta.exit_code}'

    @staticmethod
    def _probe_argv(check: ExecSpec) -> list[str]:
        argv = []
        command = check.command
        if command:
            if command == 'python':
                command = sys.executable
            argv.append(command)
        argv.extend(check.arguments)
        return argv

    @staticmethod
    async def _await_exit(managed: ManagedProcess) -> ProcessMeta:
        """等子进程退出并返回其 meta — 规避 reclaim 协程尚未更新 exit_code 的竞态.

        add_done_callback 在 fire_exit (exit_code 已更新) 时触发, 故以其为序.
        """
        fut = asyncio.get_running_loop().create_future()

        def _cb(m: ProcessMeta) -> None:
            if not fut.done():
                fut.set_result(m)

        managed.add_done_callback(_cb)
        return await fut

    def _on_node_exit(
            self,
            node: CellRuntimeInfo,
    ) -> Callable[[ProcessMeta], None]:
        """构造 on_exit 回调: 子进程退出时清理工作区 ledger 文件.

        callback 在 asyncio loop 线程触发 (Subprocesses 承诺), 无需线程安全.
        """

        def _callback(meta: ProcessMeta) -> None:
            try:
                node.delete_invalid(self._env.cell_runtimes_dir)
            except Exception:
                self._logger.warning(
                    "failed to clean ledger file for %s", node.address,
                )
            self._logger.info(
                "node cell exited: address=%s exit_code=%s",
                node.address, meta.exit_code,
            )

        return _callback

    # -- runtime 治理: 读账本 / 杀 / 清孤儿 -- #

    def list_runtimes(self) -> list[CellRuntimeInfo]:
        return list(CellRuntimeInfo.iter_runtime_info(self._env.cell_runtimes_dir))

    def get_runtime(self, address: str) -> CellRuntimeInfo | None:
        return CellRuntimeInfo.read_from_runtime_dir(self._env.cell_runtimes_dir, address)

    def kill_cell(self, address: str, *, force: bool = False) -> bool:
        """终止一个 cell 进程 (SIGTERM → grace → SIGKILL) 并清账本.

        账本里没有 = 不属本地治理域, 无操作 (False); 有 = 终止尝试 + 清账 (True).
        """
        info = self.get_runtime(address)
        if info is None:
            return False
        self._terminate(info, force=force)
        info.delete_invalid(self._env.cell_runtimes_dir)
        return True

    def prune(self, *, keep_alive: bool = False, force: bool = False) -> tuple[int, int, int]:
        """清孤儿 runtime 账本. 返回 (removed, killed, skipped).

        默认 kill 活着的孤儿 (它们持有 singleton 锁); keep_alive=True 只删死账本.
        """
        removed = killed = skipped = 0
        for info in self.list_runtimes():
            if info.is_alive():
                if keep_alive:
                    skipped += 1
                    continue
                self._terminate(info, force=force)
                killed += 1
            info.delete_invalid(self._env.cell_runtimes_dir)
            removed += 1
        return removed, killed, skipped

    @staticmethod
    def _signal_target(info: CellRuntimeInfo) -> tuple[str, int] | None:
        """pick 信号目标: pgid (进程组) 优先, 无则 pid, 再无则 None.

        pgid 覆盖 start_new_session 后的整棵子孙树; in-process cell (host/ghost,
        pgid=0) 不能杀继承的 shell 组, 只能落到 pid.
        """
        if info.pgid > 0:
            return 'pgid', info.pgid
        if info.pid > 0:
            return 'pid', info.pid
        return None

    @staticmethod
    def _send_signal(target: tuple[str, int], sig: int) -> bool:
        """向解析出的目标发信号. 目标已消失返回 False."""
        kind, target_id = target
        try:
            if kind == 'pgid':
                os.killpg(target_id, sig)
            else:
                os.kill(target_id, sig)
            return True
        except ProcessLookupError:
            return False

    @classmethod
    def _terminate(cls, info: CellRuntimeInfo, *, force: bool) -> None:
        """SIGTERM + 短 grace → SIGKILL; force=True 直接 SIGKILL."""
        target = cls._signal_target(info)
        if target is None:
            return
        if force:
            cls._send_signal(target, signal.SIGKILL)
            return
        if not cls._send_signal(target, signal.SIGTERM):
            return
        deadline = time.time() + _KILL_GRACE_SECONDS
        while time.time() < deadline:
            if not cls._send_signal(target, 0):  # signal 0 = liveness probe
                return
            time.sleep(0.1)
        cls._send_signal(target, signal.SIGKILL)
