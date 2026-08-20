import asyncio
import contextlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml
from typing_extensions import Self

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.core.blueprint.session import Session

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher

    from ._meta import DoloresMeta

__all__ = ["Dolores"]


class Dolores(Ghost):
    """Dolores — 第二个 Ghost 原型运行时.

    生命周期里挂一个 DshLauncher (DSH 推理中枢), 直接持有 matrix 的治理链
    (matrix.processes). articulate() 尚未接入 DSH 推理内核, 固定返回占位输出.
    后续逐步接入: DSH agent-loop 推理、Memento 持久化轨迹、interleaved
    thinking、ghost 反身 channel、模型自感知 (_llms).
    """

    def __init__(
        self,
        *,
        meta: "DoloresMeta",
        home: Path | None = None,
        session: Session | None = None,
        matrix: Matrix | None = None,
    ):
        self._meta = meta
        self._home = home
        self._session = session
        self._matrix = matrix
        # launcher 懒构建 — __init__ 不碰 httpx / matrix.processes (构造无副作用).
        self._dsh_launcher: "DshLauncher | None" = None
        self._exit_stack = contextlib.AsyncExitStack()

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        return ""

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        yield "hello world"

    async def __aenter__(self) -> Self:
        await self._exit_stack.__aenter__()
        # 文件 IO 卸载到 thread; session.output 留在主 loop (避免跨线程).
        action = await asyncio.to_thread(self._sync_stubs)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        if self._matrix is not None:
            await self._exit_stack.enter_async_context(self._dsh())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # ── dsh 启动 ───────────────────────────────────

    @property
    def dsh_launcher(self) -> "DshLauncher":
        """dsh launcher 句柄 — 未进入生命周期时抛清晰错误."""
        if self._dsh_launcher is None:
            raise RuntimeError("dsh launcher not started. Call __aenter__ first.")
        return self._dsh_launcher

    def _build_dsh_launcher(self) -> "DshLauncher":
        from ghoshell_moss.deepseek_harness.launcher import (
            DshLauncher,
            DshLauncherConfig,
        )

        config = self._load_config()
        dsh = config.get("dsh") or {}
        home = self._resolve_dsh_home(dsh.get("home"))
        launcher_config = DshLauncherConfig(**{**dsh, "home": home})
        return DshLauncher(launcher_config, subprocesses=self._matrix.processes)

    @contextlib.asynccontextmanager
    async def _dsh(self):
        launcher = self._build_dsh_launcher()
        self._dsh_launcher = launcher
        launcher.on_exit(self._on_dsh_exit)
        self._session.output("system", log="starting dsh")
        async with launcher:
            self._session.output(
                "system",
                f"dsh ready at {launcher.config.base_url}",
                log="dsh ready",
            )
            yield

    def _on_dsh_exit(self, exit_info) -> None:
        if self._session is None:
            return
        if exit_info.self_shutdown or exit_info.exit_code in (0, None):
            self._session.output("system", log="dsh exited")
        else:
            self._session.output(
                "error",
                log=f"dsh exited code {exit_info.exit_code}: {exit_info.stderr}",
            )

    # ── stubs 同步 ─────────────────────────────────

    def _sync_stubs(self) -> str | None:
        """同步骨架到 ghost home. 返回 'init' | 'override' | None(no-op).

        VERSION 一致时不动; 缺失时 init, 不一致时 override (全量覆盖骨架文件,
        不触碰 home 里的动态数据文件). 同步时 materialize dirs + dsh_home.
        """
        if self._home is None:
            return None
        marker = self._home / ".dolores.yml"
        current = self._read_version(marker)
        target = self._meta.VERSION
        if current == target:
            return None
        action = "override" if current is not None else "init"
        shutil.copytree(self._meta.stubs_dir(), self._home, dirs_exist_ok=True)
        self._materialize_dirs()
        self._sync_dsh_home()
        self._write_version(marker, target)
        return action

    def _load_config(self) -> dict:
        marker = self._home / ".dolores.yml"
        if not marker.exists():
            return {}
        return yaml.safe_load(marker.read_text(encoding="utf-8")) or {}

    def _materialize_dirs(self) -> None:
        for d in self._load_config().get("dirs") or []:
            (self._home / d).mkdir(parents=True, exist_ok=True)

    def _sync_dsh_home(self) -> None:
        shutil.copytree(
            self._meta.dsh_stubs_dir(),
            self._home / ".dsh",
            dirs_exist_ok=True,
        )

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)

    @staticmethod
    def _read_version(marker: Path) -> str | None:
        if not marker.exists():
            return None
        data = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
        return data.get("version")

    @staticmethod
    def _write_version(marker: Path, version: str) -> None:
        data = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
        data["version"] = version
        marker.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
