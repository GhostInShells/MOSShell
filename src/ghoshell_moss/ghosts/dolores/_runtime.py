import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from typing_extensions import Self

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.core.blueprint.session import Session

if TYPE_CHECKING:
    from ._meta import DoloresMeta

__all__ = ["Dolores"]


class Dolores(Ghost):
    """Dolores — 第二个 Ghost 原型运行时.

    骨架阶段: articulate() 尚未接入 DSH 推理内核, 固定返回 "hello world".
    后续逐步接入: DSH agent-loop 推理、Memento 持久化轨迹、interleaved
    thinking、ghost 反身 channel、模型自感知 (_llms).
    """

    def __init__(
        self,
        *,
        meta: "DoloresMeta",
        home: Path | None = None,
        session: Session | None = None,
    ):
        self._meta = meta
        self._home = home
        self._session = session

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        return ""

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        yield "hello world"

    async def __aenter__(self) -> Self:
        # 文件 IO 卸载到 thread; session.output 留在主 loop (避免跨线程).
        action = await asyncio.to_thread(self._sync_stubs)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    # ── stubs 同步 ─────────────────────────────────

    def _sync_stubs(self) -> str | None:
        """同步原型 stubs 到 ghost home. 返回 'init' | 'override' | None(no-op).

        VERSION 一致时不动; 缺失时 init, 不一致时 override (全量覆盖 stubs 文件,
        不触碰 home 里的动态数据文件).
        """
        if self._home is None:
            return None
        marker = self._home / ".dolores"
        current = self._read_version(marker)
        target = self._meta.VERSION
        if current == target:
            return None
        action = "override" if current is not None else "init"
        shutil.copytree(self._meta.stubs_dir(), self._home, dirs_exist_ok=True)
        marker.write_text(f"VERSION={target}\n", encoding="utf-8")
        return action

    @staticmethod
    def _read_version(marker: Path) -> str | None:
        if not marker.exists():
            return None
        for line in marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("VERSION="):
                return line[len("VERSION="):].strip()
        return None
