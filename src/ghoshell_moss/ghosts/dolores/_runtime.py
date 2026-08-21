import asyncio
import contextlib
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml
from typing_extensions import Self

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.ground import DefaultGroundSet, Ground
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher

    from ._ego import DoloresConfig, DoloresEgo, DoloresEgoConfig
    from ._meta import DoloresMeta

__all__ = ["Dolores"]


class Dolores(Ghost):
    """Dolores — 第二个 Ghost 原型运行时.

    生命周期里挂一个 DshLauncher (DSH 推理中枢), 直接持有 matrix 的治理链
    (matrix.processes); 挂一个 ShellTrajectory (观测轨迹, 上下文来源).
    articulate() 现阶段只把 trajectory 帧写进 output, 模型驱动是下一步槽位
    (DSH agent-loop 推理、pydantic-ai). 后续逐步接入: Memento 持久化轨迹、
    interleaved thinking、ghost 反身 channel、模型自感知 (_llms).
    """

    def __init__(
        self,
        *,
        meta: "DoloresMeta",
        home: Path | None = None,
        session: Session | None = None,
        matrix: Matrix | None = None,
        shell: MOSShell | None = None,
        base_instruction: str | None = None,
    ):
        self._meta = meta
        self._home = home
        self._session = session
        self._matrix = matrix
        self._shell = shell
        self._base_instruction = base_instruction
        # launcher / trajectory / ground 懒构建 — __init__ 不碰 httpx / matrix.processes / shell (构造无副作用).
        self._dsh_launcher: "DshLauncher | None" = None
        self._trajectory: MShellTrajectory | None = None
        self._epoch_started = False
        self._ground_set: DefaultGroundSet | None = None
        self._root_ground: Ground | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ego: "DoloresEgo | None" = None

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        """instruction = baseline (MossSystemPrompter.base_instruction) + 原型元信息 + 身份描述.

        baseline 来自 factory 从 container 取的 base_instruction (CTML + project + mode).
        两个 ghost 段从结构化 meta 派生, 不写死提示词 — 未来认知从目录构建.
        """
        parts: list[str] = []
        if self._base_instruction:
            parts.append(self._base_instruction)
        parts.append(self._meta.prototype_instruction())
        parts.append(self._meta.identity_instruction())
        return "\n\n".join(parts)

    async def ground_instruction(self) -> str | None:
        """ground 槽位 — 渲染持有的 root ground (ghost_home 认知场) 为文本.

        root ground 在 __aenter__ 打开并长期持有, 保住 snapshot 变更跟踪 (单 owner).
        由 epoch 周期调用 (Dolores Ego 装线), 本步只备元件, 不主动接入.
        """
        if self._root_ground is None:
            return None
        view = await self._root_ground.render()
        return str(view)

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        """上下文完全由 ShellTrajectory 承载 — 先写 trajectory frame, 再走 output.

        Moment 体系不动: 本步只把 shell 观测 (facade / status / events / context) 从
        perspectives 挪到 trajectory 帧, Moment 只承载外部输入 (percepts + hint).
        模型驱动委托给 ego.run() (DSH 推理中枢 transaction) — logos 流逐段 yield.
        """
        trajectory = self._trajectory
        if trajectory is not None:
            if not self._epoch_started:
                self._epoch_started = True
                # 首个 epoch: 注入全量 facade (refresh 重置观测基线).
                epoch_start = trajectory.epoch_start_point(refresh=True)
                if self._session is not None:
                    self._session.output(
                        "trajectory",
                        Message.new().with_content(epoch_start),
                        log="trajectory epoch start",
                    )
            # 每轮: 拉当前帧 delta (events + status + context + facade) → output 观测面.
            frame = trajectory.pop_frame()
            if self._session is not None:
                self._session.output(
                    "trajectory",
                    *frame.project(now=time.time()),
                    log=f"trajectory frame {frame.index}",
                )

        # 模型驱动: 委托 ego.run() (DSH 推理中枢 transaction), logos 流逐段 yield.
        if self._ego is not None:
            async for text in self._ego.run():
                yield text
        else:
            yield ""

    async def __aenter__(self) -> Self:
        await self._exit_stack.__aenter__()
        # 文件 IO 卸载到 thread; session.output 留在主 loop (避免跨线程).
        action = await asyncio.to_thread(self._sync_stubs)
        # plugin.ts 每次 override (活跃开发件, 不受 VERSION 门控), 保证最新插件进 ghost home.
        await asyncio.to_thread(self._sync_dsh_plugin)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        if self._matrix is not None:
            await self._exit_stack.enter_async_context(self._dsh())
            # ego 装线: 创建并持有 ego session (经 plugin RPC), 晚于 dsh 就绪.
            from ._ego import DoloresEgo

            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(self, self._load_ego_config())
            )
        # 挂载 ShellTrajectory — shell 由 MossRuntime 持有且已 running (ghost.__aenter__
        # 晚于 shell 启动). 未提供 shell / 未运行时跳过, trajectory 保持 None.
        if self._shell is not None and self._shell.is_running():
            self._trajectory = await self._exit_stack.enter_async_context(
                MShellTrajectory(self._shell)
            )
        # 打开并长期持有 root ground (ghost_home 认知场). stubs 同步在前 (GROUND.md 已落),
        # GroundSet 由 exit stack 管理生命周期, root ground 单 owner 持有供 epoch 渲染.
        if self._home is not None:
            self._ground_set = await self._exit_stack.enter_async_context(
                DefaultGroundSet(workspace_root=self._home)
            )
            self._root_ground = await self._ground_set.open(self._home)
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

    @property
    def trajectory(self) -> MShellTrajectory:
        """ShellTrajectory 句柄 — 未挂载 (无 shell / 未 running) 时抛清晰错误."""
        if self._trajectory is None:
            raise RuntimeError(
                "trajectory not mounted. Requires a running shell. Call __aenter__ first."
            )
        return self._trajectory

    def _build_dsh_launcher(self) -> "DshLauncher":
        from ghoshell_moss.deepseek_harness.launcher import DshLauncher

        dsh = self._load_config().dsh
        home = self._resolve_dsh_home(dsh.home)
        launcher_config = dsh.model_copy(update={"home": home})
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
        target = self._meta.VERSION
        current = self._load_config().version
        if current == target:
            return None
        action = "override" if current else "init"
        shutil.copytree(self._meta.stubs_dir(), self._home, dirs_exist_ok=True)
        self._materialize_dirs()
        self._sync_dsh_home()
        self._write_version(target)
        return action

    def _load_config(self) -> "DoloresConfig":
        from ._ego import DoloresConfig

        marker = self._home / ".dolores.yml"
        if not marker.exists():
            return DoloresConfig()
        data = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
        return DoloresConfig(**data)

    def _write_version(self, version: str) -> None:
        """写回 version (stubs 同步标记), 其余配置从当前文件重载后原样保留."""
        config = self._load_config()
        config.version = version
        marker = self._home / ".dolores.yml"
        marker.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", exclude_defaults=True, exclude_none=True),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _load_ego_config(self) -> "DoloresEgoConfig":
        return self._load_config().ego

    def _materialize_dirs(self) -> None:
        for d in self._load_config().dirs:
            (self._home / d).mkdir(parents=True, exist_ok=True)

    def _sync_dsh_home(self) -> None:
        shutil.copytree(
            self._meta.dsh_stubs_dir(),
            self._home / ".dsh",
            dirs_exist_ok=True,
        )

    def _sync_dsh_plugin(self) -> None:
        """复制 plugin.ts 到 ghost home — 每次 override, 不随 VERSION 门控.

        plugin.ts 是活跃开发件 (dsh 内核特权桥), 改动频率远高于骨架文件;
        骨架 (GROUND.md / .dolores.yml) 才版本门控, plugin 每次启动拉最新.
        """
        if self._home is None:
            return
        target = self._home / ".dsh" / "profiles" / "web" / "plugin.ts"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._meta.dsh_plugin_stub(), target)

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)
