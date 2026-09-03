import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml
from typing_extensions import Self

from ghoshell_moss.contracts.logger import get_moss_logger

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Thinking
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.ground import DefaultGroundSet, Ground
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher

    from ._ego import DoloresConfig, DoloresEgo, DoloresEgoConfig
    from ._meta import DoloresMeta
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["Dolores"]

from ._prompts import dolores_inception, dolores_protocol_notice, dolores_terminology


class Dolores(Ghost):
    """Dolores — 第二个 Ghost 原型运行时 (DSH 推理中枢集成).

    生命周期里挂 DshLauncher (推理中枢, 经 matrix.processes) + DefaultGroundSet
    (ghost_home 认知场), 持有 ego (DoloresEgo — 会话/交易窄桥). think() 委托
    ego.run_thinking() 驱动 dsh 推理, logos 逐段 yield, articulator 本侧管理.
    交易协议 (thinking/enter B 范式) 见 _run.py + plugin.ts 头注释. 能力演进
    (Memento / interleaved thinking / ghost 反身 channel / 模型自感知) 逐步接入.
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
        # launcher / ground 懒构建 — __init__ 不碰 httpx / matrix.processes / shell (构造无副作用).
        self._dsh_launcher: "DshLauncher | None" = None
        self._ground_set: DefaultGroundSet | None = None
        self._root_ground: Ground | None = None
        # ground 渲染文本缓存 — __aenter__ 里异步渲染, memories() 同步读取.
        self._ground_text: str | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ego: "DoloresEgo | None" = None
        self._facade: "MShellContextFacade | None" = None

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        """instruction = baseline + 原型元信息 + 身份描述 + 术语 + 协议段 + dolores 层.

        baseline 来自 factory 从 container 取的 base_instruction (CTML + project + mode).
        ghost 段从结构化 meta 派生, 不写死提示词 — 未来认知从目录构建.
        术语段/协议段 (fence 语义) 不可配置; dolores 层可经 ego config inception_template 替换.
        全部是静态段 (cache 稳定) — 不进 steer (尾部), 不进 per-frame 注入.
        """
        parts: list[str] = []
        if self._base_instruction:
            parts.append(self._base_instruction)
        parts.append(self._meta.prototype_instruction())
        parts.append(self._meta.identity_instruction())
        parts.append(dolores_terminology())
        parts.append(dolores_protocol_notice())
        parts.append(self._dolores_instruction())
        return "\n\n".join(parts)

    def _dolores_instruction(self) -> str:
        """dolores 人格/礼仪层 — ego config 有模板文件则替换, 槽位注入运行时路径."""
        template: str | None = None
        if self._home is not None:
            rel = self._load_ego_config().inception_template
            if rel:
                path = self._home / rel
                if path.exists():
                    template = path.read_text(encoding="utf-8")
                else:
                    self.logger.warning(
                        "inception_template %s not found in ghost home; using default", rel
                    )
        env = self._matrix.env if self._matrix is not None else None
        return dolores_inception(
            ghost_home=str(self._home) if self._home is not None else "",
            project_home=str(env.project_path) if env is not None else "",
            mode_home=str(env.mode_home) if env is not None else "",
            template=template,
        )

    async def ground_instruction(self) -> str | None:
        """ground 槽位 — 渲染持有的 root ground (ghost_home 认知场) 为文本.

        root ground 在 __aenter__ 打开并长期持有, 保住 snapshot 变更跟踪 (单 owner).
        由 epoch 周期调用 (Dolores Ego 装线), 本步只备元件, 不主动接入.
        """
        if self._root_ground is None:
            return None
        view = await self._root_ground.render()
        return str(view)

    def memories(self) -> list[Message]:
        """Ghost 的动态记忆 — ground 渲染为第一条 (存在主义, 最前).

        ground 文本在 __aenter__ 里异步渲染后缓存到 _ground_text; 本方法同步读缓存,
        供 ego 经闭包在 create_session 时取最新记忆. clone 复用同一闭包共享认知.
        """
        if self._ground_text:
            return [Message.new(tag="ground").with_content(self._ground_text)]
        return []

    async def think(self, thinking: Thinking) -> AsyncIterator[str]:
        """委托 ego.run_thinking() 驱动 dsh 推理 — 生命周期/收线/CTML 解析全归 run.

        本侧只做 async with 边界 + logos 透传 (给 mindflow 广播观测面).
        异常 (enter/消费/cancel) 经 async with 自然传播, 由 run.__aexit__ 治理.
        """
        if self._ego is not None:
            async with self._ego.run_thinking(thinking) as run:
                async for delta in run.logos():
                    yield delta
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
        # 先打开并长期持有 root ground (ghost_home 认知场). stubs 同步在前 (GROUND.md 已落),
        # GroundSet 由 exit stack 管理生命周期; memory 的 ground 段需在 ego 创建前渲染.
        if self._home is not None:
            self._ground_set = await self._exit_stack.enter_async_context(
                DefaultGroundSet(workspace_root=self._home)
            )
            self._root_ground = await self._ground_set.open(self._home)
        # 渲染 ground 文本, 缓存供 memories() 同步读取 (ego create_session 经闭包消费).
        self._ground_text = await self.ground_instruction()
        if self._matrix is not None:
            await self._exit_stack.enter_async_context(self._dsh())
            # ego 装线: 创建并持有 ego session (经 plugin RPC), 晚于 dsh 就绪.
            # 依赖倒置: ego 不 back-ref ghost, 运行上下文经 ctx/launcher/memories 闭包注入.
            from ._ego import DoloresEgo, DoloresEgoContext
            from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

            self._facade = MShellContextFacade(self._shell)
            ctx = DoloresEgoContext(
                project_home=self._home,
                project_name=self._matrix.env.project_name,
                name=self._meta.name(),
                instruction=self.system_prompt(),
                facade=self._facade,
            )
            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(
                    launcher=self.dsh_launcher,
                    ctx=ctx,
                    config=self._load_ego_config(),
                    memories=self.memories,
                )
            )
            # 绑定自醒 signal 出口到 MOSS session — matrix.session.add_signal 路由到 mindflow.
            self._ego.bind_signal_broadcast(self._matrix.session.add_signal)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._facade is not None:
            self._facade.discard()
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # ── dsh 启动 ───────────────────────────────────

    @property
    def dsh_launcher(self) -> "DshLauncher":
        """dsh launcher 句柄 — 未进入生命周期时抛清晰错误."""
        if self._dsh_launcher is None:
            raise RuntimeError("dsh launcher not started. Call __aenter__ first.")
        return self._dsh_launcher

    @property
    def logger(self) -> logging.Logger:
        """MOSS runtime logger — 经 matrix (从属当前节点) 拿; 无 matrix 时 fallback."""
        if self._matrix is not None:
            return self._matrix.logger
        return get_moss_logger()

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
