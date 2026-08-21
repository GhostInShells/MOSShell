"""DoloresEgo — Dolores 的自我/连续性层 (表面草稿, 未实现).

本文件只画 Python 侧的表面 (interface contract) + 记录方案与待讨论点,
实现留待下一步. 与 dsh_plugin/moss-dolores-ghost-plugin.ts 并行推进 —
本文件负责 ghost 侧, TS 负责 dsh 内核侧. 所有方法体均为占位 (``...``).

── 定位 ──────────────────────────────────────────────────────────────
DoloresEgo 是 Dolores ghost 的 "我". 随 ghost 创建时实例化, 进入同一
生命周期. 它把散在 Dolores 本体上的会话响应逻辑收敛成一个有边界的自我,
负责 epoch 驱动 (deferred) + 会话事件响应 + 与 DSH 推理中枢的窄桥.

── 两条线 ────────────────────────────────────────────────────────────
长命线 (随 ghost 生命周期):
  背景 watcher 只盯 turn/start 一种事件, 每次往一个固定 nucleus 打一个
  一次性 impulse. 这是自驱心跳 — mindflow 静默时也能被唤醒, 走正常
  challenge → attention → articulate 路径, 不用等外部输入.

短命线 (每次 articulate 一次):
  ``run(moment, effort)`` 是 Dolores.articulate 的委托. 内部独立
  AsyncExitStack, 启停走 RPC, 生命周期与 articulator 严格同周期.
  ``async for`` 的作用域就是 transaction 边界:
  enter = open ego session (preStep lock), exit = close.

── transaction / RPC 旁路 ────────────────────────────────────────────
articulate 进入/退出各触发一次与 plugin 的 HTTP 通讯, 开放/关闭 ego session
的 preStep 锁. 触发 ego session 运行走 RPC 旁路 (参数组织 + steer), 而不是
正常 user prompt — effort 等参数无法经 user prompt 传.

── session event 响应 (run 时监听) ───────────────────────────────────
  1. ego tool 调用: 模型 (DSH) 调 ego tool → dsh 发 tool/call event → 本侧
     监听 → 执行真实逻辑 → RPC 回调 (tool-result, 关联 callId) → 解锁 plugin.
     关联键 = tool/call 的 callId (plugin 侧 pendingCalls/arrivedResults 双 map).
  2. thinking start/end: 发 dolores 自己的 topic.
  3. 通用 dict: 所有 session event 用 dolores ego topic 包一个通用 dict 发送 (异步).
  4. logos 判定: 从事件流挑出 logos 字符串对外 return (generator yield).
  5. 异常感知: 持有异常, 下一次请求时作为上下文注入.

── moment 字段取舍 ───────────────────────────────────────────────────
  percepts + hint    — 核心输入, run 每轮喂给 DSH 的新内容.
  command_logos      — 不在 run 面. ghost_runtime 已在 articulate 之前
                       send_nowait 消费 (反射弧, ghost_runtime.py:390).
  thinking_effort    — 不在 moment 上, 在 articulator 上; =='none' 已被上游
                       短路 (ghost_runtime.py:399). 由 articulate 拆出作 run 第二参.
  perspectives       — 被 trajectory 取代 (moss_dynamic). 其它 gate 语境
                       (如 safemode) 按需经 percept/signal 进 ego, 不走
                       perspectives 通道.

── 待讨论 (seams) ────────────────────────────────────────────────────
  1. RPC 协议面: open/close session、params+steer、tool-result 的入参/出参
     形状. 先立本文件 interface, TS 侧照着接.
  2. turn/start 的事件源: timer / trajectory 帧 / attention hook / 外部 signal?
  3. 固定 nucleus 的 impulse 语义: 走正常仲裁, 还是专用自醒通道
     (strength=0 yield / 特定 priority / silent mode)? 会不会与真实输入抢 attention?
  4. logos 判定标准: DSH 返回流里怎么区分 logos vs 非 logos (CTML vs 文本 / role 标记)?
  5. ego topic 是强类型 TopicModel 还是通用 dict (当前倾向通用 dict)?

── deferred (本期不做, 但别堵缝) ─────────────────────────────────────
  epoch 周期 (ground_instruction 装线 / commit 落 Memento / compact 压上下文)
  是比 turn 更长的周期. run 面不堵死它的接入点.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from ghoshell_moss.core.blueprint.memento import Moment
from ghoshell_moss.core.blueprint.mindflow import Signal, ThinkingEffort
from ghoshell_moss.core.blueprint.session import OutputItem, Session
from ghoshell_moss.deepseek_harness.launcher import DshLauncherConfig
from ghoshell_moss.deepseek_harness.types.session_events import AssistantChunk
from ghoshell_moss.deepseek_harness.types.sessions import PromptContentPart

from .nucleus import new_dolores_ego_signal

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
    from ghoshell_moss.deepseek_harness.session import DshSession
    from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

    from ._runtime import Dolores
    from .nucleus import DoloresEgoNucleus

__all__ = ["DoloresConfig", "DoloresEgo", "DoloresEgoConfig"]

EGO_TOPIC_NAME = "dolores/ego"
"""dolores ego topic 默认名 — 通用 dict 包装所有 session event 的出口. 待讨论: 最终命名."""

THINKING_TOPIC_NAME = "dolores/thinking"
"""thinking start/end 的 topic 名. 待讨论: 独立 topic, 还是并入 ego topic 的 dict (event type 区分)."""

# plugin 路由 (与 moss-dolores-ghost-plugin.ts 的 DOLORES_* 常量对齐, 跨语言契约).
_DOLORES_EGO_CREATE = "/moss-api/ghost/dolores/ego/create"
_DOLORES_ARTICULATE_ENTER = "/moss-api/ghost/dolores/articulate/enter"
_DOLORES_ARTICULATE_EXIT = "/moss-api/ghost/dolores/articulate/exit"


class DoloresEgoConfig(BaseModel):
    """ego session 的配置 (从 .dolores.yml 的 ego: 段加载).

    字段默认值即 fallback — YAML 缺 key 时用默认, 不手动 .get().
    """

    agent_preset: str = Field(
        default="standard",
        description="dsh agent preset 名, 决定 ego session 的 persona + tool 组合.",
    )
    session_title: str = Field(
        default="{name} at {date}",
        description="session title 模板 ({name}/{date} 替换), 给人类看的会话名.",
    )
    permission: str = Field(
        default="workspace-write",
        description="sandbox 模式: read-only | workspace-write | danger-full-access.",
    )


class DoloresConfig(BaseModel):
    """ghost home 的 .dolores.yml 顶层配置. 字段默认值即 fallback."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(
        default="",
        description="stubs 同步版本标记 (对应 DoloresMeta.VERSION).",
    )
    dirs: list[str] = Field(
        default_factory=list,
        description="ghost home 里要物化的子目录.",
    )
    dsh: DshLauncherConfig = Field(
        default_factory=DshLauncherConfig,
        description="dsh launcher 配置 (binary/profile/port/...).",
    )
    ego: DoloresEgoConfig = Field(
        default_factory=DoloresEgoConfig,
        description="ego session 配置.",
    )


class DoloresEgo:
    """Dolores 的自我/连续性层. 详见模块 docstring."""

    def __init__(self, ghost: "Dolores", config: DoloresEgoConfig | None = None) -> None:
        """随 ghost 一起实例化, 构造无副作用 (不碰 httpx / session / matrix.processes).

        当前最小 slice: 只持 back-ref / config 与 ego session (DshSession).
        nucleus / watcher / run transaction / 异常感知待后续步骤.

        :param ghost: back-ref, ego 经它取 dsh_launcher / home / instruction.
        :param config: ego session 配置; None 用全默认.
        """
        self._ghost = ghost
        self._config = config or DoloresEgoConfig()
        self._session: "DshSession | None" = None
        self._ego_session_id: str | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        # self-wake gate: articulate 进行中 (Python 侧权威 flag), turn/start 监听据此决定是否自醒.
        self._articulating = False
        # 自醒 signal 出口 — host/mindflow 接总线后注入 (broadcast), 本侧不直接碰 nucleus.
        self._signal_broadcast: "Callable[[Signal], None] | None" = None

    # ── 长命线: 生命周期 ────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """进入 ghost 生命周期 (由 Dolores.__aenter__ 经 _exit_stack 进入).

        经 RPC 创建 ego session, 持有 DshSession facade.
        待后续: 固定 nucleus 注册 / 背景 watcher / session id 后台同步轮询.
        """
        await self._exit_stack.__aenter__()
        launcher = self._ghost.dsh_launcher
        result = await launcher.call(
            _DOLORES_EGO_CREATE,
            {
                "project_home": str(self._ghost._home),
                "project_name": self._ghost._matrix.env.project_name,
                "title": self._config.session_title.format(
                    name=self._ghost.meta.name(),
                    date=datetime.now().strftime("%Y-%m-%d"),
                ),
                "instruction": self._ghost.system_prompt(),
                "agent_preset": self._config.agent_preset,
                "permission": self._config.permission,
            },
        )
        self._ego_session_id = result["sessionId"]
        self._session = launcher.create_session(self._ego_session_id)
        await self._exit_stack.enter_async_context(self._session)
        # 长命线: 订阅 turn/start, 静默自醒 (self-wake 心跳).
        self._session.on_session_event("turn/start", self._on_turn_start)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出: 关闭 ego session (DshSession)."""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def session(self) -> "DshSession":
        """ego 持有的 dsh session facade — 未启动时抛清晰错误."""
        if self._session is None:
            raise RuntimeError("ego session not started. Call __aenter__ first.")
        return self._session

    # ── 短命线: run (transaction) ───────────────────────────────────

    async def run(self) -> AsyncIterator[str]:
        """最小 run transaction — per-idle 生命周期 (articulator 一 cycle 一 run).

        顺序: 先装线 event 消费 → create task 跑 ``_enter`` (开锁 + 驱动 + 阻塞到 idle)
        → 外部 yield 文本块 → exit 时 cancel task + 关锁 (finally 保证).

        无参数 (moment/effort 后续再上): 输入暂以固定 hello prompt 驱动.

        :yield: assistant 文本流块 (logos 片段), 逐段交给 Dolores.articulate.
        """
        queue: "asyncio.Queue[str | None]" = asyncio.Queue()

        async def _on_chunk(event: AssistantChunk) -> None:
            if event.chunk.type == "text-delta" and event.chunk.text:
                await queue.put(event.chunk.text)

        session = self.session
        dispose_chunk = session.on_session_event_model(AssistantChunk, _on_chunk)
        enter_task = asyncio.create_task(self._enter(queue))
        try:
            while True:
                text = await queue.get()
                if text is None:  # idle 哨兵: 整个 articulate 周期结束
                    break
                yield text
        finally:
            enter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await enter_task
            dispose_chunk()
            self._articulating = False
            await self._rpc_articulate_exit()

    async def _enter(self, queue: "asyncio.Queue[str | None]") -> None:
        """enter — 开锁 + 驱动 turn + 阻塞到 idle (per-idle done 判定, 非 turn/end).

        先开 plugin 锁, 置 Python 权威 flag, 再驱动 hello turn. 阻塞到 idle 是电平触发
        镜像, 故先 ``when_running`` 确认 turn 已启动, 再 ``when_idle`` 等整周期回 idle.

        todo: idle 权威应对回 plugin — 现在用 Python 镜像 ``when_running→when_idle``
        (电平触发, 超快 turn 时 ``when_running`` 可能错过 running 窗口而卡死). 最终形态
        是 enter RPC 在 plugin 侧 ``await agent.whenIdle()`` (进程内权威), Python 只 await
        该 RPC 返回.
        """
        await self._rpc_articulate_enter()
        self._articulating = True
        await self.session.prompt(content=[PromptContentPart(type="text", text="hello")])
        await self.session.when_running()
        await self.session.when_idle()
        await queue.put(None)

    # ── 上下文组装 ──────────────────────────────────────────────────

    async def _assemble_context(self, moment: Moment, effort: ThinkingEffort):
        """percepts + hint + trajectory 帧 + instruction (system_prompt + ground_instruction).

        perspectives 不读 (被 trajectory 取代).
        待讨论 seam #1: 返回类型 = DSH 请求 payload 形状 (RPC 入参).
        待讨论: ground_instruction (epoch 槽位) 本轮就接, 还是留给 epoch 周期.
        """
        ...

    # ── 背景 watcher (长命线) ───────────────────────────────────────

    @property
    def articulating(self) -> bool:
        """self-wake gate — Python 侧权威 flag, 由 articulate 进入/退出置 True/False."""
        return self._articulating

    @articulating.setter
    def articulating(self, value: bool) -> None:
        self._articulating = value

    def bind_signal_broadcast(self, broadcast: "Callable[[Signal], None]") -> None:
        """注入自醒 signal 出口 (host/mindflow 总线 broadcast).

        自醒 signal 由 ego 生产, 但投递归 mindflow 总线 (按 signal name 路由到
        DoloresEgoNucleus). 本方法给 host 一个接缝, ego 不直接持有 nucleus 实例.
        """
        self._signal_broadcast = broadcast

    async def _on_turn_start(self, event: "SessionEvent") -> None:
        """turn/start 监听回调 — 静默自醒心跳.

        gate: articulate 进行中 (Python 侧权威 flag) → 本 ghost 自己在驱动, 不醒.
        否则 dsh 侧自行起了一个 turn, ghost 该醒 — 发一封自醒 signal 给 nucleus.
        """
        if self._articulating:
            return
        self._emit_self_wake()

    def _emit_self_wake(self) -> None:
        """产一封自醒 signal 并投递 (无 broadcast 时静默, 供测试/未接线期)."""
        signal = new_dolores_ego_signal()
        if self._signal_broadcast is not None:
            self._signal_broadcast(signal)

    # ── 固定 nucleus ────────────────────────────────────────────────

    def nucleus(self) -> "DoloresEgoNucleus":
        """自醒 nucleus 句柄.

        impulse 走默认 mode 正常仲裁 (已定): 自醒 signal → info 级空 body impulse →
        正常 challenge, 不占专用自醒通道. 后续再织入轨迹物料.
        """
        ...

    # ── session event 响应 ──────────────────────────────────────────

    def _on_session_output(self, item: OutputItem) -> None:
        """run 期间挂到 session.on_output 的回调: 包通用 dict 发 ego topic (异步)."""
        ...

    def _on_session_signal(self, signal: Signal) -> None:
        """run 期间挂到 session.on_signal 的回调: 包通用 dict 发 ego topic (异步)."""
        ...

    async def _on_tool_call(self, call_id: str, tool_name: str, arguments: dict) -> None:
        """ego tool 执行桥: 模型调 ego tool → 本侧执行 → RPC tool-result 解锁 plugin.

        四个 ego tool (对齐 plugin 的 tool 表面):
          full_facade()             — 拉全量 channel 操作面 (→ trajectory/shell)
          get_channel_facade(path)  — 拉单个 channel facade
          moss_observe(budget?)     — 读观测轨迹 (只读, → trajectory)
          ctml_interrupt()          — 紧急停止 (→ shell.clear)
        关联键 = tool/call 的 callId (plugin 侧 pendingCalls/arrivedResults 双 map 防竞态).
        """
        ...

    def _publish_ego_topic(self, event: dict) -> None:
        """通用 dict → 包成 ego topic 发送 (必须异步).

        待讨论 seam #5: 通用 dict vs 强类型 TopicModel.
        """
        ...

    # ── RPC (窄桥, 与 plugin TS 对齐) ───────────────────────────────

    async def _rpc_create_ego_session(self, *, params: dict, steer: dict | None) -> str:
        """POST /moss-api/ghost/dolores/ego/create — 建 ego session (tool 注册 + preStep + 设 id).

        返回 ego session id, 本侧同步持有 (对齐 plugin 的 doloresEgoSessionId).
        待讨论 seam #1: params + steer 的入参形状.
        """
        ...

    async def _rpc_tool_result(self, call_id: str, result: dict) -> None:
        """POST /moss-api/ghost/dolores/tool-result — {callId, result} 解锁 plugin 侧 pending tool."""
        ...

    async def _rpc_articulate_enter(self) -> None:
        """POST /moss-api/ghost/dolores/articulate/enter — 打开 plugin 侧 perStep 锁 (articulating=true)."""
        await self._ghost.dsh_launcher.call(_DOLORES_ARTICULATE_ENTER, {})

    async def _rpc_articulate_exit(self) -> None:
        """POST /moss-api/ghost/dolores/articulate/exit — 关闭 plugin 侧 perStep 锁 (articulating=false)."""
        await self._ghost.dsh_launcher.call(_DOLORES_ARTICULATE_EXIT, {})

    # ── 异常感知 ────────────────────────────────────────────────────

    @property
    def last_error(self) -> Exception | None:
        """持有的上一轮异常, 下一轮 run 组装上下文时注入."""
        ...
