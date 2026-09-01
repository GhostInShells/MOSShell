"""DoloresEgo — Dolores 的自我/连续性层 (thinking 交易 surface, 逐步落地).

本文件负责 ghost 侧 (Python), 与 dsh_plugin/moss-dolores-ghost-plugin.ts 并行 —
TS 负责 dsh 内核侧. 早期只画表面 (``...`` 占位); 2026-08-28 起 run_thinking 落地,
其余 (epoch / tool 桥 / on_event 内部逻辑) 仍待后续.

── 定位 ──────────────────────────────────────────────────────────────
DoloresEgo 是 Dolores ghost 的 "我". 随 ghost 创建时实例化, 进入同一
生命周期. 它把散在 Dolores 本体上的会话响应逻辑收敛成一个有边界的自我,
负责 thinking 交易 (短命线) + 会话事件响应 + 与 DSH 推理中枢的窄桥.

── 两条线 ────────────────────────────────────────────────────────────
长命线 (随 ghost 生命周期):
  背景 watcher 只盯 turn/start 一种事件, 每次往一个固定 nucleus 打一个
  一次性 impulse. 这是自驱心跳 — mindflow 静默时也能被唤醒, 走正常
  challenge → attention → articulate 路径, 不用等外部输入.
  通知与背压分工: turn/start 广播 = 通知 (外部唤醒 → 自醒 signal);
  pre-step 阻塞 = 背压 (hold 住模型等 thinking/enter 注入帧).

短命线 (每次 thinking 一次):
  ``run_thinking(thinking)`` 返回 DoloresRun (async with 边界 + events() 消费):
      async with ego.run_thinking(thinking) as run:
          async for event in run.events(): ...
  Dolores.think 消费 run.events() 分派 logos, 管理 articulator; 生命周期
  (listener/enter/exit) 归 DoloresRun. 详见 _run.py 与 plugin.ts.

── thinking 交易 (收敛方案 2026-08-28, B 范式) ────────────────────────
B 范式: MOSS mindflow 是 dsh 每个 turn 的「上下文服务方」, pre-step enter 是
服务接口. 每个 turn 自包含: enter-inject → model 跑 → turn/end 收线. 帧按
状态分叉: mindflow 活跃 → live moment; idle → 静态状态快照.

  thinking/enter   — moment 一条 user message + epoch + effort + model config + thinkingToken.
                     handler 阻塞执行完: 注入帧 → openThinking (释放 pre-step 锁)
                     → 若 idle steer. moment 是自解释容器 (见下); epoch 是并列槽位
                     (recap 前情提要, 本轮恒空).
  thinking/exit    — 反转 thinking 状态; agent 非 idle 时显式 cancel (不空跑失速).
  perStep 锁       — foreign session → reject + mux 提示冻结; ego 非 thinking →
                     阻塞等反转 (背压). 锁由 thinking signal 提供 (TS promise,
                     非 python 伪 async).
  防旁路           — thinkingToken (ego/create 生成返回, enter/exit 携带校验).

  对比早期 articulate/enter (2026-08-27 收敛): 当时设想 enter 组合翻译 effort/
  percepts/enter-with-messages, "enter 包揽 exit" 候选 (SSE 等 turn/end). 2026-08-28
  落地为: pre-step enter 是统一注入点 (非锁), 完成判定回 turn/end (消费方 break 收线).
  早期方案的组合翻译细节 (effort→reasoningEffort / percepts→steer / enter-with-messages
  原子通道 / surface replace) 仍有效, 见 plugin.ts 头注释与 git log. 任务完成后再删.

── session event 响应 (run 时监听) ───────────────────────────────────
  1. logos 判定: _fetch_logos (assistant/chunk text-delta) → articulator + yield.
  2. turn/end: 消费方 break 收线 (正常路径); 毒丸只承载 enter 异常.
  3. ego tool 调用 (tool 面暂缓, 点 7): tool/call → 执行 → RPC tool-result 解锁.
  4. 异常感知: DoloresRun aexit 时 thinking.abort(reason).
  (DoloresRun._on_event 后续会有逻辑 — token 记账 / tool 桥 / seq 跟踪, 本阶段纯透传.)

── moment 一条 user message + epoch 槽位 (点 6) ──────────────────────
  moment   — as_moment_message 折叠整帧为 <moment moment_id=...> 单条消息
             (内含 echoes/percepts/dynamic/hint 子段). plugin 只按序 steer/append
             这一条, 不拆块、不镜像三块结构; moment_id 独立传 — commit 锚.
  epoch    — (与 moment 并列的新槽位) epoch 级稳定上下文: recap 前情提要 + ground_instruction.
             槽位已开, 本轮恒空 (epoch 周期 deferred, 见下); 装线后从 observer.epoch.recap 投影.
  command_logos 不在 run 面 (反射弧已在 articulate 前 send_nowait 消费).
  thinking_effort 在 articulator 上, 经 enter RPC 的 effort 字段上.
  contexts 观测由 MindflowInShell 装线的 shell trajectory 进 moment.previous,
  ego 只消费 moment, 不读 trajectory.

── yield 机制 (wait_next_moment tool, A 范式) ──────────────────────
模型在 thinking 中主动调 wait_next_moment, 阻塞等下一帧 MOSS moment. tool use 是
turn 边界信号 (非 turn 内续帧): 消费方认出 tool/call = wait_next_moment → break 收线
(同 turn/end), 触发 thinking exit. exit 时 plugin 侧 pendingYield 非空 → 不 cancel,
tool 继续 pending. 下一轮 thinking/enter 用 moment 文本构造 tool result 解锁 (moment 走
tool 返回值, 不经 surface). cancel 守卫: tool 被 cancel 时 pendingYield 清空,
moment 改走下一轮 enter 正常 steer/append 路径 (轨迹不丢, 可 debug). momentId 暂不消费.

── moment 容器 + commit 锚 (2026-08-31) ──────────────────────────────
dsh/DeepSeek 走 OpenAI-completions, 缓存是自动前缀缓存, 无 Anthropic cache_control
显式断点、无多 cache index — 所以「无痛改历史」无解 (中途摘 dynamic 破坏前缀触发
重算). 故 moment 容器化: as_moment_message 包成自解释的 <moment moment_id=...> 单条
消息, 统一 tool use / user message 两路, dynamic 留在容器里不摘 (full_moment_messages
给裸子段). commit 走主路 + 旁路 fork session (不走 dsh compact, 慢), 一个 session 多
commit、历史不折叠; 「提交 moment A 之前的历史」的下边界只能落 moment id (cache 层
给不了锚), commit 触发即按 id 注入上下文.

── 待讨论 (seams) ────────────────────────────────────────────────────
  1. external wake 的 fail-safe: pre-step 阻塞等 thinking/enter, MOSS 永不 enter
     时 turn 挂死 — 需超时后 reject + mux 提示 (plugin 侧待定).
  2. turn/start 的事件源: timer / trajectory 帧 / attention hook / 外部 signal?
  3. 固定 nucleus 的 impulse 语义: 走正常仲裁, 还是专用自醒通道?
  4. logos 判定标准: DSH 返回流里怎么区分 logos vs 非 logos?
  5. ego topic 是强类型 TopicModel 还是通用 dict (当前倾向通用 dict)?

── deferred (本期不做, 但别堵缝) ─────────────────────────────────────
  epoch 周期 (ground_instruction 装线 / commit 落 Memento / compact 压上下文).
  tool 面 (4 个 ego tool + tool-result 桥). on_event 内部逻辑 (token/工具/seq).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.moment import Moment
from ghoshell_moss.core.blueprint.mindflow import Signal, Thinking, ThinkingEffort
from ghoshell_moss.core.blueprint.session import OutputItem, Session
from ghoshell_moss.deepseek_harness.launcher import DshLauncherConfig
from ghoshell_moss.message import Message

from .nucleus import new_dolores_ego_signal

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher
    from ghoshell_moss.deepseek_harness.session import DshSession
    from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

    from ._runtime import Dolores
    from .nucleus import DoloresEgoNucleus

__all__ = ["DoloresConfig", "DoloresEgo", "DoloresEgoConfig", "DoloresEgoContext"]

EGO_TOPIC_NAME = "dolores/ego"
"""dolores ego topic 默认名 — 通用 dict 包装所有 session event 的出口. 待讨论: 最终命名."""

THINKING_TOPIC_NAME = "dolores/thinking"
"""thinking start/end 的 topic 名. 待讨论: 独立 topic, 还是并入 ego topic 的 dict (event type 区分)."""

# plugin 路由 (与 moss-dolores-ghost-plugin.ts 的 DOLORES_* 常量对齐, 跨语言契约).
_DOLORES_EGO_CREATE = "/moss-api/ghost/dolores/ego/create"
_DOLORES_THINKING_ENTER = "/moss-api/ghost/dolores/thinking/enter"
_DOLORES_THINKING_EXIT = "/moss-api/ghost/dolores/thinking/exit"

# thinking/exit 的阻塞确认超时 (fail-safe): plugin 挂死时降级, 不挂死 thinking 退出.
_EXIT_RPC_TIMEOUT = 5.0


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


@dataclasses.dataclass(frozen=True, slots=True)
class DoloresEgoContext:
    """ego 进入生命周期前的一切静态上下文 — 由 ghost 装配后注入, 阻断 back-ref 穿透.

    随 ego 创建一次性取值, 构造后不再需要访问 ghost 内部. 所有引用都经 typehint 对象 /
    变量 / 闭包注入, 不持有 ghost 反引用.

    - project_home: ego session 的工作区目录 (原 ghost._home).
    - project_name: 工作区标题名 (原 ghost._matrix.env.project_name).
    - name: ghost 名, 用于标题/身份 (原 ghost.meta.name()).
    - instruction: 组装好的 system prompt (原 ghost.system_prompt()).
    """

    project_home: Path
    project_name: str
    name: str
    instruction: str


class DoloresEgo:
    """Dolores 的自我/连续性层. 详见模块 docstring."""

    def __init__(
        self,
        *,
        launcher: "DshLauncher",
        ctx: DoloresEgoContext,
        config: DoloresEgoConfig | None = None,
        logger: LoggerItf | None = None,
        memories: Callable[[], list[Message]] | None = None,
    ) -> None:
        """随 ghost 进入生命周期前构造, 构造无副作用 (不碰 httpx / session / matrix.processes).

        依赖纪律: 一切依赖经 typehint 对象 (launcher/config/logger) / 变量 (ctx) / 闭包
        (memories / bind_signal_broadcast) 注入, 不持有 ghost 反引用, 不访问任何私有成员.

        :param launcher: dsh 推理中枢启动器, ego create + thinking enter/exit RPC 走它.
        :param ctx: 一次性会话上下文 (home/name/instruction/project_name).
        :param config: ego session 配置; None 用全默认.
        :param logger: 记录器; None fallback 到 MOSS logger.
        :param memories: ghost 动态记忆闭包 (存在主义层), create_session 时调用取最新;
            clone 复用同一闭包共享认知. None 表示无记忆.
        """
        self._launcher = launcher
        self._ctx = ctx
        self._config = config or DoloresEgoConfig()
        self._memories = memories
        self._session: "DshSession | None" = None
        self._ego_session_id: str | None = None
        # 防旁路 token (点 4): ego/create 返回, thinking/enter|exit 携带, plugin 校验拒绝非 ego 调用.
        self._thinking_token: str | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        # logger 优先调用方注入 (MOSS runtime logger), 否则 fallback.
        self._logger = logger or get_moss_logger()
        # self-wake gate: thinking 交易进行中 (Python 侧镜像 flag), turn/start 监听据此决定是否自醒.
        self._articulating = False
        # 自醒 signal 出口 — host/mindflow 接总线后注入 (broadcast), 本侧不直接碰 nucleus.
        self._signal_broadcast: "Callable[[Signal], None] | None" = None

    # ── 长命线: 生命周期 ────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """进入 ghost 生命周期 (由 Dolores.__aenter__ 经 _exit_stack 进入)."""
        await self._exit_stack.__aenter__()
        await self.create_session()
        return self

    async def create_session(self) -> str:
        """创建 ego session (可复用 — clone 共享同一 memories 闭包).

        每个 session 创建时注入 instruction + memory (ghost 动态记忆, 1:1 转 user message),
        建立"instruction 之下、对话之上"的初始表面. 返回 ego session id.
        """
        result = await self._launcher.call(
            _DOLORES_EGO_CREATE,
            {
                "project_home": str(self._ctx.project_home),
                "project_name": self._ctx.project_name,
                "title": self._config.session_title.format(
                    name=self._ctx.name,
                    date=datetime.now().strftime("%Y-%m-%d"),
                ),
                "instruction": self._ctx.instruction,
                "messages": self._assemble_initial_messages(),
                "agent_preset": self._config.agent_preset,
                "permission": self._config.permission,
            },
        )
        self._ego_session_id = result["sessionId"]
        self._thinking_token = result.get("thinkingToken")
        self._session = self._launcher.create_session(self._ego_session_id)
        await self._exit_stack.enter_async_context(self._session)
        # 长命线: 订阅 turn/start + user/message, 静默自醒 (self-wake 心跳).
        # user/message 覆盖界面直投场景 — yield 后 dsh loop 卡在等 tool result, 界面
        # 输入只产生 user/message 不产生 turn/start, 仍需自醒解锁 pending tool.
        self._session.on_session_event("turn/start", self._on_session_activity)
        self._session.on_session_event("user/message", self._on_session_activity)
        return self._ego_session_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出: 关闭 ego session (DshSession)."""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def session(self) -> "DshSession":
        """ego 持有的 dsh session facade — 未启动时抛清晰错误."""
        if self._session is None:
            raise RuntimeError("ego session not started. Call __aenter__ first.")
        return self._session

    # ── 短命线: run_thinking (transaction) ─────────────────────────

    def run_thinking(self, thinking: "Thinking") -> "DoloresRun":
        """返回 thinking 交易的 run 对象 — Dolores.think 用 ``async with`` 消费.

        async with 边界 = 显式生命周期 (见 _run.DoloresRun): aenter 绑监听+建 enter
        task, aexit cancel+解绑+补发 exit+abort. 消费者经 ``run.events()`` 拉事件,
        分派 logos/turn/end; articulator 由消费方 (Dolores.think) 管理.

        :param thinking: mindflow Thinking — moment/effort/articulator/abort 全从它取.
        """
        from ._run import DoloresRun

        return DoloresRun(ego=self, thinking=thinking)

    # ── 上下文组装 ──────────────────────────────────────────────────

    async def _assemble_context(self, moment: Moment, effort: ThinkingEffort):
        """percepts + hint + instruction (system_prompt + ground_instruction).

        上下文观测由 MindflowInShell 装线的 shell trajectory 注入 moment.previous (echoes),
        本方法只拼 moment message + instruction.
        待讨论 seam #1: 返回类型 = DSH 请求 payload 形状 (RPC 入参).
        epoch 槽位 (ground_instruction + recap) 已在 enter 开槽恒空, 装线留给 epoch 周期.
        """
        ...

    def _assemble_initial_messages(self) -> list[dict]:
        """初见上下文 messages: ghost 动态记忆 (memories 闭包) → plugin payload.

        每项 ``{"text": ...}`` — plugin 侧作为 user/message (plugin source) 注入 surface.
        memory 1:1 转 dsh user message (不做折叠; moment 折叠走 message_mapper.fold_messages).
        """
        if self._memories is None:
            return []
        return [
            {"text": msg.to_content_string()}
            for msg in self._memories()
            if not msg.is_empty()
        ]

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

    async def _on_session_activity(self, event: "SessionEvent") -> None:
        """外部会话活动监听回调 (turn/start + user/message) — 静默自醒心跳.

        gate: articulate 进行中 (Python 侧权威 flag) → 本 ghost 自己在驱动, 不醒.
        否则 dsh 侧有外部活动 (起了 turn / 界面直投消息), ghost 该醒 — 发一封
        自醒 signal 给 nucleus. 可丢弃: nucleus 造 BACKGROUND impulse, mindflow
        忙时 challenge 失败即丢, 只有空闲时才唤醒.
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

    async def enter_thinking(self, thinking: "Thinking") -> None:
        """POST /moss-api/ghost/dolores/thinking/enter — moment 一条 user message + epoch + effort + model + token.

        防旁路 (点 4): body 携带 thinkingToken, plugin 校验 — 拒绝非 ego 发起的调用.
        """
        moment = thinking.moment
        payload = {
            "moment": self._moment_payload(moment),
            "epoch": self._epoch_payload(),
            "effort": thinking.effort(),
            "model": await self._model_config(),
            "thinkingToken": self._thinking_token,
        }
        await self._launcher.call(_DOLORES_THINKING_ENTER, payload)

    async def exit_thinking(self, *, yielded: bool = False) -> None:
        """POST /moss-api/ghost/dolores/thinking/exit — 反转 thinking 状态; plugin 侧相关动作.

        yielded: MOSS 侧已明确判定本次 break 是 yield tool (wait_next_moment) 收线 —
        plugin 侧据此**不再 cancel** (tool 保持阻塞等下一帧 moment 解锁), 不依赖 plugin
        侧 pendingYield 的时序竞态. 非 yield + agent 非 idle 才由 plugin cancel.
        阻塞到确认 (避免并发), 带超时 fail-safe: plugin 挂死时降级, 不挂死 thinking 退出.
        """
        try:
            await self._launcher.call(
                _DOLORES_THINKING_EXIT,
                {
                    "thinkingToken": self._thinking_token,
                    "yielded": yielded,
                },
                timeout=_EXIT_RPC_TIMEOUT,
            )
        except Exception:
            self._logger.exception("thinking/exit failed — degraded; state may be stale")

    def _moment_payload(self, moment: Moment) -> dict:
        """moment → 一条自解释 user message 的 content blocks (点 6).

        ``as_moment_message`` 折叠整帧为 ``<moment moment_id=...>`` 单条消息
        (内含 echoes/percepts/dynamic/hint 子段), 序列化成 dsh wire content 列表
        (text 直传, image 转 base64 EncodedImageAttachment — 保留多模态). plugin
        按序 steer/append 这一条, 不拆块、不镜像三块结构. ``moment_id`` 独立传 — commit 锚.
        """
        msg = moment.as_moment_message()
        if msg is None:
            return {"contents": [], "moment_id": moment.id}
        return {
            "contents": [
                self._content_payload(content)
                for content in msg.as_contents(with_meta=True)
            ],
            "moment_id": moment.id,
        }

    def _content_payload(self, content: dict) -> dict:
        """MOSS content → dsh wire content. image 的 base64 保留, 转 EncodedImageAttachment 形状."""
        if content.get("type") == "image":
            source = content.get("source") or {}
            return {
                "type": "image",
                "mediaType": source.get("media_type"),
                "data": source.get("data", ""),
            }
        return content

    def _epoch_payload(self) -> list[dict]:
        """epoch messages → plugin payload (与 moment message 并列的新槽位).

        epoch 周期 (compact 压上下文 → recap 前情提要 + ground_instruction) 尚未装线
        (deferred), 本槽位暂恒空 — 允许为空. 装线后从 observer.epoch.recap 1:1 投影
        ``{text}``, 作为 epoch 级稳定上下文注入 (非 hot 帧).
        """
        return []

    async def _model_config(self) -> dict:
        """当前模型配置 (provider/model/reasoningEffort) — 经 session.models 拉取."""
        selection = await self.session.model_selection()
        return {
            "provider": selection.provider,
            "model": selection.model,
            "reasoningEffort": selection.reasoningEffort,
        }

    # ── 异常感知 ────────────────────────────────────────────────────

    @property
    def last_error(self) -> Exception | None:
        """持有的上一轮异常, 下一轮 run 组装上下文时注入."""
        ...
