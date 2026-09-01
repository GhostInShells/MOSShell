"""DoloresEgo — Dolores 的自我/连续性层 (thinking 交易 surface, 逐步落地).

本文件负责 ghost 侧 (Python), 与 dsh_plugin/moss-dolores-ghost-plugin.ts 并行 —
TS 负责 dsh 内核侧. 早期只画表面 (``...`` 占位); 2026-08-28 起 run_thinking 落地,
2026-09-02 起 epoch 槽位 (<epoch> 容器) 与 observe tool (approach a) 落地;
on_event 内部逻辑 (token/工具/seq) 仍待后续.

── 迭代计划 (下一步, 逐项做; 七项全稳后才做 memento) ──────────────────
  不一批做完. 逐项: 对齐 → 实现 → 提交, 做完划一项.

  1. instruction 建立 — Dolores instruction (system prompt) 建四层: 认知
     (我是谁/存在/意义)、协议 (CTML 输出协议)、交互礼仪 (与人类协作的边界/
     方式)、篇幅控制. 重点: 交互 + 语音纪律 (voice discipline, 语音向 ghost).
  2. 测试 ghost 改名 — .moss/src/MOSS/ghosts/moss.py 测试 ghost 实例
     name="moss" → "deepseek"; DoloresMeta 默认 description 讲清楚.
  3. tools 对表面 — [已对齐, observe/epoch 已实现] 工具面收敛为: observe (主动观测,
     approach a 内联返回 moment content blocks) / yield_next_moment (被动让出) /
     interleaved_logos (= articulator(replan, wait_action_done) 的 tool 表面) /
     switch_model (反身, 落文档不实现). 原 full_facade 等 4 个 ego tool 砍掉 — facade
     走 epoch.baseline 注入 (push 非 pull), 不是 tool.
  4. 正文 logos 阻塞 — ego think 流程正文 logos 流式逐段 yield 还是阻塞到
     turn/end (done) 再整体产出 — 判定.
  5. workspace / title 正式命名 — ego/create 的 workspace title (= project_name)
     与 session title (session_title 模板) 正式命名, 去临时占位.
  6. yield tool schema — wait_next_moment parameters: {} 空 obj 换更好的空定义.
  7. system prompt 分隔符/去重 — system_prompt 加分隔符、正式化、去重
     (baseline + prototype + identity + CTML 提示 分段与重复消除).

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

  thinking/enter   — context + inputs 两个 message 槽位 + epoch 槽位 + effort + model config + thinkingToken.
                     handler 阻塞执行完: inject epoch(变更时) → inject context → steer inputs →
                     openThinking (释放 pre-step 锁). 全部由 python 侧组装 (见下); epoch 是
                     <epoch> 容器 (recap + baseline), 变更时注入为背景.
  thinking/exit    — 反转 thinking 状态; 非 yield 时 agent 非 idle 则显式 cancel (不空跑失速).
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
  3. observe tool 调用 (approach a): tool/call → thinking.observe() → RPC tool-result
     内联返回 moment content blocks. (其余 tool 面 deferred, 见下.)
  4. 异常感知: DoloresRun aexit 时 thinking.abort(reason).
  (DoloresRun._on_event 后续会有逻辑 — token 记账 / tool 桥 / seq 跟踪, 本阶段纯透传.)

── moment / epoch 映射: context + inputs + epoch 三个槽位 ─────────────
  协议边界: xml-like 只在 python 侧理解 (<moment>/<inputs>/<epoch> 全在此烤好),
  plugin 是 dumb transport — 只收 content blocks、admit image、注入, 不 parse 结构.
  context — _context_message: as_moment_message 排除 percept/hint (echoes/dynamic/
            executing 折叠成 <moment moment_id=...>), inject 进上下文 (背景, 不驱动 turn).
  inputs  — _inputs_message: percepts 平铺 + optional hint 折叠成 <inputs>, steer
            作为 turn 输入 (驱动). 映射在 python 侧 (ego) 组装, plugin 只收两条现成 content.
  epoch   — _epoch_payload: epoch 变更时 (observer.epoch.id 变) 折叠成 <epoch index=N>
            容器 (recap 前情提要 + baseline 起点信息, baseline key 渲染成 <key>value</key>),
            inject 为背景 (一个槽, 不像 moment 两个槽 — 注入语义单一). 首帧也触发.
  command_logos 归 context (executing 子段), 是「感知」不是「输入」; 反射弧已在 articulate 前 send_nowait 消费.
  thinking_effort 在 articulator 上, 经 enter RPC 的 effort 字段上.
  contexts 观测由 MindflowInShell 装线的 shell trajectory 进 moment.previous,
  ego 只消费 moment, 不读 trajectory.

── yield 机制 (wait_next_moment tool, A 范式) ──────────────────────
模型在 thinking 中主动调 wait_next_moment, 阻塞等下一帧 MOSS moment. tool use 是
turn 边界信号 (非 turn 内续帧): 消费方认出 tool/call = wait_next_moment → break 收线
(同 turn/end), 触发 thinking exit. exit 时 yielded=true → 不 cancel (留 tool pending).
下一轮 thinking/enter: pendingYield 非空 → inject(context) + steer(inputs) + resolve("ok")
(str, 非 moment contents — moment 已走 context/inputs 槽位). tool 被 session.cancel 打断时
走 dsh 默认 abort, 与其它 tool 一致 (pendingYield 清空, 轨迹不丢). momentId 暂不消费.

── observe tool (approach a, 主动观测) ─────────────────────────────
与 yield 互补: yield 被动让出 (节奏权在 MOSS), observe 主动观测 (节奏权在模型).
模型调 observe → plugin execute 挂 pendingCalls[callId] → MOSS 侧认出 tool/call →
thinking.observe() 生产 moment → /tool-result RPC 按 callId 解锁, 内联返回 moment
content blocks (context + inputs 拼接, 保留图片). 不 break turn — 模型在 tool result
到达后继续思考 (interleaved thinking). tool-result 桥是单端点 (DOLORES_TOOL_RESULT),
按 callId 路由 (多 tool 各自 pending).

── moment 容器 + commit 锚 (2026-08-31) ──────────────────────────────
dsh/DeepSeek 走 OpenAI-completions, 缓存是自动前缀缓存, 无 Anthropic cache_control
显式断点、无多 cache index — 所以「无痛改历史」无解 (中途摘 dynamic 破坏前缀触发
重算). 故 moment 容器化: context 用 as_moment_message 包成自解释的 <moment moment_id=...>
消息 (排除 percept/hint), dynamic 留在容器里不摘 (full_moment_messages 给裸子段).
commit 走主路 + 旁路 fork session (不走 dsh compact, 慢), 一个 session 多
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
  epoch 周期 (compact 压上下文触发 recap 生产 / commit 落 Memento / ground_instruction
  装线). 注: epoch 槽位已实现 (<epoch> 容器), 但触发周期 (compact) 尚未装线.
  其余 tool 面 (interleaved_logos / switch_model — 落文档不实现). on_event 内部逻辑 (token/工具/seq).
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
_DOLORES_TOOL_RESULT = "/moss-api/ghost/dolores/tool-result"

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
        # epoch 跟踪: 记录已注入的 epoch id, enter 时比较 observer.epoch.id 决定是否携带 <epoch> 容器.
        self._moment_epoch: str | None = None

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
        """ego tool 执行桥 (seam, 暂未装线).

        observe 已单独在 Dolores.think 落地 (approach a: tool/call → thinking.observe() →
        _rpc_tool_result 内联返回 moment). 其余 tool 面 (interleaved_logos / switch_model)
        deferred — 本方法保留为通用 tool 分发点, 未来多 tool 时在此按 tool_name 分派.
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

    async def _rpc_tool_result(self, call_id: str, result: list[dict]) -> None:
        """POST /moss-api/ghost/dolores/tool-result — {callId, result} 解锁 plugin 侧 pending tool.

        result = wire content 段 (MomentContentPart, text + image), plugin 侧经
        durableMomentContent 转 ContentBlock (admit image). callId 透传, plugin 按
        callId 路由 (多 tool 各自 pending).
        """
        await self._launcher.call(_DOLORES_TOOL_RESULT, {"callId": call_id, "result": result})

    def _moment_content_parts(self, moment: Moment) -> list[dict]:
        """moment → observe tool result 的 content blocks (context + inputs 拼接, 内联返回).

        approach a: observe 内联返回 moment. 保留 content blocks (text + image), 不折叠
        成 string — 与 context/inputs 槽位一致, 支持图片.
        """
        payload = self._moment_payload(moment)
        return payload["context"] + payload["inputs"]

    async def enter_thinking(self, thinking: "Thinking") -> None:
        """POST /moss-api/ghost/dolores/thinking/enter — moment 一条 user message + epoch + effort + model + token.

        防旁路 (点 4): body 携带 thinkingToken, plugin 校验 — 拒绝非 ego 发起的调用.
        """
        moment = thinking.moment
        payload = {
            "moment": self._moment_payload(moment),
            "epoch": self._epoch_payload(thinking),
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

    def _context_message(self, moment: Moment) -> Message | None:
        """context 槽位 — as_moment_message 排除 percept/hint (echoes/dynamic/executing).

        折叠成一条 ``<moment moment_id=...>`` 消息, inject 进上下文 (背景, 不驱动 turn).
        无 context 内容 (echoes/dynamic/executing 均空) → None.
        """
        return moment.as_moment_message(always_return=False, with_percepts=False, with_hint=False)

    def _inputs_message(self, moment: Moment) -> Message | None:
        """inputs 槽位 — percepts + hint 包成一条 ``<inputs>`` 消息 (steer 用, 允许为空).

        percept 消息按 source 顺序平铺进容器 (不另包 ``<percepts>``), hint (optional)
        以 ``<hint>`` 子段排在最后. 无 percepts 无 hint → None.
        """
        messages: list[Message] = list(moment.percepts_messages())
        if moment.hint:
            messages.append(Message.new(tag='hint').with_content(moment.hint))
        if not messages:
            return None
        return Message.new(tag='inputs').with_messages(*messages)

    def _moment_payload(self, moment: Moment) -> dict:
        """moment → 两条 message 的 wire content (点 6): context + inputs.

        context = ``_context_message`` (echoes/dynamic/executing 折叠 ``<moment>``, inject 用);
        inputs = ``_inputs_message`` (percepts + hint 折叠 ``<inputs>``, steer 用). 映射在
        python 侧做, plugin 只收两条现成 content 分别投. text 直传, image 转 base64
        EncodedImageAttachment (保留多模态). ``moment_id`` 独立传 — commit 锚.
        """
        context_msg = self._context_message(moment)
        inputs_msg = self._inputs_message(moment)
        return {
            "context": [
                self._content_payload(content)
                for content in context_msg.as_contents(with_meta=True)
            ] if context_msg is not None else [],
            "inputs": [
                self._content_payload(content)
                for content in inputs_msg.as_contents(with_meta=True)
            ] if inputs_msg is not None else [],
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

    def _epoch_payload(self, thinking: "Thinking") -> list[dict] | None:
        """epoch 槽位 — 仅在 epoch 变更时返回 <epoch> 容器 content blocks.

        渲染成单条 ``<epoch index=N>`` 容器: ``<recap>`` 前情提要 + ``<baseline>`` 起点
        信息 (每个 baseline key 渲染成 ``<key>value</key>``). xml-like 只在 python 侧
        理解, plugin 是 dumb transport — 只收 content blocks 注入, 不 parse 结构.
        首帧与 epoch 变更都返回; 未变返回 None.
        """
        epoch = thinking.observer.epoch
        if epoch.id == self._moment_epoch:
            return None
        self._moment_epoch = epoch.id
        children: list[Message] = []
        if epoch.recap:
            children.append(Message.new(tag="recap").with_messages(*epoch.recap))
        if epoch.baseline:
            baseline_msgs = [
                Message.new(tag=key).with_content(value)
                for key, value in epoch.baseline.items()
                if value
            ]
            children.append(Message.new(tag="baseline").with_messages(*baseline_msgs))
        if not children:
            return None
        container = Message.new(tag="epoch", attributes={"index": str(epoch.index)}).with_messages(*children)
        return [
            self._content_payload(content)
            for content in container.as_contents(with_meta=True)
        ]

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
