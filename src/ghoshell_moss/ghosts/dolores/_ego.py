"""DoloresEgo — Dolores 的自我/连续性层: thinking 交易 + 外部活动自醒.

ghost 侧 Python 面, 与 dsh_plugin/moss-dolores-ghost-plugin.ts 并行 (TS 持 dsh 内核侧:
ego/create + thinking/enter|exit + tool-result + perStep 锁). 本文件持 ego 会话状态、
moment→wire 序列化 (三槽位) 与外部活动自醒 watch. 完整协议演进 / 迭代计划见 FEATURE
(ghost-prototype-dolores) + plugin.ts 头注释.

── 两条线 ──
长命线 (随 ghost 生命周期): 背景 watcher 盯 turn/start + user/message → 自醒 signal
(DoloresEgoNucleus). thinking 交易进行中 (is_thinking) 时不自醒 — 交易内由 MOSS 自驱.
短命线 (每次 thinking 一次): run_thinking(thinking) → DoloresRun (async with 交易边界
+ 事件流). 交易生命周期 (enter/exit/yield/observe/perStep) 归 run, 详见 _run.py.

── moment 三槽位 (enter 注入语义, python 侧组装) ──
xml-like (<moment>/<inputs>/<epoch> 容器) 只在 python 侧烤好, plugin 是 dumb transport
(收 content blocks / admit image / 注入, 不 parse). context = echoes/dynamic/executing
折叠 <moment> (inject, 不驱动 turn); inputs = percepts 平铺 + optional <hint> (steer,
驱动 turn); epoch = epoch 变更时 (observer.epoch.id 变) → <epoch index=N> recap+baseline
(inject 背景, 首帧也触发).
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
from ghoshell_moss.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent
from ghoshell_moss.message import Content, Message

from .nucleus import new_dolores_ego_signal

if TYPE_CHECKING:
    from ._runtime import Dolores
    from .nucleus import DoloresEgoNucleus
    from ._run import DoloresRun
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

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
    inception_template: str = Field(
        default="",
        description=(
            "instruction 模板文件路径 (相对 ghost home). 只替换 dolores 人格/礼仪层, "
            "协议段 (fence 语义 + tool 锚点) 不可替换. 空 = 内置默认模板. "
            "槽位: {ghost_home} / {project_home} / {mode_home}."
        ),
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
    - facade: shell 上下文表面 (ghost 持有, 供 append_ctml 刷新 meta 用).
    """

    project_home: Path
    project_name: str
    name: str
    instruction: str
    facade: "MShellContextFacade"


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
        self._facade = ctx.facade
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
        self._thinking_event = asyncio.Event()
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
        return DoloresRun(ego=self, thinking=thinking, thinking_event=self._thinking_event, facade=self._facade)

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
    def is_thinking(self) -> bool:
        """self-wake gate — 是否处于 thinking 交易中 (thinking_event 由 run aenter/aexit 置位)."""
        return self._thinking_event.is_set()

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
        if self.is_thinking:
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

        fetch_next_moment / wait_next_moment / append_ctml 分派已落在 DoloresRun._handle_tool_use_event (见 _run.py);
        其余 tool 面 (interleaved_logos / switch_model) deferred — 本方法保留为通用 tool
        分发点, 未来多 tool 时在此按 tool_name 分派.
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

    async def _rpc_tool_result(
            self,
            call_id: str,
            result: dict | list | str | None,
            moment: list[dict] | None = None,
    ) -> None:
        """POST /moss-api/ghost/dolores/tool-result — {callId, result, moment} 解锁 pending tool.

        result = tool 给模型的返回值 (fetch_next_moment 为 "{epoch}-{moment}" 短 id). moment = 要注入
        上下文的 moment content 段 (MomentContentPart, text + image); plugin 侧先 inject
        moment 再 resolve result (见 plugin /tool-result). callId 透传, plugin 按 callId 路由.
        """
        await self._launcher.call(_DOLORES_TOOL_RESULT, {
            "callId": call_id,
            "result": result,
            "moment": moment,
        })

    def moment_context_parts(self, moment: Moment, moment_id: str) -> list[dict]:
        """moment → 注入上下文的 content blocks (context 槽位, 排除 percept/hint).

        fetch_next_moment tool 经 tool-result RPC 把 moment 注入下一个 step 的上下文 (背景, 不驱动
        turn). 保留 content blocks (text + image), 不折叠成 string. 无 context 内容 → 空.
        """
        context_msg = self._context_message(moment, moment_id)
        if context_msg is None:
            return []
        return [self._content_payload(content) for content in context_msg.as_contents(with_meta=True)]

    async def enter_thinking(self, thinking: "Thinking") -> None:
        """POST /moss-api/ghost/dolores/thinking/enter — moment 一条 user message + epoch + effort + model + token.

        防旁路 (点 4): body 携带 thinkingToken, plugin 校验 — 拒绝非 ego 发起的调用.
        """
        moment = thinking.moment
        moment_ref = f"{thinking.observer.epoch.index}-{moment.index}"
        payload = {
            "moment": self._moment_payload(moment, moment_ref),
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

    def _context_message(self, moment: Moment, moment_id: str) -> Message | None:
        """context 槽位 — as_moment_message 排除 percept/hint (echoes/dynamic/executing).

        折叠成一条 ``<moment moment_id=...>`` 消息, inject 进上下文 (背景, 不驱动 turn).
        moment_id = "{epoch.index}-{moment.index}" 组合 id (非 uuid). 无 context 内容
        (echoes/dynamic/executing 均空) → None.
        """
        return moment.as_moment_message(
            always_return=False,
            with_moment_id=False,
            with_percepts=False,
            with_hint=False,
            attributes={'moment_id': moment_id},
        )

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

    def _moment_payload(self, moment: Moment, moment_id: str) -> dict:
        """moment → 两条 message 的 wire content (点 6): context + inputs.

        context = ``_context_message`` (echoes/dynamic/executing 折叠 ``<moment>``, inject 用);
        inputs = ``_inputs_message`` (percepts + hint 折叠 ``<inputs>``, steer 用). 映射在
        python 侧做, plugin 只收两条现成 content 分别投. text 直传, image 转 base64
        EncodedImageAttachment (保留多模态). ``moment_id`` = "{epoch.index}-{moment.index}".
        """
        context_msg = self._context_message(moment, moment_id)
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
            "moment_id": moment_id,
        }

    def _content_payload(self, content: Content | dict) -> dict[str, Any]:
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
