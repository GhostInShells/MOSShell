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

from typing import TYPE_CHECKING, AsyncIterator, Callable

from typing_extensions import Self

from ghoshell_moss.core.blueprint.memento import Moment
from ghoshell_moss.core.blueprint.mindflow import Impulse, Nucleus, Signal, ThinkingEffort
from ghoshell_moss.core.blueprint.session import OutputItem, Session

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory

    from ._runtime import Dolores

__all__ = ["DoloresEgo", "DoloresEgoNucleus"]

EGO_TOPIC_NAME = "dolores/ego"
"""dolores ego topic 默认名 — 通用 dict 包装所有 session event 的出口. 待讨论: 最终命名."""

THINKING_TOPIC_NAME = "dolores/thinking"
"""thinking start/end 的 topic 名. 待讨论: 独立 topic, 还是并入 ego topic 的 dict (event type 区分)."""


class DoloresEgoNucleus(Nucleus):
    """固定 self-nucleus — 接收背景 watcher 打来的一次性 self-wake impulse.

    职责极窄: 把 watcher 的 turn/start impulse 反射进 mindflow, 让静默的
    mindflow 也能走 challenge → attention → articulate 自醒一轮.

    待讨论 seam #3: 这个 impulse 的仲裁语义 — 走正常 attention 仲裁,
    还是专用自醒通道 (strength=0 yield / 特定 priority / silent mode)?
    关键在它不能和真实输入抢 attention.
    """

    def name(self) -> str:
        """自解释名 — 待讨论: 最终命名."""
        ...

    def description(self) -> str:
        """一句话自解释: ego 自醒通道."""
        ...

    def status(self) -> str:
        """红点式状态提示, 空则忽略."""
        ...

    def signals(self) -> list[str]:
        """声明监听的 signal 类型. ego 自醒走 impulse 直投, 是否还需要 signal 面 — 待讨论."""
        ...

    def clear(self) -> None:
        """排空讯号 (极限故障还原)."""
        ...

    def add_signal(self, signal: Signal) -> None:
        """接受信号 → 生成 impulse. 无背压, 不阻塞."""
        ...

    def with_bus(
        self,
        signal_broadcast: Callable[[Signal], None],
        impulse_notify: Callable[[Impulse], None],
    ) -> None:
        """注册总线: 广播 signal / 投递 impulse."""
        ...

    def suppress(self, suppress_by: Impulse) -> None:
        """impulse 未被接纳时的回调."""
        ...

    def pop_impulse(self, impulse: Impulse) -> None:
        """impulse 被 pop 的通知."""
        ...

    def peek(self, no_stale: bool = True) -> Impulse | None:
        """查看最新 impulse."""
        ...

    def is_running(self) -> bool:
        ...

    async def __aenter__(self) -> Self:
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


class DoloresEgo:
    """Dolores 的自我/连续性层. 详见模块 docstring."""

    def __init__(self, ghost: "Dolores") -> None:
        """随 ghost 一起实例化, 构造无副作用 (不碰 httpx / session / matrix.processes).

        :param ghost: back-ref, ego 经它取 session / trajectory / shell / instruction.
            待讨论: 直取 private 还是走 ghost 的 public-internal accessor
            (见 tests/CLAUDE.md 的 public-internal 约定).

        fields:
          _ghost          — back-ref
          _session        — 从 ghost 取 (on_output / on_signal / topics / output)
          _exit_stack     — 生命周期栈 (AsyncExitStack)
          _nucleus        — 固定 self-nucleus (懒构建)
          _ego_session_id — 与 plugin 的 doloresEgoSessionId 同步
          _last_error     — 持有的上一轮异常, 下一轮 run 注入上下文
          _watcher_task   — 背景 watcher task (长命线)
          _sync_task      — ego session id 后台同步轮询 task
        """
        ...

    # ── 长命线: 生命周期 ────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """进入 ghost 生命周期 (由 Dolores.__aenter__ 经 _exit_stack 进入).

        顺序: 注册固定 nucleus → 启动背景 watcher → 启动 ego session id 后台
        同步轮询 (对齐 plugin 的双机制: 显式 create RPC + 周期校准, 防漂移/重启失同步).
        注意: 晚于 Dolores 的 dsh/trajectory/ground 挂载, 保证醒来时看到 fully-wired ghost.
        """
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出: 停 watcher / 停同步轮询 / 关仍开着的 ego session."""
        ...

    # ── 短命线: run (transaction) ───────────────────────────────────

    async def run(self, moment: Moment, effort: ThinkingEffort = '') -> AsyncIterator[str]:
        """每轮 articulate 的委托 — 一个 transaction.

        ``async for`` 作用域就是 transaction 边界:
          enter: RPC open ego session (preStep lock) + 组装上下文 (percepts+hint
                 +trajectory+instruction+上一轮异常) + 订阅 session event
          loop:  监听 session event (ego tool / thinking topic / 通用 dict),
                 判定 logos, yield 出去
          exit:  RPC close (finally) — 无论成功/异常都关锁

        :param moment: articulator.moment (percepts/hint 的载体).
        :param effort: articulator.thinking_effort() 拆出; =='none' 已被上游短路.
        :yield: 判定为 logos 的字符串, 逐段交给 Dolores.articulate → send_nowait.
        """
        ...

    # ── 上下文组装 ──────────────────────────────────────────────────

    async def _assemble_context(self, moment: Moment, effort: ThinkingEffort):
        """percepts + hint + trajectory 帧 + instruction (system_prompt + ground_instruction).

        perspectives 不读 (被 trajectory 取代).
        待讨论 seam #1: 返回类型 = DSH 请求 payload 形状 (RPC 入参).
        待讨论: ground_instruction (epoch 槽位) 本轮就接, 还是留给 epoch 周期.
        """
        ...

    # ── 背景 watcher (长命线) ───────────────────────────────────────

    async def _watch_turn_start(self) -> None:
        """后台 task: 监听 turn/start 事件 → 往固定 nucleus 打一次性 impulse.

        待讨论 seam #2: turn/start 的事件源 (timer / trajectory 帧 /
        attention hook / 外部 signal)? 这是 "自驱" 和 "事件驱动" 的分界.
        """
        ...

    # ── 固定 nucleus ────────────────────────────────────────────────

    def nucleus(self) -> "DoloresEgoNucleus":
        """自醒 nucleus 句柄. 待讨论 seam #3: 一次性 impulse 的仲裁语义."""
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
        """POST /plugin-api/ego/create — 建 ego session (tool 注册 + preStep + 设 id).

        返回 ego session id, 本侧同步持有 (对齐 plugin 的 doloresEgoSessionId).
        待讨论 seam #1: params + steer 的入参形状.
        """
        ...

    async def _rpc_tool_result(self, call_id: str, result: dict) -> None:
        """POST /plugin-api/tool-result — {callId, result} 解锁 plugin 侧 pending tool."""
        ...

    # ── 异常感知 ────────────────────────────────────────────────────

    @property
    def last_error(self) -> Exception | None:
        """持有的上一轮异常, 下一轮 run 组装上下文时注入."""
        ...
