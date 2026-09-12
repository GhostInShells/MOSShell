"""ListenerNucleus — 把 ASR 输入流 (首包/分句包/尾包) 治理成 listener 感知 impulse.

故事: "listener" 是会听的器官 (对齐 g1.listen / voice-input 的命名), 它把麦克风拾到的
语音转成文本流, 再以三种包型递进给 mindflow. 这不是再做一个 app, 而是收束 ASR 输入
到一个可治理的感知单元, 取代旧的 audio-nucleus.

设计 (人类架构师 + 本 session 对齐, skeleton 供 review):
- signal meta 只定义 3 种包, 分句粒度, 不做 partial delta (delta 不稳定, 分句稳定;
  asr 发不发是它的事, 本 nucleus 只接收稳定句).
- signal.id (wire 上的 turn/segment 身份) 与 impulse.id 解耦 — nucleus 自持 impulse id,
  每个 turn 分配一个整体 id, 该 turn 的所有包共用它, 走 mindflow 的 same-id absorb.
- 三个正交核心 API (独立开关):
  1. ``first_packet_interrupt`` — 首包打断. mechanism = ``complete=False`` 高强抢占
     attention, attended 后降回运行参数 (priority→INFO, strength→正常) —
     challenge vs run 分离 (0c349f00), 让下一句能可靠打断. 首包 priority 默认 WARNING.
  2. ``clause_response`` — 分句响应. 开启后每句一个**独立 impulse id** (complete=True,
     互相打断), 携带累计 buffer; ghost 逐句思考、被下一句打断, 产生连续思考帧
     (类人聆听). **默认关闭**.
  3. 尾包 commit — 恒为高优 notice, 同 id 换成一个正常 ``complete=True`` impulse,
     等待响应; 分句响应开启时做 diff (只发未送达的分句).

与旧 AudioNucleus 的关系: 它是 "capture→ASR 泵 + 全局 interrupt" 的形状, 首包打断写死
``interrupt=True`` (shell.stop_interpretation), 非强度抢占. AudioNucleus 在 listener
落地后退役删除, 本文件是替换它的骨架.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from pydantic import Field

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType, Message, unique_id
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, Impulse,
)

__all__ = [
    "ListenerPacket",
    "ListenerSignal",
    "ListenerNucleus",
    "ListenerNucleusMeta",
    "new_listener_signal",
]


class ListenerPacket(str, Enum):
    """ASR 输入包的三种类型 — 对应 listener 器官的三段语义.

    一个 turn (一次说完的话) 由这三种包按序组成: 首包 → (分句包)* → 尾包.
    """

    FIRST = "first"    # 首包: 第一个有语义的包 (text 非空), 用于抢占/占坑; 不是 segment/turn 标记
    CLAUSE = "clause"  # 分句包: 一句稳定句 (ASR definite), 是递送内容的最小单元
    TAIL = "tail"      # 尾包: commit (VAD / 手动 / 其它), 结束本 turn


class ListenerSignal(SignalMeta):
    """Listener 感知 signal — ASR 音频流的分句结果, 进 ListenerNucleus.

    signal name = "listener". 一个 turn 的包用同一 ``turn_id`` 关联, ``clause_index``
    保序; impulse id 由 nucleus 自持, 与 turn_id 解耦 (nucleus 决定怎么映射到 attention).

    字段设计遵循 SignalMeta 三尺度: 功能性 (nucleus 判决用途) / 易生产 (ASR/wire 天然
    拿到) / 未来语义 (何时会用于分档/去重). 不承载 "给 ghost 看的消息" —— 那是信号主体,
    经 ``to_signal(messages=...)`` 承载, 不进 metadata.
    """

    packet: ListenerPacket = Field(description="包型: first / clause / tail")
    text: str = Field(default="", description="稳定句文本; 首包 = 第一个有语义的包 (text 非空); 尾包 text 不参与递送 (内容由分句 sent 态决定)")
    turn_id: str = Field(default="", description="一次 turn 的身份, 对应 asr 的 segment_id (tail 界定), 一个 turn 一个值")
    clause_index: int = Field(default=0, description="句序号, 用于 FIFO 保序 / diff / 去重")
    start_ms: int = Field(default=0, description="引擎相对起始时间 (流内), 非墙钟")
    end_ms: int = Field(default=0, description="引擎相对结束时间 (流内), 非墙钟")
    confidence: float = Field(default=0.0, description="识别置信度, 未来用于分句门控/低优丢弃")

    @classmethod
    def signal_name(cls) -> SignalName:
        return "listener"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


@dataclass
class ClauseState:
    """分句的简单状态 — nucleus FIFO 记账的最小单元.

    ``sent`` 标记该分句是否已通过 impulse 送达模型, 供尾包 diff 计算"未送达"差集.
    """

    clause_index: int
    text: str
    sent: bool = False


class ListenerNucleus(Nucleus):
    """Listener 感知单元 — ASR 包流 → listener impulse.

    结构: turn 级 impulse id (自持) + 分句 FIFO buffer + 三个正交开关.
    Cache 模式 (pull-based, 对齐 Command/Interrupt Nucleus): ``add_signal`` 立即构造
    impulse 写入 ``_impulse_cache`` 并通知 mindflow; mindflow 经 ``peek()`` 拉取,
    仲裁后经 ``attended`` / ``suppress`` 确认.

    冷却 (cooldown) 语义: 只在仲裁失败 (suppress) 时启动, 且只压首包/分句的 impulse
    **发射** — 不压 buffering, 更不压尾包 commit (内容绝不丢). 本 skeleton 表达设计
    意图, 具体仲裁路径 (同 id absorb / wait_ready 卡帧) 在 step 2 结合 Mindflow loop 实现.
    """

    NAME = "listener_nucleus"

    def __init__(
            self,
            *,
            name: str = NAME,
            first_packet_interrupt: bool = True,
            clause_response: bool = False,
            first_packet_priority: Priority = Priority.WARNING,
            first_strength: int = 150,
            normal_strength: int = 100,
            clause_priority: Priority = Priority.NOTICE,
            tail_priority: Priority = Priority.NOTICE,
            run_priority: Priority = Priority.INFO,
            suppress_seconds: float = 0.5,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._logger = logger or get_moss_logger()

        # 三个正交核心开关 (独立 API).
        self._first_packet_interrupt = first_packet_interrupt
        self._clause_response = clause_response

        # 强度/优先级旋钮.
        self._first_packet_priority = first_packet_priority
        self._first_strength = first_strength        # 首包高强, 赢得预占
        self._normal_strength = normal_strength      # attended 后降回 (运行强度)
        self._run_priority = run_priority            # attended 后降回 (运行优先级)
        self._clause_priority = clause_priority
        self._tail_priority = tail_priority
        self._suppress_seconds = suppress_seconds

        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False

        # turn 级状态 — nucleus 自持 impulse id.
        self._impulse_id: str | None = None
        self._turn_active = False
        self._clauses: list[ClauseState] = []
        self._last_clause_index = 0

        # 仲裁冷却 (失败侧) + impulse cache.
        self._suppress_until = 0.0
        self._impulse_cache: Impulse | None = None

    # ── 三个核心 API (独立开关) ──

    def set_first_packet_interrupt(self, enabled: bool) -> None:
        """API 1 — 是否开启首包打断 (首包高强 complete=False 抢占, attended 降回)."""
        self._first_packet_interrupt = enabled

    def set_clause_response(self, enabled: bool) -> None:
        """API 2 — 是否开启分句响应 (每句独立 impulse id + 互相打断, 连续思考帧)."""
        self._clause_response = enabled

    def set_first_packet_priority(self, priority: Priority) -> None:
        """首包抢占 priority — 参数化, 未来可调 (甚至开放给反身性控制)."""
        self._first_packet_priority = priority

    # ── Nucleus ABC ──

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return ("listener sense nucleus — 把 ASR 包流 (first/clause/tail) "
                "治理成 impulse, 逐包递送 or 缓冲至 commit")

    def status(self) -> str:
        if self._turn_active:
            return f"listening: {len(self._clauses)} clauses, tail pending"
        if self._impulse_cache:
            return f"pending: {self._impulse_cache.description[:50]}"
        return ""

    def signals(self) -> list[SignalName]:
        return [ListenerSignal.signal_name()]

    def clear(self) -> None:
        # 极限故障恢复: 全量重置 (turn 状态 + 冷却 + cache).
        self._reset_turn()
        self._suppress_until = 0.0

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        meta = ListenerSignal.from_signal(signal)
        if meta is None:
            return
        if meta.packet == ListenerPacket.FIRST:
            self._on_first(meta)
        elif meta.packet == ListenerPacket.CLAUSE:
            self._on_clause(meta)
        else:
            self._on_tail(meta)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 失败侧: 不重发 impulse, 但保留分句 buffer — 内容不丢.
        # 进入冷却, 只压后续首包/分句的发射 (尾包 commit 不受约束).
        self._impulse_cache = None
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> Impulse | None:
        if impulse is self._impulse_cache:
            self._impulse_cache = None
        if impulse is None:
            return None
        if self._clause_response and impulse.complete:
            # 分句/尾包被 attended → 标记已送达, 尾包 diff 不再重发.
            self._mark_clauses_sent(impulse)
        # challenge → run 参数分离 (0c349f00): 抢到 attention 后降为运行优先级(INFO)
        # + 运行强度, 让下一句可靠打断 (优先级直接赢, 不靠强度衰减).
        if impulse.priority != self._run_priority or impulse.strength != self._normal_strength:
            impulse.priority = self._run_priority
            impulse.strength = self._normal_strength
            return impulse
        return None

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse_cache is None:
            return None
        if no_stale and self._impulse_cache.is_stale():
            self._impulse_cache = None
            return None
        return self._impulse_cache

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        self.clear()

    # ── turn 处理 ──

    def _on_first(self, meta: ListenerSignal) -> None:
        # 新 turn 开始. 只重置 turn 状态, 不动冷却 (冷却跨 turn 持续, 直到到期).
        self._reset_turn()
        self._impulse_id = unique_id()
        self._turn_active = True
        self._last_clause_index = 0

        if not self._first_packet_interrupt:
            # 不开首包打断: 不抢 attention, 只开局, 等第一条内容再发 impulse.
            return
        if self._in_cooldown():
            # 冷却期内不发射抢占, 但仍开局 (后续分句会 buffer, 尾包会提交).
            return

        impulse = self._build_impulse(
            meta,
            complete=False,
            priority=self._first_packet_priority,
            strength=self._first_strength,
        )
        # 首包抢占不触发思考, 只占坑 (content 由后续 clause/tail 递送).
        impulse.thinking_effort = 'none'
        self._fire(impulse)

    def _on_clause(self, meta: ListenerSignal) -> None:
        self._ensure_turn(meta)
        self._clauses.append(ClauseState(clause_index=meta.clause_index, text=meta.text))
        self._last_clause_index = max(self._last_clause_index, meta.clause_index)

        if not self._clause_response:
            # 不开分句响应: 只累积 buffer, 等尾包一次性提交.
            return
        if self._in_cooldown():
            # 冷却期内不发射, 但分句已 buffer (内容不丢).
            return

        # 分句响应: 每句一个独立 impulse id (不走 turn 的 same-id absorb), complete=True
        # 让 ghost 逐句思考; 新分句挑战上一个分句的 attention → 连续思考帧.
        impulse = self._build_impulse(
            meta,
            complete=True,
            priority=self._clause_priority,
            strength=self._normal_strength,
            impulse_id=unique_id(),
        )
        impulse.hint = "可思考; 等待尾包或立刻回复."
        self._fire(impulse)

    def _on_tail(self, meta: ListenerSignal) -> None:
        self._ensure_turn(meta)
        # commit: 同 id, complete=True, 换成一个正常 impulse 等待响应.
        # 尾包 commit 不受冷却约束 — 内容绝不丢.
        impulse = self._build_impulse(
            meta,
            complete=True,
            priority=self._tail_priority,
            strength=self._normal_strength,
            text=self._tail_diff(),
        )
        self._turn_active = False
        self._fire(impulse)

    # ── internals ──

    def _build_impulse(
            self,
            meta: ListenerSignal,
            *,
            complete: bool,
            priority: Priority,
            strength: int,
            text: str | None = None,
            impulse_id: str | None = None,
    ) -> Impulse:
        """构建 impulse, 消息体为 buffer 或指定 text.

        text=None → 用当前 buffer (分句响应/首包); text=传入 → 覆盖 (尾包 diff).
        impulse_id 缺省用 turn 级 id (首包/尾包 same-id absorb); 分句传入 fresh id.
        """
        data = text if text is not None else self._buffer_text()
        messages = [Message.new().with_content(data)] if data else []
        return Impulse(
            source=self.name(),
            trace_id=meta.turn_id or unique_id(),
            id=impulse_id or self._impulse_id or unique_id(),
            priority=priority,
            strength=strength,
            complete=complete,
            messages=messages,
            description=meta.text[:40] or "listener turn",
        )

    def _buffer_text(self) -> str:
        return "\n".join(c.text for c in self._clauses if c.text)

    def _tail_diff(self) -> str:
        """尾包 diff — 只发未送达的分句 (内容完全由分句 sent 态决定).

        尾包自身的 text 不参与递送 — 分句未送达则发全部, 已送达则发空.
        """
        unsent = [c.text for c in self._clauses if not c.sent and c.text]
        return "\n".join(unsent)

    def _mark_clauses_sent(self, impulse: Impulse) -> None:
        # 分句 impulse 被 attended → 该 impulse 携带的累计 buffer 已进模型上下文,
        # 逐句标记 sent, 尾包 diff 据此跳过 (避免模型二次看见).
        for c in self._clauses:
            c.sent = True

    def _ensure_turn(self, meta: ListenerSignal) -> None:
        # 容错: 分句/尾包先于首包到达时 (asr 丢首包), 自动开一个新 turn.
        if self._impulse_id is None:
            self._impulse_id = unique_id()
            self._turn_active = True

    def _reset_turn(self) -> None:
        # 只重置 turn 状态, 不动冷却.
        self._impulse_id = None
        self._turn_active = False
        self._clauses.clear()
        self._last_clause_index = 0
        self._impulse_cache = None

    def _in_cooldown(self) -> bool:
        return time.monotonic() < self._suppress_until

    def _fire(self, impulse: Impulse) -> None:
        self._impulse_cache = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)


class ListenerNucleusMeta(NucleusMeta):
    """Factory meta — 让 ``moss manifests nuclei`` 可发现 ListenerNucleus."""

    def __init__(
            self,
            *,
            first_packet_interrupt: bool = True,
            clause_response: bool = False,
    ):
        self._first_packet_interrupt = first_packet_interrupt
        self._clause_response = clause_response

    def name(self) -> str:
        return ListenerNucleus.NAME

    def description(self) -> str:
        return ("listener nucleus — turns asr packet stream (first/clause/tail) "
                "into listener impulses")

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield ListenerSignal

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return ListenerNucleus(
            first_packet_interrupt=self._first_packet_interrupt,
            clause_response=self._clause_response,
            logger=logger,
        )


def new_listener_signal(
        packet: ListenerPacket,
        text: str = "",
        *,
        turn_id: str = "",
        clause_index: int = 0,
        start_ms: int = 0,
        end_ms: int = 0,
        confidence: float = 0.0,
        priority: Priority | None = None,
        description: str = "",
) -> Signal:
    """Helper — 构造一条 ``listener`` signal.

    首包: packet=FIRST, text 通常为空. 分句: packet=CLAUSE. 尾包: packet=TAIL.
    """
    return ListenerSignal(
        packet=packet,
        text=text,
        turn_id=turn_id,
        clause_index=clause_index,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
    ).to_signal(
        text,
        description=description or (text[:40] or packet.value),
        priority=priority,
    )
