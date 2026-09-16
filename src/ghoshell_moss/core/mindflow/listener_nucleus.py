"""ListenerNucleus — 把 listener 侧已决定的意图协议化成 listener impulse.

listener 侧 (ListenerController) 已经自己组织好"何时打断 / 何时发送 / 什么优先级 /
什么失败侧模式", 本 nucleus 的角色是**听觉**的协议映射: 把一条 listener signal 机械
映射成 impulse, 不做理解.

两条映射 (以 ``Signal.complete`` 区分):

- ``complete=False`` → 打断包: ``interrupt`` + 高强 + ``thinking_effort='none'`` 抢占
  注意力占坑. 抢占失败 (suppress) 丢弃 — 首包不携带内容, 丢弃不丢信息. 冷却期
  (suppress 或 attended) 内不发射, 防多源/单源连续抢占风暴.
- ``complete=True`` → 发送包: ``mode=notify`` (默认) + 端侧配 priority (默认 INFO)
  完整响应. 抢占失败 buffer 进历史 (notify) — 内容绝不丢, 不受冷却约束.

same-id 链: 首包与发送包共享 ``segment_id`` (asr 的 segment id) 作为 impulse id,
让 mindflow 走 same-id absorb (首包占坑 → 发送包填充), 构成"打断 → 响应"的连续语义.
跨 listener 实例也靠 segment_id 天然区分.

冷却双档: ``suppress`` (仲裁失败) 大冷却, ``attended`` (抢占成功) 小冷却 — 后者防
同一源抢到之后立刻被自己下一个包再抢.

消息体包在 ``<listen source="..." created="...">`` 里: 来源由 ``source`` 区分
(asr / wake_word ...), ``created`` 是 signal 到达墙钟. 说话人 / 声纹等附加信息是
impulse 不具备的字段, 由端侧组织进消息体.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from pydantic import Field

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message, unique_id
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Nucleus, NucleusMeta, Priority, Signal, SignalMeta, SignalName,
)

__all__ = [
    "ListenerSignal",
    "ListenerNucleus",
    "ListenerNucleusMeta",
    "new_listener_signal",
]


class ListenerSignal(SignalMeta):
    """Listener 感知 signal — 退化为一种, meta 全量自解释.

    listener 侧已决定好意图, 本 signal 只表达这些意图, nucleus 机械映射成 impulse.
    ``complete`` / ``priority`` / ``hint`` / ``description`` 是 Signal 通用字段 (不进
    metadata), 端侧经 ``to_signal(complete=..., priority=..., hint=...)`` 直接设.

    metadata 字段 (nucleus 读它们填 impulse):

    - ``source``: 聆听来源 (asr / wake_word ...), 递送消息 ``<listen>`` tag 的
      source attribute.
    - ``segment_id``: same-id 键, 对应 asr 的 segment_id (tail 界定). 首包
      (complete=False) 与发送包 (complete=True) 共享此 id → same-id absorb.
    - ``interrupt``: 模型 attended 这个 signal 前是否停下当前行为 (barge_in). 端侧可配.
    - ``mode``: 失败侧模式 (notify / aside / ''). 端侧可配, 发送包默认 notify.
    - ``logos``: command logos 条件反射, 端侧可配.

    文本 / 说话人 / 声纹等附加信息是 impulse 不具备的字段, 端侧组织进消息体.
    """

    source: str = Field(default="asr", description="聆听来源 — 产出该包的上游感知源 (asr / wake_word ...)")
    segment_id: str = Field(default="", description="same-id 键 — 对应 asr 的 segment_id, 首包与发送包共享")
    interrupt: bool = Field(default=False, description="模型 attended 前是否停下当前行为 (barge_in)")
    mode: str = Field(default="", description="失败侧模式: notify / aside / ''(default). 发送包默认 notify")
    logos: str = Field(default="", description="command logos 条件反射, 随 impulse 发送")

    @classmethod
    def signal_name(cls) -> SignalName:
        return "listener"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO

    @classmethod
    def xml_tag(cls) -> str:
        """递送消息的 xml tag 名 — 所有聆听包共用一个 tag, 来源由 ``source`` attribute 区分."""
        return "listen"


class ListenerNucleus(Nucleus):
    """Listener 感知单元 — listener signal → impulse 的纯协议映射.

    只做两件事: 首包打断 (complete=False → interrupt impulse) 与发送包 notify
    (complete=True → notify impulse). 不累积分句、不做 diff、不管理递送范式 —
    那些理解都在 listener 侧.

    冷却 (cooldown) 双档: ``suppress`` 大冷却 + ``attended`` 小冷却, 只压打断包
    发射, 不压发送包 (内容绝不丢). same-id = ``segment_id``.
    """

    NAME = "listener_nucleus"

    def __init__(
            self,
            *,
            name: str = NAME,
            first_strength: int = 150,
            normal_strength: int = 100,
            suppress_seconds: float = 0.5,
            attended_seconds: float = 0.1,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._logger = logger or get_moss_logger()

        self._first_strength = first_strength        # 首包高强, 赢得预占
        self._normal_strength = normal_strength      # attended 后降回 (运行强度)
        self._suppress_seconds = suppress_seconds    # 仲裁失败大冷却
        self._attended_seconds = attended_seconds    # 抢占成功小冷却

        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False

        # 冷却双档 + impulse cache.
        self._suppress_until = 0.0
        self._attended_until = 0.0
        self._impulse_cache: Impulse | None = None

        # 反身 channel — 惰性构建, 一次生成后持有.
        self._channel: Channel | None = None

    # ── Nucleus ABC ──

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return ("listener sense nucleus — 首包打断 (interrupt impulse) 与发送包 "
                "notify impulse 的纯协议映射")

    def status(self) -> str:
        if self._impulse_cache:
            return f"pending: {self._impulse_cache.description[:50]}"
        if self._in_cooldown():
            return "cooldown"
        return ""

    def signals(self) -> list[SignalName]:
        return [ListenerSignal.signal_name()]

    def clear(self) -> None:
        self._suppress_until = 0.0
        self._attended_until = 0.0
        self._impulse_cache = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        meta = ListenerSignal.from_signal(signal)
        if meta is None:
            return
        if signal.complete:
            self._on_deliver(meta, signal)
        else:
            self._on_interrupt(meta, signal)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 仲裁失败: 丢弃缓存, 进入大冷却 — 只压打断包发射, 发送包不受约束.
        self._impulse_cache = None
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> Impulse | None:
        if impulse is self._impulse_cache:
            self._impulse_cache = None
        # 抢占成功也进小冷却 — 防同一源抢到后立刻又被自己下一个包抢.
        self._attended_until = time.monotonic() + self._attended_seconds
        if impulse is None:
            return None
        # challenge → run 参数分离: 抢到 attention 后降为运行优先级(INFO) + 运行强度,
        # 让下一句可靠打断 (优先级直接赢, 不靠强度衰减).
        if impulse.priority != Priority.INFO or impulse.strength != self._normal_strength:
            impulse.priority = Priority.INFO
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

    def as_channel(self) -> Channel | None:
        if self._channel is None:
            self._channel = self._build_channel()
        return self._channel

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        self.clear()

    # ── 两套协议映射 ──

    def _on_interrupt(self, meta: ListenerSignal, signal: Signal) -> None:
        # 打断包: 冷却期 (suppress 或 attended) 内不发射 — 防抢占风暴.
        if self._in_cooldown():
            return
        impulse = Impulse(
            source=self.name(),
            trace_id=meta.segment_id or unique_id(),
            id=meta.segment_id or unique_id(),
            priority=signal.priority,
            strength=self._first_strength,
            complete=False,
            interrupt=meta.interrupt,
            mode=meta.mode,
            thinking_effort='none',
            messages=[],
            description=signal.description,
        )
        self._fire(impulse)

    def _on_deliver(self, meta: ListenerSignal, signal: Signal) -> None:
        # 发送包: notify (默认) + 端侧配 priority. 不受冷却约束 — 内容绝不丢.
        impulse = Impulse(
            source=self.name(),
            trace_id=meta.segment_id or unique_id(),
            id=meta.segment_id or unique_id(),
            priority=signal.priority,
            strength=self._normal_strength,
            complete=True,
            interrupt=meta.interrupt,
            mode=meta.mode or ChallengeMode.notify.value,
            logos=meta.logos,
            hint=signal.hint,
            messages=self._wrap_messages(signal, meta.source),
            description=signal.description,
        )
        self._fire(impulse)

    # ── internals ──

    def _in_cooldown(self) -> bool:
        now = time.monotonic()
        return now < self._suppress_until or now < self._attended_until

    def _fire(self, impulse: Impulse) -> None:
        self._impulse_cache = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def _wrap_messages(self, signal: Signal, source: str) -> list[Message]:
        """把 signal 的文本消息包进 ``<listen source=... created=...>`` tag."""
        data = self._signal_text(signal)
        if not data:
            return []
        return [self._listen_message(data, source, signal.created_at)]

    def _signal_text(self, signal: Signal) -> str:
        """汇总 signal 所有消息体的文本 content."""
        texts: list[str] = []
        for msg in signal.messages:
            for c in msg.contents:
                if isinstance(c, dict) and c.get('type') == 'text':
                    texts.append(c.get('text', ''))
        return '\n'.join(texts)

    def _listen_message(self, data: str, source: str, created_at) -> Message:
        """一条 <listen source=... created=...> 包裹的消息. created 取 signal 到达墙钟."""
        attributes = {"source": source} if source else None
        message = Message.new(tag=ListenerSignal.xml_tag(), attributes=attributes, timestamp=True)
        message.meta.created = created_at
        return message.with_content(data)

    # ── 反身 channel (最小平面) ──

    def _build_channel(self) -> Channel:
        from ghoshell_moss.core.blueprint.states_channel import new_prime_channel

        channel = new_prime_channel(self.name(), description=self.description())

        @channel.build.notice
        def notice() -> str:
            return self.status() or "listening ready"

        @channel.build.command(name="configure")
        async def _configure(
                first_strength: int | None = None,
                suppress_seconds: float | None = None,
                attended_seconds: float | None = None,
        ) -> str:
            """Dynamically tune the barge-in parameters (first-packet strength / cooldowns)."""
            if first_strength is not None:
                self._first_strength = first_strength
            if suppress_seconds is not None:
                self._suppress_seconds = suppress_seconds
            if attended_seconds is not None:
                self._attended_seconds = attended_seconds
            return (
                f"configured: first_strength={self._first_strength} "
                f"suppress={self._suppress_seconds}s attended={self._attended_seconds}s"
            )

        return channel


class ListenerNucleusMeta(NucleusMeta):
    """Factory meta — 让 ``moss manifests nuclei`` 可发现 ListenerNucleus."""

    def name(self) -> str:
        return ListenerNucleus.NAME

    def description(self) -> str:
        return ("listener nucleus — 首包打断 (interrupt) 与发送包 notify 的纯协议映射")

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield ListenerSignal

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return ListenerNucleus(logger=logger)


def new_listener_signal(
        text: str = "",
        *,
        segment_id: str = "",
        source: str = "asr",
        interrupt: bool = False,
        mode: str = "",
        logos: str = "",
        complete: bool = True,
        priority: Priority | None = None,
        description: str = "",
        hint: str = "",
) -> Signal:
    """Helper — 构造一条 ``listener`` signal.

    打断包: ``complete=False`` + ``interrupt`` + 高 priority (端侧配, 默认由 nucleus 用
    signal.priority). 发送包: ``complete=True`` + ``mode=notify`` (默认) + INFO priority.
    ``segment_id`` 是首包/发送包的 same-id 键.
    """
    return ListenerSignal(
        source=source,
        segment_id=segment_id,
        interrupt=interrupt,
        mode=mode,
        logos=logos,
    ).to_signal(
        text,
        description=description or (text[:40] or ("barge-in" if not complete else "deliver")),
        priority=priority,
        hint=hint,
        complete=complete,
    )
