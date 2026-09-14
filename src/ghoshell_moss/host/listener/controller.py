"""Listener Controller — 判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

判停逻辑 (聆听礼仪) 通过 ListenerState.on_event_creating 挂载到识别层 (inline await),
判停时调 state.commit(). commit 机制 (发负序号切段) 已在 recognition 层, 这里只决定
何时调用.

信号发射是独立于判停的第二职责: 本层是 RecognitionEvent (text axis) → listener signal
(first/clause/tail) 的生产边界. 构造时注入 ``signal_broadcast`` (signal sink), 存在时
注册一条 listener 级 on_recognition_result 观察者, 机械地把每个识别事件翻译成 listener
signal 并广播.

第三职责是运行时自解释: 追踪当前礼仪 (off/once/always), 提供合成快照 (``snapshot()``)
与随身 channel (``as_channel()``), 让模型能判断"耳朵开没开、什么模式".
"""
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Optional

import janus
from typing_extensions import Self
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.asr import ASR, RecognitionEvent, RecognitionPhase
from ghoshell_moss.contracts.listener import Listener, ListenerState
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import Publisher, TopicService
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerPacket, new_listener_signal
from ghoshell_moss.topics import ClauseTopic

__all__ = [
    "ListenerController",
    "PacketTranslator",
    "ListenEtiquette",
    "ListenerSnapshot",
]


class PacketTranslator:
    """RecognitionEvent (text axis) → listener packets (first/clause/tail).

    把 ASR 识别事件翻译成 listener 包流。``clause_index`` 按 segment 计数, FIRST 重置。
    PARTIAL 不产出包 — listener 只关心 first/clause/tail 三个语义点。
    """

    def __init__(self) -> None:
        self._clause_index = 0

    def translate(self, result: RecognitionEvent) -> list[tuple[ListenerPacket, str, int]]:
        packets: list[tuple[ListenerPacket, str, int]] = []
        if result.phase == RecognitionPhase.FIRST:
            self._clause_index = 0
            packets.append((ListenerPacket.FIRST, result.text, 0))
        elif result.phase == RecognitionPhase.CLAUSE:
            self._clause_index += 1
            clause_text = result.clause.text if result.clause else result.text
            packets.append((ListenerPacket.CLAUSE, clause_text, self._clause_index))
        elif result.phase == RecognitionPhase.TAIL:
            packets.append((ListenerPacket.TAIL, result.text, self._clause_index))
        return packets


class ListenEtiquette(str, Enum):
    """聆听礼仪 — 判停策略, 决定"什么时候算说完"."""

    OFF = "off"
    ONCE = "once"
    ALWAYS = "always"


@dataclass
class ListenerSnapshot:
    """合成快照 — 状态 (mode/listening) + ASR 当前参数值.

    温数据, 进 notice, 变了才重发. 参数 schema 是冷数据 (进 instruction), 不在此.
    """

    mode: str
    listening: bool
    asr_params: dict

    def render_notice(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def render_status(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class ListenerController:
    """判停逻辑装线 + listener signal 生产边界 + 运行时自解释.

    持有 listener (听) + asr (configure vad). once/always 是长时间运行的 async method,
    内部管理一条 listening session 的生命周期; 结果经 listener 的观察面 (on_recognition_*)
    流出.

    三个职责:
    - 判停 (on_event_creating 决定何时 commit);
    - 信号发射 (注入 signal_broadcast 时把识别事件翻译成 listener signal 广播);
    - 自解释 (追踪当前礼仪, 提供 ``snapshot()`` 与随身 ``as_channel()``).
    """

    def __init__(
            self,
            *,
            listener: Listener,
            asr: ASR,
            logger: Optional[LoggerItf] = None,
            signal_broadcast: Optional[Callable[[Signal], None]] = None,
    ):
        self._listener = listener
        self._asr = asr
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[ListenerController]"
        self._active_task: Optional[asyncio.Task] = None
        # 信号发射: 存在 sink 时注册一条 listener 级观察者 (跨 session 稳定), 机械地把
        # 每个识别事件翻译成 listener signal 并广播. 无 sink 则只做判停, 不发 signal.
        self._signal_broadcast = signal_broadcast
        self._translator = PacketTranslator()
        if signal_broadcast is not None:
            listener.on_recognition_result(self._emit_event)

        self._mode: ListenEtiquette = ListenEtiquette.OFF
        self._channel: Optional[Channel] = None
        # clause → topic 装线 (懒, 由 with_topic_service 启动).
        self._topic_task: Optional[asyncio.Task] = None
        self._topic_disposer: Optional[Callable[[], None]] = None
        self._topic_publisher: Optional[Publisher] = None

    # ── 生命周期: controller 托管 listener ──

    async def __aenter__(self) -> Self:
        await self._listener.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        await self._close_topic_wiring()
        await self._listener.__aexit__(exc_type, exc_val, exc_tb)

    # ── 聆听礼仪 ──

    def once(
            self,
            *,
            clause_vad: Optional[int] = None,
            keywords: Optional[list[str]] = None,
            timeout: float = 60.0,
    ) -> asyncio.Future:
        """半双工: 拿 clause 立刻 commit, 尾包后结束一次聆听.

        command 语义: 立即返回 Future (外部可 await 阻塞或忽略), 内部 spawn 状态机.
        新 method 调用 cancel 旧的状态机 (同一时刻至多一条 session).

        clause_vad 覆盖 ASR 分句判停时间 (end_window_size). keywords 在 once 语义下
        冗余 (clause 即 commit), 仅为接口一致保留.
        """
        self._mode = ListenEtiquette.ONCE
        self._cancel_active()
        task = asyncio.create_task(self._run_once(clause_vad=clause_vad, keywords=keywords, timeout=timeout))
        self._active_task = task
        return task

    def always(
            self,
            *,
            clause_vad: Optional[int] = None,
            speech_vad: float = 1.5,
            keywords: Optional[list[str]] = None,
            timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """持续聆听: clause 后等待 speech_vad, 活动信号 reset, 静默到 speech_vad commit.

        立即返回 Future; ``timeout=None`` 表示常驻 (直到 ``stop()`` 或新礼仪取消).
        命中 keywords 的 clause 立刻 commit (不等静默).
        """
        self._mode = ListenEtiquette.ALWAYS
        self._cancel_active()
        task = asyncio.create_task(self._run_always(
            clause_vad=clause_vad, speech_vad=speech_vad, keywords=keywords, timeout=timeout,
        ))
        self._active_task = task
        return task

    def stop(self) -> None:
        """停止聆听: 取消活跃 session, 回到 off."""
        self._mode = ListenEtiquette.OFF
        self._cancel_active()

    def snapshot(self) -> ListenerSnapshot:
        """合成当前状态快照 (mode + listening + ASR 参数值)."""
        info = self._asr.get_info()
        return ListenerSnapshot(
            mode=self._mode.value,
            listening=self._listener.is_listening(),
            asr_params=dict(info.params),
        )

    # ── 观察面 (供外部消费者) ──

    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Callable[[], None]:
        """跨 session 观察识别结果 (CLI 渲染等). 委托给 listener."""
        return self._listener.on_recognition_result(callback)

    async def listen(self) -> ListenerState:
        """开一条裸 listening session (未启动) — 供 enter 等需手动 commit 的消费者."""
        return await self._listener.listen()

    # ── clause → topic 装线 ──

    async def with_topic_service(self, service: TopicService) -> None:
        """懒装线: 把识别到的 CLAUSE 发布成 ClauseTopic 到 ``service``.

        启动一个内部持有的 drain task: ``listener.on_recognition_result`` 的回调把 CLAUSE
        结果线程安全入队, task 出队 pub. 说话人身份 role 固定 user (听侧).
        """
        publisher = service.model_publisher(creator="listener", model=ClauseTopic)
        queue: janus.Queue = janus.Queue()

        def _on_clause(result: RecognitionEvent) -> None:
            if result.phase != RecognitionPhase.CLAUSE:
                return
            clause = result.clause
            text = clause.text if clause else result.text
            queue.sync_q.put_nowait(ClauseTopic(text=text, role="user"))

        async def _drain() -> None:
            while True:
                topic = await queue.async_q.get()
                publisher.pub(topic)

        await publisher.__aenter__()
        self._topic_disposer = self._listener.on_recognition_result(_on_clause)
        self._topic_task = asyncio.create_task(_drain())
        self._topic_publisher = publisher

    async def _close_topic_wiring(self) -> None:
        """关闭 clause→topic 装线 (cancel drain task + dispose 回调 + 退出 publisher)."""
        if self._topic_disposer is not None:
            self._topic_disposer()
            self._topic_disposer = None
        if self._topic_task is not None:
            self._topic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._topic_task
            self._topic_task = None
        if self._topic_publisher is not None:
            await self._topic_publisher.__aexit__(None, None, None)
            self._topic_publisher = None

    # ── 状态机 (内部, 每个 method 一个) ──

    async def _run_once(
            self,
            *,
            clause_vad: Optional[int],
            keywords: Optional[list[str]],
            timeout: float,
    ) -> None:
        self._apply_clause_vad(clause_vad)
        state = await self._listener.listen()
        committed = False
        done = asyncio.Event()

        async def on_event(event: RecognitionEvent) -> None:
            nonlocal committed
            if event.phase == RecognitionPhase.CLAUSE and not committed:
                committed = True
                state.commit()

        def on_result(result: RecognitionEvent) -> None:
            # 结束条件 = 尾包 (TAIL) 已处理, 不是 segment 切分 — segment 切分早于
            # TAIL 经 _pump 从 queue 取出, 用 segment 判结束会丢尾包 signal.
            if result.phase == RecognitionPhase.TAIL:
                done.set()

        state.on_event_creating(on_event)
        state.on_recognition_result(on_result)

        async with state:
            try:
                await asyncio.wait_for(done.wait(), timeout)
            except asyncio.TimeoutError:
                self._logger.warning("%s once: no tail within %.1fs", self._log_prefix, timeout)

    async def _run_always(
            self,
            *,
            clause_vad: Optional[int],
            speech_vad: float,
            keywords: Optional[list[str]],
            timeout: float,
    ) -> None:
        self._apply_clause_vad(clause_vad)
        state = await self._listener.listen()

        last_activity = time.monotonic()
        waiting = False

        async def on_event(event: RecognitionEvent) -> None:
            nonlocal last_activity, waiting
            if event.phase == RecognitionPhase.CLAUSE:
                last_activity = time.monotonic()
                clause_text = event.clause.text if event.clause else event.text
                if keywords and any(k in clause_text for k in keywords):
                    waiting = False
                    state.commit()
                else:
                    waiting = True
            elif event.phase in (RecognitionPhase.FIRST, RecognitionPhase.PARTIAL):
                # 活动信号 (用户还在说): reset 等待.
                last_activity = time.monotonic()
                waiting = False

        state.on_event_creating(on_event)

        async def _watch() -> None:
            nonlocal waiting
            while True:
                await asyncio.sleep(0.05)
                if waiting and time.monotonic() - last_activity >= speech_vad:
                    state.commit()
                    waiting = False

        async with state:
            watch_task = asyncio.create_task(_watch())
            try:
                if timeout is None:
                    await asyncio.Event().wait()
                else:
                    await asyncio.sleep(timeout)
            finally:
                watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watch_task

    # ── 信号发射 (RecognitionEvent → listener signal) ──

    def _emit_event(self, result: RecognitionEvent) -> None:
        """识别事件 → 逐包翻译 → 广播. 仅在有 sink 时注册本观察者."""
        for packet, text, clause_index in self._translator.translate(result):
            self._emit_signal(packet, result, text, clause_index)

    def _emit_signal(
            self,
            packet: ListenerPacket,
            result: RecognitionEvent,
            text: str,
            clause_index: int,
    ) -> None:
        clause = result.clause
        self._signal_broadcast(new_listener_signal(
            packet,
            text,
            turn_id=result.segment_id,
            clause_index=clause_index,
            start_ms=clause.start_ms if clause else 0,
            end_ms=clause.end_ms if clause else 0,
            description=f"listener:{packet.value}",
        ))

    # ── internals ──

    def _cancel_active(self) -> None:
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()

    # ── internals ──

    def _apply_clause_vad(self, clause_vad: Optional[int]) -> None:
        """用 clause_vad 覆盖 ASR 分句判停时间, 保留其余 params."""
        if clause_vad is None:
            return
        params = dict(self._asr.get_info().params)
        params["end_window_size"] = clause_vad
        self._asr.configure(params)

    # ── 反身 channel ──

    def as_channel(self) -> Channel:
        """随身 channel (惰性构建并持有) — 把聆听礼仪暴露成模型可控制的命令面."""
        if self._channel is None:
            self._channel = self._build_channel()
        return self._channel

    def _build_channel(self) -> Channel:
        chan = new_channel(name="listener", description="语音输入控制 — 开启/关闭聆听、切换礼仪、调 ASR")

        @chan.build.instruction
        def instruction() -> str:
            info = self._asr.get_info()
            return (
                f"ASR audio contract: {info.sample_rate}Hz, {info.bits}-bit, {info.channel}ch.\n"
                f"ASR tunable params schema:\n"
                f"{json.dumps(info.params_schema, ensure_ascii=False)}"
            )

        @chan.build.notice
        def notice() -> str:
            return self.snapshot().render_notice()

        @chan.build.command(blocking=False)
        async def once(timeout: float = 60.0) -> str:
            """Hear one utterance — stop as soon as a sentence finishes (or after `timeout` seconds)."""
            self.once(timeout=timeout)
            return "listening (once)"

        @chan.build.command(blocking=False)
        async def always(silence: float = 1.5) -> str:
            """Keep listening continuously — commit after `silence` seconds of quiet."""
            self.always(speech_vad=silence, timeout=None)
            return "listening (always)"

        @chan.build.command(blocking=False)
        async def stop() -> str:
            """Stop listening."""
            self.stop()
            return "stopped"

        @chan.build.command()
        async def status() -> str:
            """Report the ear's current state (mode, listening, ASR params)."""
            return self.snapshot().render_status()

        @chan.build.command()
        async def configure_asr(params: dict) -> str:
            """Change ASR params — pass only the keys to change (schema is in the instruction)."""
            current = dict(self._asr.get_info().params)
            self._asr.configure({**current, **params})
            return "asr configured"

        return chan
