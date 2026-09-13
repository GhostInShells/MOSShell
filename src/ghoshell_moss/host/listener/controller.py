"""Listener Controller — 判停逻辑装线 + listener signal 生产边界 (系统级封装).

判停逻辑 (聆听礼仪) 通过 ListenerState.on_event_creating 挂载到识别层 (inline await),
判停时调 state.commit(). commit 机制 (发负序号切段) 已在 recognition 层, 这里只决定
何时调用. 第 4 步先做 once / always / 关键字 三种表面; llm 校验 / 快捷响应后置.

信号发射是独立于判停的第二职责: 本层是 RecognitionEvent (text axis) → listener signal
(first/clause/tail) 的生产边界. 构造时注入 ``signal_broadcast`` (signal sink), 存在时
注册一条 listener 级 on_recognition_result 观察者, 机械地把每个识别事件翻译成 listener
signal 并广播. signal 发射封装在本层, 而非散落在各调用方 (CLI/ghost) 的 on_event 逻辑里.
"""
import asyncio
import contextlib
import logging
import time
from typing import Callable, Optional

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.asr import ASR, RecognitionEvent, RecognitionPhase
from ghoshell_moss.contracts.listener import Listener
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerPacket, new_listener_signal

__all__ = ["ListenerController", "PacketTranslator"]


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


class ListenerController:
    """判停逻辑装线 + listener signal 生产边界.

    持有 listener (听) + asr (configure vad). once/always 是长时间运行的 async method,
    内部管理一条 listening session 的生命周期; 结果经 listener 的观察面 (on_recognition_*)
    流出.

    两个职责: 判停 (on_event_creating 决定何时 commit) 与信号发射 (注入 signal_broadcast
    时, 把识别事件机械翻译成 listener signal 广播). 无 sink 则只做判停.
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
            timeout: float = 60.0,
    ) -> asyncio.Future:
        """持续聆听: clause 后等待 speech_vad, 活动信号 reset, 静默到 speech_vad commit.

        command 语义: 立即返回 Future, 内部 spawn 状态机, 新 method 调用 cancel 旧的.
        命中 keywords 的 clause 立刻 commit (不等静默). 真实 commit 时机 = clause_vad
        (分句判停) + speech_vad (静默等待).
        """
        self._cancel_active()
        task = asyncio.create_task(self._run_always(
            clause_vad=clause_vad, speech_vad=speech_vad, keywords=keywords, timeout=timeout,
        ))
        self._active_task = task
        return task

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
