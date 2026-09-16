"""ListenerController — 判停逻辑契约行为.

验证三种聆听礼仪表面: once 拿 clause 立刻 commit; always 命中关键字立刻 commit /
静默 segment_vad commit / 活动信号 (partial) reset 等待不 commit.
"""
import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from ghoshell_moss.contracts.asr import ASRInfo, RecognitionClause, RecognitionEvent, RecognitionPhase
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerSignal
from ghoshell_moss.host.listener.controller import ListenerController, ModelListenerController


class _MockState:
    def __init__(self):
        self.event_creating = []
        self.segment = []
        self.result = []
        self.committed = 0
        self.entered = asyncio.Event()
        self.exited = False

    def on_event_creating(self, cb):
        self.event_creating.append(cb)

    def on_recognition_segment(self, cb):
        self.segment.append(cb)

    def on_recognition_result(self, cb):
        self.result.append(cb)

    def commit(self):
        self.committed += 1

    async def __aenter__(self):
        self.entered.set()
        return self

    async def __aexit__(self, *args):
        self.exited = True


class _MockListener:
    def __init__(self):
        self.state = None
        self.listened = asyncio.Event()
        self.result_observers = []
        self.running = False
        self.entered = False
        self.exited = False

    def on_recognition_result(self, cb):
        self.result_observers.append(cb)
        return lambda: None

    def is_running(self):
        return self.running

    async def listen(self):
        self.state = _MockState()
        self.listened.set()
        return self.state

    async def __aenter__(self):
        self.running = True
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.running = False
        self.exited = True


class _MockASR:
    def __init__(self):
        self.configured = []

    def get_info(self):
        return ASRInfo(params={"end_window_size": 800})

    def configure(self, params):
        self.configured.append(params)


def _clause(text: str) -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id="g",
        phase=RecognitionPhase.CLAUSE, text=text,
        clause=RecognitionClause(text=text),
    )


def _partial(text: str) -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id="g",
        phase=RecognitionPhase.PARTIAL, text=text,
    )


async def _start_controller(controller, method, **kwargs):
    task = method(**kwargs)  # method 返回 Future, 内部已 spawn 状态机
    listener = controller._listener
    await listener.listened.wait()
    state = listener.state
    await state.entered.wait()
    await asyncio.sleep(0)
    return task, state


async def _stop(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_once_commits_on_clause():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(controller, controller.once)

    for cb in state.event_creating:
        await cb(_clause("你好"))
    assert state.committed == 1

    tail = RecognitionEvent(
        stream_id="s", segment_id="g",
        phase=RecognitionPhase.TAIL, text="你好",
    )
    for cb in state.result:
        cb(tail)  # 尾包处理 → once 退出
    await task


@pytest.mark.asyncio
async def test_always_commits_on_keyword():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        segment_vad=5.0, keywords=["我说完了"], timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我说完了"))
    assert state.committed == 1  # 命中关键字立刻 commit, 不等 segment_vad

    await _stop(task)


@pytest.mark.asyncio
async def test_always_commits_after_segment_vad():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        segment_vad=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
    assert state.committed == 0  # clause 后未到 segment_vad

    await asyncio.sleep(0.2)  # 超过 segment_vad (0.1s)
    assert state.committed == 1  # 静默超时 commit

    await _stop(task)


@pytest.mark.asyncio
async def test_new_method_cancels_active():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    task1 = controller.always(segment_vad=5.0, timeout=5.0)
    await listener.listened.wait()
    state1 = listener.state
    await state1.entered.wait()
    await asyncio.sleep(0)

    # 新 method 调用 → cancel 旧的状态机.
    task2 = controller.always(segment_vad=5.0, timeout=5.0)
    with contextlib.suppress(asyncio.CancelledError):
        await task1  # 等旧状态机真正结束 (cancel 传播 + __aexit__)

    assert task1.done()
    assert state1.exited  # 旧 session 已优雅关闭

    await _stop(task2)


@pytest.mark.asyncio
async def test_controller_owns_listener_when_not_running():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    await controller.__aenter__()
    assert listener.entered  # 未启动 → controller 进入并托管

    await controller.__aexit__(None, None, None)
    assert listener.exited  # 托管的 → 退出时收


@pytest.mark.asyncio
async def test_controller_borrows_listener_when_running():
    listener = _MockListener()
    await listener.__aenter__()  # 宿主已启动
    controller = ListenerController(listener=listener, asr=_MockASR())

    await controller.__aenter__()
    assert listener.entered  # 已 running → 不重复进入
    assert not listener.exited

    await controller.__aexit__(None, None, None)
    assert not listener.exited  # 借用的 → 不关闭, 归宿主


@pytest.mark.asyncio
async def test_always_commits_after_clause_silence():
    """clause 后 segment_vad 静默 commit; partial 不重置静默计时 (clause-based)."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        segment_vad=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
        await cb(_partial("你好啊"))  # partial 不重置静默计时

    await asyncio.sleep(0.2)  # 超过 segment_vad (0.1s)
    assert state.committed == 1  # clause 后静默超时 commit

    await _stop(task)


# ============================================================
# 信号发射 — RecognitionEvent → listener signal (系统级封装)
# ============================================================

def _first(text: str, segment_id: str = "g") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id=segment_id,
        phase=RecognitionPhase.FIRST, text=text,
    )


def _tail(text: str, segment_id: str = "g") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id="s", segment_id=segment_id,
        phase=RecognitionPhase.TAIL, text=text,
    )


def test_signal_broadcast_clause_not_emitted():
    emitted = []
    listener = _MockListener()
    ListenerController(listener=listener, asr=_MockASR(), signal_broadcast=emitted.append)

    assert len(listener.result_observers) == 1  # 有 sink → 注册发射观察者
    listener.result_observers[0](_clause("你好"))

    assert emitted == []  # CLAUSE 不上行 — 判停已在 listener 侧消化


def test_signal_broadcast_interrupt_then_deliver():
    emitted = []
    listener = _MockListener()
    ListenerController(listener=listener, asr=_MockASR(), signal_broadcast=emitted.append)
    cb = listener.result_observers[0]

    cb(_first("你", segment_id="t1"))
    cb(_clause("你好"))
    cb(_tail("你好", segment_id="t1"))

    assert len(emitted) == 2  # 打断包 + 发送包 (CLAUSE 不上行)
    interrupt_sig, deliver_sig = emitted
    assert interrupt_sig.complete is False
    assert deliver_sig.complete is True

    interrupt_meta = ListenerSignal.from_signal(interrupt_sig)
    deliver_meta = ListenerSignal.from_signal(deliver_sig)
    assert interrupt_meta.segment_id == "t1"
    assert deliver_meta.segment_id == "t1"  # same-id
    assert interrupt_meta.interrupt is True
    assert interrupt_sig.priority == Priority.WARNING
    assert deliver_sig.priority == Priority.INFO


def test_no_signal_broadcast_registers_no_observer():
    listener = _MockListener()
    ListenerController(listener=listener, asr=_MockASR())
    assert listener.result_observers == []  # 无 sink → 不注册, 只做判停


# ============================================================
# 智能判停 (llm judge) — ModelListenerController + StopJudge 装线
# ============================================================

class _MockCaller:
    def __init__(self, scores):
        self._scores = list(scores)

    async def run_messages(self, prompt):
        return SimpleNamespace(content=str(self._scores.pop(0)))


@pytest.mark.asyncio
async def test_llm_judge_commits_when_judge_confident():
    listener = _MockListener()
    controller = ModelListenerController(
        listener=listener, asr=_MockASR(),
        caller=_MockCaller([9]),
    )
    task, state = await _start_controller(
        controller, controller.llm_judge, judge_delay=0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我觉得应该这样"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert state.committed == 1  # 打分 9 >= threshold 7 → commit

    await _stop(task)


@pytest.mark.asyncio
async def test_llm_judge_keeps_waiting_when_judge_unsure():
    listener = _MockListener()
    controller = ModelListenerController(
        listener=listener, asr=_MockASR(),
        caller=_MockCaller([1]),
    )
    task, state = await _start_controller(
        controller, controller.llm_judge, judge_delay=0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我觉得应该这样"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert state.committed == 0  # 打分 1 < threshold 7 → 不 commit

    await _stop(task)
