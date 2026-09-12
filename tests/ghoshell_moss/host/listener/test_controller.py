"""ListenerController — 判停逻辑契约行为.

验证三种聆听礼仪表面: once 拿 clause 立刻 commit; always 命中关键字立刻 commit /
静默 speech_vad commit / 活动信号 (partial) reset 等待不 commit.
"""
import asyncio
import contextlib

import pytest

from ghoshell_moss.contracts.asr import ASRInfo, Clause, RecognitionEvent, RecognitionPhase
from ghoshell_moss.host.listener.controller import ListenerController


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

    async def listen(self):
        self.state = _MockState()
        self.listened.set()
        return self.state


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
        clause=Clause(text=text),
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
        speech_vad=5.0, keywords=["我说完了"], timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我说完了"))
    assert state.committed == 1  # 命中关键字立刻 commit, 不等 speech_vad

    await _stop(task)


@pytest.mark.asyncio
async def test_always_commits_after_speech_vad():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        speech_vad=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
    assert state.committed == 0  # clause 后未到 speech_vad

    await asyncio.sleep(0.2)  # 超过 speech_vad (0.1s)
    assert state.committed == 1  # 静默超时 commit

    await _stop(task)


@pytest.mark.asyncio
async def test_new_method_cancels_active():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    task1 = controller.always(speech_vad=5.0, timeout=5.0)
    await listener.listened.wait()
    state1 = listener.state
    await state1.entered.wait()
    await asyncio.sleep(0)

    # 新 method 调用 → cancel 旧的状态机.
    task2 = controller.always(speech_vad=5.0, timeout=5.0)
    with contextlib.suppress(asyncio.CancelledError):
        await task1  # 等旧状态机真正结束 (cancel 传播 + __aexit__)

    assert task1.done()
    assert state1.exited  # 旧 session 已优雅关闭

    await _stop(task2)


@pytest.mark.asyncio
async def test_always_resets_on_partial():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        speech_vad=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))     # 启动等待
        await cb(_partial("你好啊"))  # 活动信号 reset 等待

    await asyncio.sleep(0.2)  # 超过 speech_vad, 但已 reset
    assert state.committed == 0  # 不 commit

    await _stop(task)
