"""ListenerController — 判停逻辑契约行为.

验证三种聆听礼仪表面: once 拿 clause 立刻 commit; always 命中关键字立刻 commit /
静默 silence commit / 活动信号 (partial) reset 等待不 commit.
"""
import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from ghoshell_moss.contracts.asr import ASRInfo, RecognitionClause, RecognitionEvent, RecognitionPhase, RecognitionSegment
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerSignal
from ghoshell_moss.host.listener.controller import ListenerController
from ghoshell_moss.host.listener.etiquette import always, knock


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
        self.segment_observers = []
        self.running = False
        self.entered = False
        self.exited = False

    def on_recognition_result(self, cb):
        self.result_observers.append(cb)
        return lambda: None

    def on_recognition_segment(self, cb):
        self.segment_observers.append(cb)
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
        silence=5.0, keywords=["我说完了"], timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我说完了"))
    assert state.committed == 1  # 命中关键字立刻 commit, 不等 silence

    await _stop(task)


@pytest.mark.asyncio
async def test_always_commits_after_silence():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        silence=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
    assert state.committed == 0  # clause 后未到 silence

    await asyncio.sleep(0.2)  # 超过 silence (0.1s)
    assert state.committed == 1  # 静默超时 commit

    await _stop(task)


@pytest.mark.asyncio
async def test_new_method_cancels_active():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    task1 = controller.always(silence=5.0, timeout=5.0)
    await listener.listened.wait()
    state1 = listener.state
    await state1.entered.wait()
    await asyncio.sleep(0)

    # 新 method 调用 → cancel 旧的状态机.
    task2 = controller.always(silence=5.0, timeout=5.0)
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
    """clause 后 silence 静默 commit; partial 不重置静默计时 (clause-based)."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(
        controller, controller.always,
        silence=0.1, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
        await cb(_partial("你好啊"))  # partial 不重置静默计时

    await asyncio.sleep(0.2)  # 超过 silence (0.1s)
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

    assert len(listener.result_observers) == 3  # 发射 + buffer + expect 观察者
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


def test_no_signal_broadcast_registers_no_emitter():
    listener = _MockListener()
    ListenerController(listener=listener, asr=_MockASR())
    # 无 sink → 不注册发射观察者 (只剩 buffer + expect 观察者).
    assert len(listener.result_observers) == 2


def test_signal_wired_reflects_broadcast_injection():
    """signal 发射面是否接通 = signal_broadcast 注入与否 (静态事实, 不进 snapshot)."""
    listener = _MockListener()
    wired = ListenerController(listener=listener, asr=_MockASR(), signal_broadcast=lambda s: None)
    not_wired = ListenerController(listener=listener, asr=_MockASR())

    assert wired.signal_wired() is True
    assert not_wired.signal_wired() is False


def test_on_signal_emit_observes_first_and_tail():
    """FIRST/TAIL 触发时回调 hint — 发没发、发什么, 供 TUI 观测."""
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=lambda s: None,
    )
    hints: list[str] = []
    controller.on_signal_emit(hints.append)
    cb = listener.result_observers[0]

    cb(_first("你", segment_id="t1"))
    cb(_tail("你好", segment_id="t1"))

    assert hints[0].startswith("onset interrupt=True")
    assert hints[1].startswith("deliver mode=")


@pytest.mark.asyncio
async def test_on_signal_emit_observes_suppressed_emit():
    """emit=False 的礼仪: onset/deliver 被抑制时也回调 hint (不静默)."""
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=lambda s: None,
    )
    hints: list[str] = []
    controller.on_signal_emit(hints.append)
    cb = listener.result_observers[0]

    spec = always.model_copy(deep=True)
    spec.onset.emit = False
    spec.deliver.emit = False
    task = controller.run_etiquette(spec, timeout=5.0)

    cb(_first("你", segment_id="t1"))
    cb(_tail("你好", segment_id="t1"))

    assert hints[0] == "onset suppressed (emit=False)"
    assert hints[1] == "deliver suppressed (emit=False)"

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ============================================================
# expect 语音输入超时提醒 — 纯异步, 不打断模型
# ============================================================

def _expect_voice_observer(listener):
    """expect 回调是最后一个 result observer (在 buffer 之后注册)."""
    return listener.result_observers[-1]


@pytest.mark.asyncio
async def test_expect_returns_incrementing_index():
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=lambda s: None,
    )
    i1 = controller.expect(timeout=10.0)
    i2 = controller.expect(timeout=10.0)
    assert i1 == "1"
    assert i2 == "2"
    controller.stop()  # 清空 pending expect


@pytest.mark.asyncio
async def test_expect_timeout_emits_notify_signal():
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
        cell_name="ghost_cell",
    )
    index = controller.expect(timeout=0.05)
    await asyncio.sleep(0.1)

    assert len(emitted) == 1
    sig = emitted[0]
    assert sig.complete is True
    meta = ListenerSignal.from_signal(sig)
    assert meta.source == "ghost_cell"  # 不是 "asr" — 系统提示, 与识别结果区分
    assert meta.interrupt is False
    assert f"#{index}" in sig.description or f"#{index}" in _signal_text(sig)


def _signal_text(sig) -> str:
    texts = []
    for msg in sig.messages:
        for c in msg.contents:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
    return "\n".join(texts)


@pytest.mark.asyncio
async def test_expect_satisfied_by_voice_silently():
    """语音到达 → expect 满足, 不发任何 signal."""
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    controller.expect(timeout=0.2)
    voice_cb = _expect_voice_observer(listener)

    await asyncio.sleep(0.05)
    voice_cb(_first("你", segment_id="t1"))  # 语音到达 → set 所有 pending events
    await asyncio.sleep(0.3)  # 超过 timeout

    assert emitted == []  # 期待已满足, 不发超时 signal


@pytest.mark.asyncio
async def test_expect_voice_sets_all_pending():
    """一次语音到达满足所有 pending expect (各自超时都取消)."""
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    controller.expect(timeout=0.2)
    controller.expect(timeout=0.2)
    voice_cb = _expect_voice_observer(listener)

    await asyncio.sleep(0.05)
    voice_cb(_clause("你好"))
    await asyncio.sleep(0.3)

    assert emitted == []  # 语音到达, 两个 expect 都静默满足, 不发超时 signal


# ============================================================
# 空闲超时 — 长时间无识别事件自动结束 (发 signal + 关闭)
# ============================================================


@pytest.mark.asyncio
async def test_always_idle_timeout_emits_signal_and_ends():
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
        cell_name="ghost_cell",
    )
    spec = always.model_copy(deep=True)
    spec.idle_timeout = 0.05
    task = controller.run_etiquette(spec)
    await listener.listened.wait()
    state = listener.state
    await state.entered.wait()
    await asyncio.sleep(0)

    await asyncio.sleep(0.1)  # 超过 idle_timeout, 无任何识别事件
    assert len(emitted) == 1
    sig = emitted[0]
    assert sig.complete is True
    meta = ListenerSignal.from_signal(sig)
    assert meta.source == "ghost_cell"  # 系统提示, 与 asr 识别结果 (source="asr") 区分
    assert meta.interrupt is False
    assert "idle-timeout" in sig.description

    await task
    assert state.exited  # 空闲超时后会话结束


@pytest.mark.asyncio
async def test_always_activity_resets_idle_timeout():
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    spec = always.model_copy(deep=True)
    spec.idle_timeout = 0.05
    task = controller.run_etiquette(spec)
    await listener.listened.wait()
    state = listener.state
    await state.entered.wait()
    await asyncio.sleep(0)

    # 持续有识别事件 → 空闲计时不断重置, 不超时.
    for _ in range(5):
        for cb in state.result:
            cb(_partial("你好"))
        await asyncio.sleep(0.03)  # < idle_timeout
    assert emitted == []

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ============================================================
# 智能判停 (llm judge) — caller 注入 + StopJudge 装线
# ============================================================

class _MockCaller:
    def __init__(self, scores):
        self._scores = list(scores)

    async def run_messages(self, prompt):
        return SimpleNamespace(content=str(self._scores.pop(0)))


@pytest.mark.asyncio
async def test_scored_commits_when_judge_confident():
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(),
        stop_caller_factory=lambda instruction: _MockCaller([9]),
    )
    task, state = await _start_controller(
        controller, controller.scored, delay=0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我觉得应该这样"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert state.committed == 1  # 打分 9 >= threshold 7 → commit

    await _stop(task)


@pytest.mark.asyncio
async def test_scored_keeps_waiting_when_judge_unsure():
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(),
        stop_caller_factory=lambda instruction: _MockCaller([1]),
    )
    task, state = await _start_controller(
        controller, controller.scored, delay=0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我觉得应该这样"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert state.committed == 0  # 打分 1 < threshold 7 → 不 commit

    await _stop(task)


@pytest.mark.asyncio
async def test_scored_degrades_without_caller():
    """caller 未注入 → judge 件静默缺席, 礼仪退回纯 silence (不炸)."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    assert controller.can_stop_judge() is False

    task, state = await _start_controller(
        controller, controller.scored,
        silence=0.1, delay=0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("我觉得应该这样"))
    await asyncio.sleep(0.2)
    assert state.committed == 1  # silence 兜底仍生效

    await _stop(task)


@pytest.mark.asyncio
async def test_with_stop_detector_replaces_assembly():
    """出口位点可整段替换: 注入的 factory 拿到 (礼仪, commit), 自行决定判停单元."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    seen: list[str] = []

    class _Replace:
        def __init__(self, name, commit):
            self._name = name
            self._commit = commit

        async def feed(self, event):
            self._commit()

        def close(self):
            pass

    def factory(etiquette, commit):
        seen.append(etiquette.name)
        return _Replace(etiquette.name, commit)

    controller.with_stop_detector(factory)
    task, state = await _start_controller(
        controller, controller.always, silence=5.0, timeout=5.0,
    )

    for cb in state.event_creating:
        await cb(_clause("你好"))
    assert seen == ["always"]  # 装配走的是注入的 factory
    assert state.committed == 1

    await _stop(task)


# ============================================================
# ListenLifecycle 表面 — pause 急停/恢复 (moss runtime 级联入口)
# ============================================================


@pytest.mark.asyncio
async def test_pause_stops_active_session():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(controller, controller.always, silence=5.0, timeout=5.0)

    controller.pause(True)
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.done()
    assert state.exited  # 急停 → 活跃 session 优雅关闭


@pytest.mark.asyncio
async def test_pause_resumes_default_etiquette():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task, state = await _start_controller(controller, controller.always, silence=5.0, timeout=5.0)

    controller.pause(True)
    with contextlib.suppress(asyncio.CancelledError):
        await task

    controller.pause(False)
    for _ in range(3):
        await asyncio.sleep(0)
    assert listener.state is not None
    assert listener.state is not state  # 恢复 → 新 session

    controller.stop()


# ============================================================
# 人类锁 (pause) — channel 消失 + run_etiquette 门控
# ============================================================


@pytest.mark.asyncio
async def test_paused_run_etiquette_is_gated():
    """锁着时 run_etiquette 返回已完成 future, 不启动 listener session."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    controller.pause(True)  # 上锁

    fut = controller.always(silence=5.0)
    await fut  # 已完成 future, 立刻返回
    assert not listener.listened.is_set()  # listener.listen() 未被调用
    assert controller.active_etiquette() is None  # 未激活任何礼仪


@pytest.mark.asyncio
async def test_paused_channel_is_unavailable_to_model():
    """锁着时 channel available=False → 整个 channel 从模型面消失."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    channel = controller.as_channel()

    assert channel.build.is_available() is True  # 未锁, 模型可见
    controller.pause(True)
    assert channel.build.is_available() is False  # 锁后消失


@pytest.mark.asyncio
async def test_paused_reflected_in_snapshot():
    """人类锁进入 snapshot — TUI / 日志与 controller 同源."""
    listener = _MockListener()
    # snapshot 读 listener.is_listening; mock 补上 default False.
    listener.is_listening = lambda: False  # type: ignore[method-assign]
    controller = ListenerController(listener=listener, asr=_MockASR())

    assert controller.snapshot().paused is False
    controller.pause(True)
    assert controller.snapshot().paused is True
    assert controller.is_paused() is True


# ============================================================
# segment buffer 感知 (deliver.emit=False 强制留存) — notice 门控
# ============================================================

def _segment(text: str, segment_id: str = "g") -> RecognitionSegment:
    return RecognitionSegment(id=segment_id, stream_id="s", text=text)


@pytest.mark.asyncio
async def test_emit_true_no_last_heard():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    # 无激活礼仪 → 事件不入 buffer, notice 无 last_heard.
    listener.result_observers[0](_partial("你好"))
    listener.segment_observers[0](_segment("你好"))

    notices = await controller.as_channel().build.get_named_notices()
    assert notices["last_heard"] is None


@pytest.mark.asyncio
async def test_emit_false_retains_and_reflects_last_heard():
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())

    # deliver.emit=False → 强制留存 (单一旋钮), notice 暴露 last_heard.
    spec = always.model_copy(deep=True)
    spec.deliver.emit = False
    task = controller.run_etiquette(spec, timeout=5.0)

    listener.result_observers[0](_partial("你好"))
    listener.segment_observers[0](_segment("你好"))

    notices = await controller.as_channel().build.get_named_notices()
    assert notices["last_heard"] == "你好"

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ============================================================
# 人类强发送 (send_now) — 输入法语义
# ============================================================

@pytest.mark.asyncio
async def test_send_now_drains_buffer_when_retaining():
    """emit=False (输入法): send_now drain buffer 并发一条 deliver signal."""
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    spec = always.model_copy(deep=True)
    spec.deliver.emit = False
    task = controller.run_etiquette(spec, timeout=5.0)

    listener.result_observers[0](_partial("你好"))
    listener.segment_observers[0](_segment("你好"))

    result = controller.send_now()
    assert result == "sent"
    assert len(emitted) == 1
    sig = emitted[0]
    assert sig.complete is True
    assert ListenerSignal.from_signal(sig).interrupt is False
    assert "你好" in _signal_text(sig)

    # 已 drain → 再 send 为空, 不发 signal.
    result2 = controller.send_now()
    assert result2 == "buffer empty — nothing to send"
    assert len(emitted) == 1

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_send_now_empty_buffer_sends_nothing():
    """buffer 空 → 提示空, 不发送任何东西."""
    emitted = []
    listener = _MockListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    spec = always.model_copy(deep=True)
    spec.deliver.emit = False
    task = controller.run_etiquette(spec, timeout=5.0)

    result = controller.send_now()
    assert result == "buffer empty — nothing to send"
    assert emitted == []

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_send_now_force_commits_when_auto_send():
    """emit=True (自动): send_now 强 commit, 走现有 commit 链路, 不自己发 signal."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    # 直接启动 always (emit=True 默认), 拿当前 session.
    task, state = await _start_controller(controller, controller.always, silence=5.0, timeout=5.0)

    result = controller.send_now()
    assert result == "committed"
    assert state.committed == 1  # 强 commit 切段

    await _stop(task)


@pytest.mark.asyncio
async def test_send_now_no_sink_drains_only():
    """无 signal sink → send_now 只 drain, 不发 signal (副作用只有 drain)."""
    listener = _MockListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    spec = always.model_copy(deep=True)
    spec.deliver.emit = False
    task = controller.run_etiquette(spec, timeout=5.0)

    listener.result_observers[0](_partial("你好"))
    listener.segment_observers[0](_segment("你好"))

    result = controller.send_now()  # 无 sink, 不发 signal 但 drain
    assert result == "sent"
    result2 = controller.send_now()
    assert result2 == "buffer empty — nothing to send"  # 已 drain

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ============================================================
# knock 中间态 — 空闲降级 + 能量检测 (声音门铃)
# ============================================================


class _MockSoundListener(_MockListener):
    """支持能量检测的 mock listener — 手动 trigger_sound 模拟"检测到声音". """

    def __init__(self):
        super().__init__()
        self.sound_detected_callbacks = []
        self.detection_started = asyncio.Event()
        self.detection_stopped = asyncio.Event()
        self.detection_params = {}

    def on_sound_detected(self, callback):
        self.sound_detected_callbacks.append(callback)
        return lambda: self.sound_detected_callbacks.remove(callback)

    def start_sound_detection(self, *, threshold_db, cooldown):
        self.detection_params = {"threshold_db": threshold_db, "cooldown": cooldown}
        self.detection_started.set()

    def stop_sound_detection(self):
        self.detection_stopped.set()

    def trigger_sound(self):
        for cb in list(self.sound_detected_callbacks):
            cb()


@pytest.mark.asyncio
async def test_knock_etiquette_detects_sound_without_asr():
    """knock 礼仪走能量检测, 不启动 ASR (listen 不被调用)."""
    listener = _MockSoundListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    task = controller.run_etiquette(knock.model_copy(deep=True))

    await listener.detection_started.wait()
    assert not listener.listened.is_set()  # 不做 ASR — listen() 未被调用

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_knock_emits_knock_signal_on_sound():
    """检测到声音 → 发一条 knock signal."""
    emitted = []
    listener = _MockSoundListener()
    controller = ListenerController(
        listener=listener, asr=_MockASR(), signal_broadcast=emitted.append,
    )
    task = controller.run_etiquette(knock.model_copy(deep=True))
    await listener.detection_started.wait()

    listener.trigger_sound()
    assert len(emitted) == 1
    assert emitted[0].name == "knock"

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_always_idle_degrades_to_knock():
    """always 空闲超时 → 降级到 knock 中间态 (不彻底结束)."""
    listener = _MockSoundListener()
    controller = ListenerController(listener=listener, asr=_MockASR())
    spec = always.model_copy(deep=True)
    spec.idle_timeout = 0.05
    task = controller.run_etiquette(spec)
    await listener.listened.wait()
    state = listener.state
    await state.entered.wait()
    await asyncio.sleep(0)

    await asyncio.sleep(0.1)  # 超过 idle_timeout → 降级
    await listener.detection_started.wait()  # knock 启动能量检测

    assert controller.active_etiquette().name == "knock"

    controller.stop()
    with contextlib.suppress(asyncio.CancelledError):
        await task
