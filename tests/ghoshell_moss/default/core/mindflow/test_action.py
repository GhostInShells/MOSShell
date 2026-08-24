"""BaseAction / BaseArticulator / ActionLogosRequest / BaseActionGate 单测.

聚焦 ghost_runtime 装线消费的 Action 契约:

  _run_action                -> async with action / wait_ready() / is_aborted()
  _run_interpreter_with_action -> action.logos() / replaned / set_compiled() / wait_until_done()

本文件只装线 _action.py, 不依赖 _think.py —— 生产构建 (think 侧) 与调度 (runtime 侧)
解耦, 这里单测的是 runtime 侧消费的 Action 行为.
"""
import asyncio

import janus
import pytest

from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver, Moment
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.mindflow._action import (
    ActionLogosRequest,
    BaseAction,
    BaseActionGate,
    BaseArticulator,
)


class _AttentionStub:
    """BaseAction 只依赖 attention 的几个行为, 用鸭子类型 stub 避开 Attention ABC 全量抽象."""

    def __init__(self):
        self._aborted = False
        self._reason = ''
        self._abort_event = ThreadSafeEvent()

    def is_aborted(self) -> bool:
        return self._aborted

    def abort(self, reason) -> None:
        self._aborted = True
        self._reason = str(reason)
        self._abort_event.set()

    def abort_reason(self) -> str:
        return self._reason

    async def wait_abort(self) -> None:
        await self._abort_event.wait()


def _make_action(*, replaned: bool = False, attention=None, moments=None):
    """构造一个 BaseAction, 返回 (action, 注入的协作事件)."""
    attention = attention or _AttentionStub()
    moments = moments or BaseMomentsObserver(max_size=10)
    logos_queue = janus.Queue()
    compiled_event = ThreadSafeEvent()
    action_stop_event = ThreadSafeEvent()
    mindflow_stop_event = ThreadSafeEvent()
    action = BaseAction(
        attention=attention,
        moments=moments,
        replaned=replaned,
        logos_queue=logos_queue,
        compiled_event=compiled_event,
        action_stop_event=action_stop_event,
        mindflow_stop_event=mindflow_stop_event,
    )
    events = {
        'logos_queue': logos_queue,
        'compiled_event': compiled_event,
        'action_stop_event': action_stop_event,
        'mindflow_stop_event': mindflow_stop_event,
    }
    return action, events


# -- BaseAction: 静态契约 (ghost_runtime 读取的结构) ---------------------------------

@pytest.mark.asyncio
async def test_replaned_reflects_configured_value():
    """replaned 决定 interpreter kind (clear vs append), 必须忠实返回配置值."""
    replan, _ = _make_action(replaned=True)
    append, _ = _make_action(replaned=False)
    assert replan.replaned is True
    assert append.replaned is False


@pytest.mark.asyncio
async def test_attention_property_returns_injected_attention():
    """action 持有 attention, 供 runtime 观察/转发."""
    attention = _AttentionStub()
    action, _ = _make_action(attention=attention)
    assert action.attention is attention


@pytest.mark.asyncio
async def test_is_running_gated_by_lifecycle():
    """is_running 只在进入后、未停止、未 abort 时成立 — 合同: 运行中才消费 logos."""
    action, _ = _make_action()
    assert action.is_running() is False
    async with action:
        assert action.is_running() is True
        action.attention.abort('preempted')
        await action.wait_ready()
        assert action.is_running() is False


# -- BaseAction.wait_ready: 装线顺序 wait_ready() -> logos() ------------------------

@pytest.mark.asyncio
async def test_wait_ready_returns_after_first_meaningful_delta():
    """等第一个有语义的帧: 前导空 delta 被缓冲, 有意义 delta 到场即返回, logos 一次性吐出."""
    action, ev = _make_action()
    async with action:
        ev['logos_queue'].sync_q.put_nowait('')
        ev['logos_queue'].sync_q.put_nowait('hello ')
        ev['logos_queue'].sync_q.put_nowait('world')
        ev['logos_queue'].sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == ['hello ', 'world']


@pytest.mark.asyncio
async def test_wait_ready_empty_stream_returns_without_yield():
    """空流 (safemode 否决路径 articulator 一个 delta 都不发): wait_ready 返回, logos 零 yield, 不挂死."""
    action, ev = _make_action()
    async with action:
        ev['logos_queue'].sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == []


@pytest.mark.asyncio
async def test_wait_ready_returns_on_abort():
    """abort 触发时 wait_ready 返回 (不抛), 由调用方 is_aborted() 判别干净退出."""
    action, _ = _make_action()
    async with action:
        action.attention.abort('preempted')
        await asyncio.wait_for(action.wait_ready(), 2.0)
        assert action.is_aborted() is True


# -- BaseAction.logos: 流式消费 + executed_logos 落迹 ---------------------------------

@pytest.mark.asyncio
async def test_logos_streams_and_records_executed_logos():
    """logos() 逐一吐出 delta, 并把 executed logos 落到 moments (经观察帧读取, 公共字段)."""
    moments = BaseMomentsObserver(max_size=10)
    action, ev = _make_action(moments=moments)
    async with action:
        ev['logos_queue'].sync_q.put_nowait('abc')
        ev['logos_queue'].sync_q.put_nowait('def')
        ev['logos_queue'].sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == ['abc', 'def']
    # executed_logos 是 Results 公共字段, 通过 observe 一帧读取 — 不捅私有成员.
    assert moments.observe().previous.executed_logos == 'abcdef'


@pytest.mark.asyncio
async def test_logos_buffers_leading_empty_deltas_until_meaningful():
    """无 wait_ready 直接消费 logos: 前导空/空白 delta 被缓冲, 只在首个有意义处合并吐出一次.

    缓冲保留原始空白 (忠实还原模型输出), 仅避免把空 delta 单独吐出去污染执行.
    """
    action, ev = _make_action()
    async with action:
        ev['logos_queue'].sync_q.put_nowait('')
        ev['logos_queue'].sync_q.put_nowait('   ')
        ev['logos_queue'].sync_q.put_nowait('real')
        ev['logos_queue'].sync_q.put_nowait(None)
        got = [delta async for delta in action.logos()]
    assert got == ['   real']


# -- BaseAction: 编译 / 生命周期 / abort --------------------------------------------

@pytest.mark.asyncio
async def test_set_compiled_signals_compiled_event():
    """set_compiled() 置位编译事件 — interpreter 编译完成通知, Think 得以继续."""
    action, ev = _make_action()
    async with action:
        assert ev['compiled_event'].is_set() is False
        action.set_compiled()
        assert ev['compiled_event'].is_set() is True


@pytest.mark.asyncio
async def test_abort_sets_is_aborted_and_abort_reason():
    """abort(reason) 通过 attention 传播 + 关闭 queue; abort_reason 可回溯."""
    action, _ = _make_action()
    async with action:
        action.abort('preempted')
        assert action.is_aborted() is True
        assert action.abort_reason() == 'preempted'


@pytest.mark.asyncio
async def test_wait_until_done_awaits_pending_future():
    """wait_until_done 等待挂起的 future 完成再返回 (签名应为 append 而非 extend)."""
    action, _ = _make_action()
    async with action:
        fut = asyncio.Future()
        asyncio.get_running_loop().call_later(0.05, fut.set_result, 1)
        await asyncio.wait_for(action.wait_until_done(fut), 2.0)
        assert fut.done() is True


@pytest.mark.asyncio
async def test_wait_until_done_cancels_pending_futures_on_abort():
    """abort 时 wait_until_done 取消挂起的 future, 自身不抛 CancelledError."""
    action, _ = _make_action()
    async with action:
        fut = asyncio.Future()
        action.attention.abort('preempted')
        await asyncio.wait_for(action.wait_until_done(fut), 2.0)
        assert fut.cancelled() is True


# -- BaseArticulator: 生产者侧 (send -> 同一 queue -> action 消费) ---------------------

@pytest.mark.asyncio
async def test_articulator_send_nowait_streams_to_action_and_accumulates_moment():
    """send_nowait 入队给 action, 并把 logos 累积到 moment.logos."""
    action, ev = _make_action()
    moment = Moment()
    articulator = BaseArticulator(
        moment=moment,
        logos_queue=ev['logos_queue'],
        compiled_event=ev['compiled_event'],
        action_stop_event=ev['action_stop_event'],
    )
    async with action:
        articulator.send_nowait('hello')
        articulator.send_nowait('world')
        ev['logos_queue'].sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == ['hello', 'world']
    assert moment.logos == 'helloworld'


@pytest.mark.asyncio
async def test_articulator_send_streams_to_action_and_accumulates_moment():
    """send (背压) 同样入队给 action 并累积 moment.logos."""
    action, ev = _make_action()
    moment = Moment()
    articulator = BaseArticulator(
        moment=moment,
        logos_queue=ev['logos_queue'],
        compiled_event=ev['compiled_event'],
        action_stop_event=ev['action_stop_event'],
    )
    async with action:
        await articulator.send('foo')
        await articulator.send('bar')
        ev['logos_queue'].sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == ['foo', 'bar']
    assert moment.logos == 'foobar'


@pytest.mark.asyncio
async def test_articulator_wait_compiled_unblocks_when_compiled():
    """wait_compiled() 提交后阻塞, 直到 compiled_event 就绪 — interpreter 编译完成语义."""
    action, ev = _make_action()
    articulator = BaseArticulator(
        moment=Moment(),
        logos_queue=ev['logos_queue'],
        compiled_event=ev['compiled_event'],
        action_stop_event=ev['action_stop_event'],
    )
    task = asyncio.create_task(asyncio.wait_for(articulator.wait_compiled(), 2.0))
    await asyncio.sleep(0)
    assert task.done() is False
    ev['compiled_event'].set()
    await asyncio.wait_for(task, 2.0)


# -- ActionLogosRequest: safemode 闸门 approve / reject 通向 action -------------------

@pytest.mark.asyncio
async def test_logos_request_approve_pushes_logos_to_action():
    """approve() 把缓冲的 logos 灌进 action queue, 回调 put_action, approved()=True."""
    action, ev = _make_action()
    put = []
    request = ActionLogosRequest(logos='do-thing', action=action, put_action=put.append)
    async with action:
        await request.approve('note')
        assert request.approved() is True
        assert len(put) == 1 and put[0] is action
        got = [delta async for delta in action.logos()]
    assert got == ['do-thing']


@pytest.mark.asyncio
async def test_logos_request_reject_aborts_action():
    """reject() 置 approved()=False 并 abort 整个 action."""
    action, _ = _make_action()
    request = ActionLogosRequest(logos='bad', action=action, put_action=lambda a: None)
    msg = await request.reject('no way')
    assert request.approved() is False
    assert action.is_aborted() is True
    assert msg == 'abort the running attention'


@pytest.mark.asyncio
async def test_logos_request_settled_only_once():
    """approve/reject 只能结算一次, 重复结算抛 RuntimeError."""
    action, _ = _make_action()
    request = ActionLogosRequest(logos='x', action=action, put_action=lambda a: None)
    await request.approve('note')
    with pytest.raises(RuntimeError):
        await request.approve('again')
    with pytest.raises(RuntimeError):
        await request.reject('no')


# -- BaseActionGate: 队列式等待, stop -> None ---------------------------------------

@pytest.mark.asyncio
async def test_gate_wait_request_returns_queued_request():
    """wait_request 返回已入队的 request."""
    gate = BaseActionGate(stop_event=ThreadSafeEvent())
    action, _ = _make_action()
    request = ActionLogosRequest(logos='x', action=action, put_action=lambda a: None)
    gate.add_request(request)
    result = await asyncio.wait_for(gate.wait_request(), 2.0)
    assert result is request


@pytest.mark.asyncio
async def test_gate_wait_request_skips_done_requests():
    """已结算 (done) 的 request 被跳过, 只返回活的 request."""
    gate = BaseActionGate(stop_event=ThreadSafeEvent())
    action, _ = _make_action()
    done_request = ActionLogosRequest(logos='done', action=action, put_action=lambda a: None)
    live_request = ActionLogosRequest(logos='live', action=action, put_action=lambda a: None)
    done_request.commit()
    gate.add_request(done_request)
    gate.add_request(live_request)
    result = await asyncio.wait_for(gate.wait_request(), 2.0)
    assert result is live_request


@pytest.mark.asyncio
async def test_gate_wait_request_unblocks_when_request_arrives():
    """阻塞中的 wait_request 在 request 入队后被唤醒返回."""
    gate = BaseActionGate(stop_event=ThreadSafeEvent())
    action, _ = _make_action()
    request = ActionLogosRequest(logos='x', action=action, put_action=lambda a: None)
    task = asyncio.create_task(asyncio.wait_for(gate.wait_request(), 2.0))
    await asyncio.sleep(0)
    gate.add_request(request)
    result = await asyncio.wait_for(task, 2.0)
    assert result is request


@pytest.mark.asyncio
async def test_gate_wait_request_returns_none_on_stop():
    """stop 前置位: wait_request 立即返回 None (gate 已终止, 运行结束)."""
    stop_event = ThreadSafeEvent()
    gate = BaseActionGate(stop_event=stop_event)
    stop_event.set()
    result = await asyncio.wait_for(gate.wait_request(), 2.0)
    assert result is None


@pytest.mark.asyncio
async def test_gate_wait_request_unblocks_on_stop():
    """阻塞中的 wait_request 在 stop 后被唤醒, 返回 None 而非异常."""
    stop_event = ThreadSafeEvent()
    gate = BaseActionGate(stop_event=stop_event)
    task = asyncio.create_task(asyncio.wait_for(gate.wait_request(), 2.0))
    await asyncio.sleep(0)
    stop_event.set()
    result = await asyncio.wait_for(task, 2.0)
    assert result is None
