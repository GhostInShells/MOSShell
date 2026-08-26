"""BaseThinking 单测 — thinking 与 action 的装线契约.

聚焦 ghost_runtime 消费的 Thinking 契约:

  _run_thinking            -> async with thinking / thinking.moment
                              thinking.effort() / thinking.articulator() / wait_until_done(*tasks)

本文件只装线 _think.py (以及它驱动的 _action.py 的 BaseAction/BaseArticulator),
不搭 _mindflow.py 全量调度. 关键协议承诺:

  - moment: 首次访问懒观测一帧, observe() 逐帧替换
  - effort: 从 attention.draw_from() 的发起 impulse 读, 不从 thinking 单独携带
  - warrant: 直接持有, per-thinking 单例
  - articulator 接入策略随 warrant 是否注册切换:
      非 gated -> 立即 put_action (action 进 action 循环), logos 直入共享 queue
      gated    -> commit 时创建审批 task 并 await warrant 裁决, 通过才 put_action, 拒绝则 abort
  - 生命周期: enter/exit/stop/abort/wait_abort/wait_until_done 与 Action 对齐
"""
import asyncio

import janus
import pytest

from ghoshell_moss.core.blueprint.mindflow import Action, Impulse, Priority
from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.mindflow import BaseAttention, BaseThinking


def _make_attention(*, thinking_effort: str = 'medium') -> BaseAttention:
    """构造正式 BaseAttention, effort 由发起 impulse 的 thinking_effort 决定."""
    return BaseAttention(impulse=Impulse(
        source='test',
        priority=Priority.NOTICE,
        thinking_effort=thinking_effort,
        strength=100,
    ))


def _make_thinking(*, attention=None, observer=None, gated: bool = False, thinking_effort: str = 'medium'):
    """构造 BaseThinking, 返回 (thinking, {put, observer, mindflow_stop_event})."""
    attention = attention or _make_attention(thinking_effort=thinking_effort)
    observer = observer or BaseMomentsObserver(max_size=10)
    put = []
    mindflow_stop_event = ThreadSafeEvent()
    thinking = BaseThinking(
        attention=attention,
        observer=observer,
        put_action=put.append,
        mindflow_stop_event=mindflow_stop_event,
    )
    if gated:
        async def _approve_always(logos: str) -> tuple[bool, str]:
            return True, ''

        thinking.register_gate(_approve_always)
    return thinking, {'put': put, 'observer': observer, 'mindflow_stop_event': mindflow_stop_event}


# -- moment / observer --------------------------------------------------------

@pytest.mark.asyncio
async def test_moment_lazily_observes_first_frame_once():
    """首次访问 moment 观测一帧 (进 observer 历史); 二次访问不重复观测."""
    thinking, env = _make_thinking()
    assert len(env['observer'].moments()) == 0  # 访问前无帧

    first = thinking.moment
    assert len(env['observer'].moments()) == 1
    assert thinking.moment is first
    assert len(env['observer'].moments()) == 1  # 懒观测只发生一次


@pytest.mark.asyncio
async def test_observe_replaces_held_moment():
    """observe() 生成新帧并替换 thinking 持有的 moment."""
    thinking, env = _make_thinking()
    first = thinking.moment
    second = thinking.observe()
    assert second is not first
    assert thinking.moment is second
    assert len(env['observer'].moments()) == 2


@pytest.mark.asyncio
async def test_attention_and_observer_properties():
    """attention / observer 属性忠实回传注入对象."""
    attention = _make_attention()
    observer = BaseMomentsObserver(max_size=10)
    thinking, _ = _make_thinking(attention=attention, observer=observer)
    assert thinking.attention is attention
    assert thinking.observer is observer


# -- effort -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_effort_read_from_attention_impulse():
    """effort 来自 attention.draw_from() 的发起 impulse — 不从 thinking 单独携带."""
    thinking, _ = _make_thinking(thinking_effort='high')
    assert thinking.effort() == 'high'


@pytest.mark.asyncio
async def test_effort_none_short_circuits_runtime():
    """effort == 'none' 是 ghost_runtime 提前 return 的开关, 必须从 attention 如实读出."""
    thinking, _ = _make_thinking(thinking_effort='none')
    assert thinking.effort() == 'none'


# -- articulator 接入策略 ------------------------------------------------------

@pytest.mark.asyncio
async def test_articulator_non_gated_dispatches_action_immediately():
    """非 gated: articulator() 立即 put_action, logos 直入共享 queue 并被 action 消费."""
    thinking, env = _make_thinking()
    articulator = thinking.articulator(replan=True)

    assert len(env['put']) == 1
    action = env['put'][0]
    assert action.replaned is True

    async with action:
        articulator.send_nowait('do a thing')
        action.logos_queue.sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
    assert got == ['do a thing']


@pytest.mark.asyncio
async def test_gated_articulator_defers_action_until_commit():
    """gated: 注册 approve 回调后, articulator 在 commit 前不投递 action, commit 时才投递."""
    thinking, env = _make_thinking(gated=True)
    articulator = thinking.articulator()

    articulator.send_nowait('gated thing')  # 缓冲
    assert env['put'] == []  # commit 前尚未投递 action

    async def run_commit():
        async with articulator:
            await articulator.wait_action_done()

    task = asyncio.create_task(run_commit())
    # commit → approve(True) → 投递 action.
    for _ in range(100):
        if env['put']:
            break
        await asyncio.sleep(0.01)
    assert len(env['put']) == 1

    action = env['put'][0]
    async with action:
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]
        action.set_compiled()
    assert got == ['gated thing']
    await asyncio.wait_for(task, 2.0)


# -- articulator ↔ action 1:1 混合 --------------------------------------------

@pytest.mark.asyncio
async def test_articulator_action_one_to_one_shared_queue():
    """1:1 混合: 一个 articulator 的 delta 恰好流入它配对的唯一 action (共享 logos_queue)."""
    thinking, env = _make_thinking()
    articulator = thinking.articulator()
    assert len(env['put']) == 1
    action = env['put'][0]

    async with action:
        articulator.send_nowait('hello ')
        articulator.send_nowait('world')
        action.logos_queue.sync_q.put_nowait(None)
        await asyncio.wait_for(action.wait_ready(), 2.0)
        got = [delta async for delta in action.logos()]

    assert got == ['hello ', 'world']


@pytest.mark.asyncio
async def test_articulator_action_one_to_one_compiled_handshake():
    """1:1 混合: articulator.wait_compiled() 被配对 action.set_compiled() 放行 (共享 compiled_event)."""
    thinking, env = _make_thinking()
    articulator = thinking.articulator()
    assert len(env['put']) == 1
    action = env['put'][0]

    async with action:
        waiting = asyncio.create_task(asyncio.wait_for(articulator.wait_compiled(), 2.0))
        await asyncio.sleep(0)
        assert waiting.done() is False
        action.set_compiled()
        await asyncio.wait_for(waiting, 2.0)


@pytest.mark.asyncio
async def test_articulator_action_one_to_one_stop_handshake():
    """1:1 混合: articulator.wait_action_done() 被配对 action 停止时放行 (共享 action_stop_event)."""
    thinking, env = _make_thinking()
    articulator = thinking.articulator()
    assert len(env['put']) == 1
    action = env['put'][0]

    async with action, articulator:
        waiting = asyncio.create_task(asyncio.wait_for(articulator.wait_action_done(), 2.0))
        await asyncio.sleep(0)
        assert waiting.done() is False
        await action.stop()
        await asyncio.wait_for(waiting, 2.0)


# -- 双 task 卸载 (thinking -> janus queue -> action) --------------------------

@pytest.mark.asyncio
async def test_thinking_action_offload_two_frames_compiled():
    """双 task 混合: thinking 侧生产两帧 articulator (各自 wait_compiled),
    另一侧用 janus queue 卸载 action 并逐帧 set_compiled 放行."""
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    compiled = []

    async def consume_thinking():
        for _ in range(2):
            articulator = thinking.articulator()
            await asyncio.wait_for(articulator.wait_compiled(), 2.0)
            compiled.append(articulator)

    async def offload_actions():
        for _ in range(2):
            action = await asyncio.wait_for(action_queue.async_q.get(), 2.0)
            action.set_compiled()

    await asyncio.wait_for(
        asyncio.gather(consume_thinking(), offload_actions()),
        5.0,
    )
    assert len(compiled) == 2


@pytest.mark.asyncio
async def test_articulator_exits_before_action_after_compiled():
    """双 task 时序: action 侧 set_compiled 后停留片刻再退出;
    articulator 侧只阻塞到 compiled, 应先完成 (完成顺序进同一数组)."""
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    finished: list[str] = []

    async def consume_articulator():
        articulator = thinking.articulator()
        await articulator.wait_compiled()
        finished.append('articulator')

    async def consume_action():
        action = await action_queue.async_q.get()
        action.set_compiled()
        await asyncio.sleep(0.05)
        finished.append('action')

    await asyncio.wait_for(
        asyncio.gather(consume_articulator(), consume_action()),
        5.0,
    )
    assert finished == ['articulator', 'action']


@pytest.mark.asyncio
async def test_articulator_exits_after_action_when_waiting_action_done():
    """双 task 时序: articulator 侧 wait_action_done, 只有 action 侧退出时才置位,
    故完成顺序反转 — action 先退出, articulator 后退出."""
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    finished: list[str] = []

    async def consume_articulator():
        articulator = thinking.articulator()
        async with articulator:
            await articulator.wait_action_done()
            finished.append('articulator')

    async def consume_action():
        action = await action_queue.async_q.get()
        assert len(finished) == 0
        async with action:
            finished.append('action')

    await asyncio.wait_for(
        asyncio.gather(consume_articulator(), consume_action()),
        5.0,
    )
    assert finished == ['action', 'articulator']


@pytest.mark.asyncio
async def test_articulator_and_action_full_lifecycle():
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    finished: list[str] = []

    commited = asyncio.Event()
    action_done = asyncio.Event()

    async def consume_articulator():
        articulator = thinking.articulator()
        async with articulator:
            await articulator.send("hello")
            # 没有结束.
            assert not commited.is_set()

            await articulator.send(" world")
            assert not commited.is_set()
            # commit
            await articulator.wait_compiled()
            assert commited.is_set()
            assert not action_done.is_set()
            await articulator.wait_action_done()
            assert action_done.is_set()

    logos = ''

    async def consume_action():
        nonlocal logos
        action = await action_queue.async_q.get()
        assert len(finished) == 0
        async with action:
            async for delta in action.logos():
                logos += delta
            # 先设置 committed
            commited.set()
            action.set_compiled()
            await asyncio.sleep(0.02)
            action_done.set()

    async with thinking:
        await asyncio.wait_for(
            asyncio.gather(consume_articulator(), consume_action()),
            2.0,
        )
    assert logos == 'hello world'


# -- 生命周期 ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_running_gated_by_lifecycle():
    """is_running 只在进入后、退出前成立; 退出后 is_aborted 置位."""
    thinking, _ = _make_thinking()
    assert thinking.is_running() is False

    async with thinking:
        assert thinking.is_running() is True
        assert thinking.is_aborted() is False

    assert thinking.is_running() is False
    assert thinking.is_aborted() is True


@pytest.mark.asyncio
async def test_abort_cascades_to_attention_and_records_reason():
    """abort(reason) 级联到 attention 并记录 abort_reason."""
    attention = _make_attention()
    thinking, _ = _make_thinking(attention=attention)

    thinking.abort('preempted')
    assert thinking.is_aborted() is True
    assert thinking.abort_reason() == 'preempted'
    assert attention.is_aborted() is True


@pytest.mark.asyncio
async def test_mindflow_stop_event_aborts_thinking():
    """mindflow_stop_event 置位同样终止 thinking (冗余兜底信号)."""
    thinking, env = _make_thinking()
    env['mindflow_stop_event'].set()
    assert thinking.is_aborted() is True


@pytest.mark.asyncio
async def test_wait_until_done_awaits_pending_future():
    """wait_until_done 等待挂起的 future 完成再返回."""
    thinking, _ = _make_thinking()
    fut = asyncio.Future()
    asyncio.get_running_loop().call_later(0.05, fut.set_result, 1)
    async with thinking:
        await asyncio.wait_for(thinking.wait_until_done(fut), 2.0)
    assert fut.done() is True


@pytest.mark.asyncio
async def test_wait_until_done_cancels_pending_future_on_abort():
    """abort 时 wait_until_done 取消挂起的 future, 自身不抛 CancelledError."""
    thinking, _ = _make_thinking()
    fut = asyncio.Future()
    async with thinking:
        thinking.abort('preempted')
        await asyncio.wait_for(thinking.wait_until_done(fut), 2.0)
    assert fut.cancelled() is True


@pytest.mark.asyncio
async def test_wait_abort_returns_on_stop():
    """wait_abort 阻塞直到 stop() 置位."""
    thinking, _ = _make_thinking()
    task = asyncio.create_task(asyncio.wait_for(thinking.wait_abort(), 2.0))
    await asyncio.sleep(0)
    assert task.done() is False
    await thinking.stop()
    await asyncio.wait_for(task, 2.0)


@pytest.mark.asyncio
async def test_action_thinking_aborted_with_attention():
    """thinking 退出时 action 不会一起退出 — 两者生命周期独立, 各自 stop_event 驱动."""
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    logos = ''

    finished = []
    compiled = asyncio.Event()
    got_action = None

    async def _finish(key: str):
        await asyncio.sleep(1.0)
        finished.append(key)

    async def consume_action():
        nonlocal logos, got_action
        action = await action_queue.async_q.get()
        got_action = action
        async with action:
            async for delta in action.logos():
                logos += delta
            action.set_compiled()
            compiled.set()
            await action.wait_until_done(asyncio.ensure_future(_finish('action')))

    async def run_thinking():
        async with thinking:
            async with thinking.articulator() as articulator:
                for c in 'hello world':
                    await articulator.send(c)
            assert compiled.is_set()
            await thinking.wait_until_done(asyncio.ensure_future(_finish('thinking')))

    thinking_task = asyncio.create_task(run_thinking())
    action_task = asyncio.create_task(consume_action())
    await compiled.wait()
    # 立刻终止.
    thinking.attention.abort("abort")
    await asyncio.gather(thinking_task, action_task, return_exceptions=False)
    # 两个应该都没完成 _finish
    assert finished == []

    assert thinking.abort_reason() == 'abort'
    assert got_action is not None
    assert isinstance(got_action, Action)
    assert got_action.abort_reason() == 'abort'


@pytest.mark.asyncio
async def test_thinking_waits_last_action_before_exit():
    """thinking 正常退出时等最后一个 action 退出 (wait_last_action_done 兜底).

    articulate 只 wait_compiled (不自保证 wait_action_done) 时, thinking 的
    __aexit__ 仍要阻塞到 action 退出, 否则最后一帧的 observe 会丢失, attention
    会被误判自然结束而失序. 第三方观测: 从外部事件判断 thinking 不会在 action
    退出前就退出.
    """
    action_queue: janus.Queue[Action] = janus.Queue(maxsize=10)
    thinking = BaseThinking(
        attention=_make_attention(),
        observer=BaseMomentsObserver(max_size=10),
        put_action=action_queue.sync_q.put_nowait,
        mindflow_stop_event=ThreadSafeEvent(),
    )

    action_compiled = asyncio.Event()
    release_action = asyncio.Event()
    thinking_exited = asyncio.Event()

    async def consume_action():
        action = await action_queue.async_q.get()
        async with action:
            async for _ in action.logos():
                pass
            action.set_compiled()
            action_compiled.set()
            # action 已编译但仍存活 (模拟执行中).
            await release_action.wait()

    async def run_thinking():
        async with thinking:
            # articulate 只 wait_compiled, 不自保证 wait_action_done.
            async with thinking.articulator() as articulator:
                articulator.send_nowait('logos')
                await articulator.wait_compiled()
        thinking_exited.set()

    action_task = asyncio.create_task(consume_action())
    thinking_task = asyncio.create_task(run_thinking())

    # 等 action 编译完成 (此时 action 仍存活).
    await asyncio.wait_for(action_compiled.wait(), 2.0)
    # 给 __aexit__ 的 _wait_last_action_done 一个事件循环窗口.
    await asyncio.sleep(0.05)

    # 关键断言: action 还没退出, thinking 也不该退出.
    assert not thinking_exited.is_set()

    # 放 action 退出 → thinking 才退出.
    release_action.set()
    await action_task
    await thinking_task
    assert thinking_exited.is_set()
