from typing import Callable, Coroutine
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.core.mindflow.base_mindflow import BaseMindflow
from ghoshell_moss.core.concepts.mindflow import Mindflow, Signal, Priority, Articulate, Action, Nucleus
import janus
import uvloop
import threading
import time
import pytest
import asyncio


def make_base_mindflow() -> BaseMindflow:
    from ghoshell_moss.contracts.logger import get_console_logger
    return BaseMindflow(logger=get_console_logger())


@pytest.mark.asyncio
async def test_full_link_signal_to_impulse():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    # 会自动注册 bus. 而且启动前不能用 add .
    mindflow.with_nucleus(nucleus)

    async with mindflow:
        await mindflow.wait_started()
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.source == "test_sensor"
                assert impulse.priority == Priority.NOTICE
                break


@pytest.mark.asyncio
async def test_suppress_and_stale_race_condition():
    """验证 suppress 和 stale 结合后的行为"""
    mindflow = make_base_mindflow()
    # 冷静期 0.1s, beat 0.05s
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event",
        # 每次 suppress 要 0.1 秒后才能继续.
        suppress_seconds=0.1,
        # 高频尝试 pulse, 实际上会阻塞到 suppress.
        pulse_beat_interval=0.01
    )

    mindflow.with_nucleus(nucleus)

    count = 0

    wait_started = asyncio.Event()

    async def _counter_task():
        nonlocal count
        async for attention in mindflow.loop():
            async with attention:
                wait_started.set()
                # 判断没有过期.
                assert not attention.peek().is_stale()
                # 模拟 Attention 耗时处理
                await asyncio.sleep(0.11)
                count += 1

    async with mindflow:
        task = asyncio.create_task(_counter_task())

        # 1. 第一个信号，正常通过
        mindflow.on_signal(Signal.new(name="vision_event", priority=Priority.NOTICE))
        # 让出等待状态.
        await wait_started.wait()
        # 2. 紧接着发第二个信号，它在 suppress 期间，且 stale 为 0.09s
        # 这个信号会成功挑战一次, 然后因为 suppress 而过期.
        challenger = Signal.new(name="vision_event", priority=Priority.NOTICE, stale_timeout=0.08)
        mindflow.on_signal(challenger)

        # 3. 等待足够久，让冷静期过期，让第二个信号 Stale
        await asyncio.sleep(0.15)
        assert challenger.__state__ == 'dispatched'

        mindflow.close()
        await task

    # 结果验证：只有第一个信号成功了，第二个被 suppress 压制并因 Stale 被丢弃
    assert count == 1


@pytest.mark.asyncio
async def test_mindflow_able_to_close():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    # 会自动注册 bus. 而且启动前不能用 add .
    mindflow.with_nucleus(nucleus)
    async with mindflow:
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.source == "test_sensor"
                assert impulse.priority == Priority.NOTICE
                # 调用之后应该不会阻塞, 都会退出.
                mindflow.close()


@pytest.mark.asyncio
async def test_mindflow_run_in_task():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    count = 0

    async def _run_in_task():
        nonlocal count
        # 会自动注册 bus. 而且启动前不能用 add .
        mindflow.with_nucleus(nucleus)
        async with mindflow:
            sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
            mindflow.on_signal(sig)
            async for attention in mindflow.loop():
                async with attention:
                    impulse = attention.peek()
                    assert impulse.source == "test_sensor"
                    assert impulse.priority == Priority.NOTICE
                    # 验证完 impulse 直接退出.
                    count += 1
                assert attention.is_aborted()
                break
        assert not mindflow.is_running()

    task = asyncio.create_task(_run_in_task())
    await task
    # 只有一个信号, 不会有第二个行为.
    assert count == 1


@pytest.mark.asyncio
async def test_mindflow_run_with_multi_signal():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event",
    )

    count = []

    one_done = asyncio.Event()

    mindflow.with_nucleus(nucleus)

    async def _run_in_task():
        # 会自动注册 bus. 而且启动前不能用 add .
        await mindflow.wait_started()
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.priority == Priority.NOTICE
                count.append(1)
            one_done.set()
            assert attention.is_aborted()

    async def _main():
        await asyncio.sleep(0.0)
        # 不等待启动, 信号会被丢弃掉.
        await mindflow.wait_started()
        assert len(count) == 0
        # 接受一个讯号, 处理完时应该都没有下一个 attention 生成出来.
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        await asyncio.sleep(0.0)
        await one_done.wait()
        # 拿到一个信号时, count 只会为1.
        assert len(count) == 1
        assert nucleus.peek() is None

        one_done.clear()
        # 尝试发送第二个信号.
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        await asyncio.sleep(0.1)
        await one_done.wait()
        # 然后就直接退出.
        mindflow.close()

    async with mindflow:
        task = asyncio.create_task(_run_in_task())
        main_task = asyncio.create_task(_main())
        # main task 会先结束.
        await main_task
        await task

    # 只有一个信号, 不会有第二个行为.
    assert len(count) == 2


def test_mindflow_in_differ_thread():
    # 验证十次没有一次出错.
    for i in range(10):
        # 多测几次, 看看会不会有意料外的时序错乱.
        _test_mindflow_in_differ_thread(i)


def _test_mindflow_in_differ_thread(i: int):
    mindflow = make_base_mindflow()
    vision_nucleus = BufferNucleus(
        name="test_sensor_vision",
        description="Sensor unit",
        target_signal="vision_event",
    )
    listen_nucleus = BufferNucleus(
        name="test_sensor_listen",
        description="Sensor unit",
        target_signal="listen_event",
        suppress_seconds=0.1,
    )
    mindflow.with_nucleus(vision_nucleus)
    mindflow.with_nucleus(listen_nucleus)
    articulate_queue = janus.Queue()
    action_queue = janus.Queue()
    articulate_loop_started = threading.Event()
    action_loop_started = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    attention_count = 0
    attention_loop_count = 0

    async def _main():
        nonlocal attention_count, second_done, attention_loop_count
        async with mindflow:
            count = 0
            async for attention in mindflow.loop():
                count += 1
                async with attention:
                    attention_count += 1
                    async for articulate, action in attention.loop():
                        attention_loop_count += 1
                        articulate_queue.sync_q.put_nowait(articulate)
                        action_queue.sync_q.put_nowait(action)
                    # 应该阻塞到 action / articulate 都执行完.
                first_done.set()
                timestamps.append(('attention_done', time.time()))
                if count == 2:
                    # 第二个 attention 完成时退出.
                    break
            second_done.set()
        articulate_queue.shutdown()
        action_queue.shutdown()

    content = "hello world"

    async def _articulate_loop():
        await mindflow.wait_started()
        while mindflow.is_running():
            articulate_loop_started.set()
            try:
                articulate = await articulate_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                break
            timestamps.append(('articulate_start', time.time()))
            async with articulate:
                for c in content:
                    await articulate.send(c)
            timestamps.append(('articulate_done', time.time()))

    got = []
    timestamps = []

    async def _actions():
        await mindflow.wait_started()
        while mindflow.is_running():
            action_loop_started.set()
            try:
                action = await action_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                break
            timestamps.append(('action_start', time.time()))
            async with action:
                received = ''
                async for delta in action.logos():
                    received += delta
                # 取保执行完的会放入.
                got.append(received)
            # 调试用的时间戳.
            timestamps.append(("action_done", time.time()))

    def _run_main():
        asyncio.set_event_loop(uvloop.new_event_loop())
        asyncio.run(_main())

    def _run_articulate():
        asyncio.set_event_loop(uvloop.new_event_loop())
        asyncio.run(_articulate_loop())

    def _run_actions():
        asyncio.set_event_loop(uvloop.new_event_loop())
        asyncio.run(_actions())

    t_main = threading.Thread(target=_run_main)
    t_articulate = threading.Thread(target=_run_articulate)
    t_actions = threading.Thread(target=_run_actions)
    t_main.start()
    t_articulate.start()
    t_actions.start()
    # 等待启动完了再推入信号.
    assert mindflow.wait_started_sync(2)
    assert articulate_loop_started.wait(2)
    assert action_loop_started.wait(2)
    # 第一个信号输出成功.
    signal_1 = Signal.new(name="vision_event", priority=Priority.NOTICE, strength=100)
    signal_2 = Signal.new(name="listen_event", priority=Priority.NOTICE, strength=90)
    mindflow.on_signal(signal_1)
    # 第二个信号应该被抑制.
    mindflow.on_signal(signal_2)
    assert signal_1.__state__ == 'pending'
    assert signal_2.__state__ == 'pending'
    # 等待到第二个运行结束. 预计还得快.
    try:
        # 仅仅用来对齐线程时序. 不用卡那么死.
        assert first_done.wait(10)
        # 用于对齐时序.
        done = second_done.wait(10)
        assert attention_count == 2
        assert attention_loop_count == 2
        assert done, got
        assert len(got) == 2
        mindflow.close()
        t_main.join()
        t_articulate.join()
        t_actions.join()
        assert signal_1.__state__ == 'dispatched'
        assert signal_2.__state__ == 'dispatched'
    finally:
        mindflow.close()
        # debug 才用.
        # print('++++', i, signal_1.__state__, signal_2.__state__)
        # print('++++', i, timestamps)


class MindflowSuite:
    """想做更多的测试, 简单做一个套件. """

    def __init__(
            self,
            mindflow: Mindflow | None = None,
            *nuclei: Nucleus,
    ) -> None:
        self.mindflow = mindflow or make_base_mindflow()
        self.articulate_queue: janus.Queue[Articulate | None] = janus.Queue()
        self.action_queue: janus.Queue[Action | None] = janus.Queue()
        self._all_started = threading.Barrier(3)
        self._is_started = threading.Event()
        for n in nuclei:
            self.mindflow.with_nucleus(n)
        self._main_t: threading.Thread | None = None
        self._articulate_t: threading.Thread | None = None
        self._action_t: threading.Thread | None = None

    def _run(
            self,
            articulate_func: Callable[[Articulate], Coroutine[None, None, None]],
            action_func: Callable[[Action], Coroutine[None, None, None]]
    ) -> None:

        def _run_articulate_loop():
            nonlocal articulate_func
            asyncio.set_event_loop(uvloop.new_event_loop())
            asyncio.run(self._articulate_loop(articulate_func))

        def _run_action_loop():
            nonlocal action_func
            asyncio.set_event_loop(uvloop.new_event_loop())
            asyncio.run(self._action_loop(action_func))

        def _main():
            asyncio.set_event_loop(uvloop.new_event_loop())
            asyncio.run(self._main_loop())

        self._main_t = threading.Thread(target=_main)
        self._articulate_t = threading.Thread(target=_run_articulate_loop)
        self._action_t = threading.Thread(target=_run_action_loop)

        self._main_t.start()
        self._articulate_t.start()
        self._action_t.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def new_nucleus(name: str) -> Nucleus:
        return BufferNucleus(
            name=name,
            description=name,
            target_signal=name,
        )

    def run_in_thread(
            self,
            articulate_func: Callable[[Articulate], Coroutine[None, None, None]],
            action_func: Callable[[Action], Coroutine[None, None, None]]
    ):
        self._run(articulate_func, action_func)
        assert self._is_started.wait(3)

    def _join(self) -> None:
        if self._main_t is not None:
            self._main_t.join()
            self._main_t = None
        if self._articulate_t is not None:
            self._articulate_t.join()
            self._articulate_t = None
        if self._action_t is not None:
            self._action_t.join()
            self._action_t = None

    def close(self) -> None:
        self.mindflow.close()
        self._join()

    async def _articulate_loop(self, articulate_func: Callable[[Articulate], Coroutine[None, None, None]]) -> None:
        self._all_started.wait()
        try:
            await self.mindflow.wait_started()
            while self.mindflow.is_running():
                item = await self.articulate_queue.async_q.get()
                if item is None:
                    break
                async with item:
                    await item.create_task(articulate_func(item))
        except janus.AsyncQueueShutDown:
            pass

    async def _action_loop(self, action_func: Callable[[Action], Coroutine[None, None, None]]) -> None:
        self._all_started.wait()
        try:
            await self.mindflow.wait_started()
            while self.mindflow.is_running():
                item = await self.action_queue.async_q.get()
                if item is None:
                    break
                async with item:
                    await item.create_task(action_func(item))
        except janus.AsyncQueueShutDown:
            pass

    async def _main_loop(self):
        self._all_started.wait()
        async with self.mindflow:
            self._is_started.set()
            async for attention in self.mindflow.loop():
                async with attention:
                    # 会阻塞在这里.
                    async for articulate, action in attention.loop():
                        self.articulate_queue.sync_q.put_nowait(articulate)
                        self.action_queue.sync_q.put_nowait(action)
        self.articulate_queue.shutdown(immediate=True)
        self.action_queue.shutdown(immediate=True)


def test_suite_baseline():
    suite = MindflowSuite()
    nucleus = suite.new_nucleus("test")
    suite.mindflow.with_nucleus(nucleus)
    content = 'hello world'
    got = []
    done_event = threading.Event()

    async def _articulate_func(articulate: Articulate) -> None:
        for char in content:
            await articulate.send(char)

    async def _action_func(action: Action) -> None:
        received = ''
        async for delta in action.logos():
            received += delta
        got.append(received)
        done_event.set()

    with suite:
        suite.run_in_thread(_articulate_func, _action_func)
        suite.mindflow.on_signal(Signal.new('test'))
        assert done_event.wait(2)


def test_suite_consuming_alot_of_signals():
    suite = MindflowSuite()
    nucleus = suite.new_nucleus("test")
    suite.mindflow.with_nucleus(nucleus)
    content = 'hello world'
    got = []
    _done_event = threading.Event()

    async def _articulate_func(articulate: Articulate) -> None:
        for char in content:
            await articulate.send(char)

    async def _action_func(action: Action) -> None:
        received = ''
        async for delta in action.logos():
            received += delta
        _done_event.set()
        got.append(received)

    with suite:
        # 测试连续处理十个.
        suite.run_in_thread(_articulate_func, _action_func)
        for i in range(10):
            suite.mindflow.on_signal(Signal.new('test'))
            _done_event.wait()
            _done_event.clear()
            if len(got) == 10:
                break
            time.sleep(0.1)
    for line in got:
        assert line == content


def test_suite_consuming_endless_observe():
    suite = MindflowSuite()
    nucleus = suite.new_nucleus("test")
    suite.mindflow.with_nucleus(nucleus)
    content = 'hello world'
    got = []
    done_event = threading.Event()

    async def _articulate_func(articulate: Articulate) -> None:
        for char in content:
            await articulate.send(char)

    async def _action_func(action: Action) -> None:
        received = ''
        async for delta in action.logos():
            received += delta
        got.append(received)
        if len(got) < 10:
            action.outcome('hello', observe=True)
            return
        done_event.set()

    with suite:
        # 测试连续处理十个.
        suite.run_in_thread(_articulate_func, _action_func)
        # 只发送一个信号.
        suite.mindflow.on_signal(Signal.new('test'))
        done_event.wait()
        assert len(got) == 10


def test_wait_first_impulse_complete():
    suite = MindflowSuite()
    nucleus = suite.new_nucleus("test")
    suite.mindflow.with_nucleus(nucleus)

    content = 'hello world'
    got = []
    done_event = threading.Event()

    async def _articulate_func(articulate: Articulate) -> None:
        for char in content:
            await articulate.send(char)

    async def _action_func(action: Action) -> None:
        received = ''
        async for delta in action.logos():
            received += delta
        got.append(received)
        done_event.set()

    suite.run_in_thread(_articulate_func, _action_func)
    incomplete = Signal.new("test", complete=False, stale_timeout=0.1)
    suite.mindflow.on_signal(incomplete)
    assert incomplete.__state__ == "pending"
    # 0.1 秒后还在阻塞.
    time.sleep(0.05)
    assert not done_event.is_set()
    attention = suite.mindflow.attention()
    assert attention is not None
    time.sleep(0.02)
    assert not done_event.is_set()
    # 投入一个 complete.
    complete = Signal.new("test", complete=True)
    complete.id = incomplete.id
    # 手动塞入 signal.
    nucleus.on_signal(complete)
    suite.mindflow.on_signal(complete)
    assert done_event.wait(1)
    assert len(got) == 1
    suite.close()
