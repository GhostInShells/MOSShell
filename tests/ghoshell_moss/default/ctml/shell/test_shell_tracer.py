"""Smoke tests for Shell Tracer Protocol.

覆盖:
- Tracer 收到 on_task_pushed + on_task_done (task 生命周期)
- Tracer 收到 on_interpreter_stopped (append 分支旧 interpreter close)
- is_closed()=True 的 tracer 被 shell 跳过
- tracer 抛异常不影响主流程 (fire and forget)
"""
import pytest
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.concepts.shell import Tracer
from ghoshell_moss.core.concepts.command import CommandTask
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.py_channel import PyChannel


class _RecorderTracer:
    """Tracer 实现: 记录所有 fire 事件. 满足 Tracer Protocol duck-type."""

    def __init__(self, running: bool = True, closed: bool = False):
        self._running = running
        self._closed = closed
        self.pushed: list[CommandTask] = []
        self.done: list[CommandTask] = []
        self.stopped: list[Interpreter] = []

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return self._closed

    def on_task_pushed(self, task: CommandTask) -> None:
        self.pushed.append(task)

    def on_task_done(self, task: CommandTask) -> None:
        self.done.append(task)

    def on_interpreter_stopped(self, interpreter: Interpreter) -> None:
        self.stopped.append(interpreter)


@pytest.mark.asyncio
async def test_tracer_receives_task_pushed_and_done():
    shell = new_ctml_shell("tracer_test")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)
    tracer = _RecorderTracer()

    async with shell:
        shell.add_tracer(tracer)
        async with shell.interpreter_in_ctx() as i:
            i.feed("<chan:hello />")
            i.commit()
            await i.wait_tasks(timeout=2)

    assert len(tracer.pushed) == 1, f"expected 1 pushed, got {len(tracer.pushed)}"
    assert len(tracer.done) == 1, f"expected 1 done, got {len(tracer.done)}"
    assert tracer.pushed[0] is tracer.done[0], "same task should fire both hooks"
    assert tracer.pushed[0].success()


@pytest.mark.asyncio
async def test_tracer_receives_interpreter_stopped_on_append():
    """append 分支下, 旧 interpreter close 时 fire on_interpreter_stopped."""
    shell = new_ctml_shell("tracer_test_append")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def noop() -> None:
        pass

    shell.main_channel.import_channels(chan)
    tracer = _RecorderTracer()

    async with shell:
        shell.add_tracer(tracer)
        # 第一个 interpreter (clear 分支, 没有旧 interpreter, 不会 fire stopped)
        i1 = await shell.interpreter("clear")
        async with i1:
            i1.feed("<chan:noop />")
            i1.commit()
            await i1.wait_stopped()

        # 第二个 interpreter 用 append — 会先关旧的, 触发 stopped
        i2 = await shell.interpreter("append")
        async with i2:
            i2.feed("<chan:noop />")
            i2.commit()
            await i2.wait_stopped()

    # append 关闭 i1 时 fire, i2 关闭时不 fire (因为 i2 close 时是通过 async with 退出, 不走 shell.interpreter 路径)
    assert len(tracer.stopped) >= 1, f"expected >=1 stopped, got {len(tracer.stopped)}"
    # i1 是被 append 关的那个
    assert tracer.stopped[0] is i1


@pytest.mark.asyncio
async def test_closed_tracer_is_skipped():
    shell = new_ctml_shell("tracer_test_closed")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)
    closed = _RecorderTracer(closed=True)
    alive = _RecorderTracer()

    async with shell:
        shell.add_tracer(closed)
        shell.add_tracer(alive)
        async with shell.interpreter_in_ctx() as i:
            i.feed("<chan:hello />")
            i.commit()
            await i.wait_tasks(timeout=2)

    assert len(closed.pushed) == 0, "closed tracer should never receive events"
    assert len(closed.done) == 0
    assert len(alive.pushed) == 1
    assert len(alive.done) == 1


@pytest.mark.asyncio
async def test_not_running_tracer_is_skipped():
    shell = new_ctml_shell("tracer_test_paused")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)
    paused = _RecorderTracer(running=False)

    async with shell:
        shell.add_tracer(paused)
        async with shell.interpreter_in_ctx() as i:
            i.feed("<chan:hello />")
            i.commit()
            await i.wait_tasks(timeout=2)

    assert len(paused.pushed) == 0
    assert len(paused.done) == 0


@pytest.mark.asyncio
async def test_tracer_exception_does_not_break_shell():
    """一个 tracer 抛异常, 不影响其他 tracer 和主流程 (fire and forget)."""
    shell = new_ctml_shell("tracer_test_exc")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    class _ExplodingTracer:
        def is_running(self) -> bool: return True
        def is_closed(self) -> bool: return False
        def on_task_pushed(self, task): raise RuntimeError("boom pushed")
        def on_task_done(self, task): raise RuntimeError("boom done")
        def on_interpreter_stopped(self, interp): raise RuntimeError("boom stopped")

    good = _RecorderTracer()
    async with shell:
        shell.add_tracer(_ExplodingTracer())
        shell.add_tracer(good)
        async with shell.interpreter_in_ctx() as i:
            i.feed("<chan:hello />")
            i.commit()
            await i.wait_tasks(timeout=2)

    # 好 tracer 依然收到事件
    assert len(good.pushed) == 1
    assert len(good.done) == 1
