"""Smoke tests for InterleavedThinkingToolset.

覆盖:
- buffered / drain 语义 (drain 后清空, buffered 保留)
- status 反映当下 shell.interpreting() 活指针
- wait_interpreter_done 的时间语义 (阻塞 → interpreter stop → 唤醒)
- K9 兜底: 空 outcome 在 TaskDone.as_message 里被合法非空包裹
- InterpreterStopped 只在有 parsing_exception 时进 buffer
- close 唤醒所有 pending waiter
- async with 生命周期
"""
import asyncio
import pytest

from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.host.interleaved_thinking import (
    InterleavedThinkingToolset,
    TaskDone,
    InterpreterStopped,
    InterpreterStatus,
)


@pytest.mark.asyncio
async def test_drain_after_task_done():
    shell = new_ctml_shell("its_test")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            assert toolset.drain() == []
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)
            events = toolset.drain()
            assert len(events) == 1
            assert isinstance(events[0], TaskDone)
            assert events[0].result.result == "world"
            # drain 后清空
            assert toolset.drain() == []


@pytest.mark.asyncio
async def test_buffered_preserves_state():
    """buffered 只读, 不影响 drain."""
    shell = new_ctml_shell("its_buffered")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            b1 = toolset.buffered()
            b2 = toolset.buffered()
            assert len(b1) == 1
            assert len(b2) == 1
            # drain 后 buffered 也空
            toolset.drain()
            assert toolset.buffered() == []


@pytest.mark.asyncio
async def test_task_done_as_message_k9_empty_outcome():
    """K9: 空 result + 空 messages 不该在投影里蒸发存在性."""
    shell = new_ctml_shell("its_k9")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def silent() -> None:
        return None  # 空 outcome — 现存代码会把它 as_messages() 变成 []

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:silent />")
                i.commit()
                await i.wait_tasks(timeout=2)
            events = toolset.drain()
            assert len(events) == 1
            done = events[0]
            assert isinstance(done, TaskDone)
            msgs = done.as_message()
            assert len(msgs) == 1, "空 outcome 也必须有 event, 存在性不蒸发"
            content = msgs[0].to_content_string()
            assert "(no output)" in content


@pytest.mark.asyncio
async def test_status_reflects_shell_interpreting():
    """status 从 shell.interpreting() 读活指针."""
    shell = new_ctml_shell("its_status")
    chan = PyChannel(name="chan")

    started = asyncio.Event()
    release = asyncio.Event()

    @chan.build.command()
    async def blocker() -> str:
        started.set()
        await release.wait()
        return "released"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            # 无 interpreter 时
            s0 = toolset.status()
            assert isinstance(s0, InterpreterStatus)
            assert s0.running is False

            # 起一个 interpreter, 让它 hold 一个 blocker
            i = await shell.interpreter("clear")
            async with i:
                i.feed("<chan:blocker />")
                i.commit()
                await i.wait_compiled()
                await started.wait()  # 等 blocker 真正进入执行

                s1 = toolset.status()
                assert s1.running is True
                assert "chan:blocker" in s1.ongoing_callers

                # 放行
                release.set()
                await i.wait_stopped()

            # interpreter 关闭后, running=False
            s2 = toolset.status()
            assert s2.running is False


@pytest.mark.asyncio
async def test_wait_interpreter_done_returns_immediately_when_idle():
    """无 running interpreter 时, wait_interpreter_done 立即返回."""
    shell = new_ctml_shell("its_wait_idle")

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            # 无 interpreter, 应立即返回
            status = await asyncio.wait_for(toolset.wait_interpreter_done(), timeout=1.0)
            assert status.running is False


@pytest.mark.asyncio
async def test_wait_interpreter_done_wakes_on_stop():
    """有 running interpreter 时, wait 阻塞直到 stop."""
    shell = new_ctml_shell("its_wait_stop")
    chan = PyChannel(name="chan")

    started = asyncio.Event()
    release = asyncio.Event()

    @chan.build.command()
    async def blocker() -> str:
        started.set()
        await release.wait()
        return "released"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            i = await shell.interpreter("clear")
            async with i:
                i.feed("<chan:blocker />")
                i.commit()
                await i.wait_compiled()
                await started.wait()

                # 起 wait 协程, 应该阻塞
                wait_task = asyncio.create_task(toolset.wait_interpreter_done())
                await asyncio.sleep(0.1)
                assert not wait_task.done(), "wait should block while interpreter running"

                # 放行 + interpreter 自然 stop
                release.set()
                await i.wait_stopped()

            # interpreter 已 close (async with exit + shell.__aexit__)
            # wait_task 应被 on_interpreter_stopped 唤醒
            status = await asyncio.wait_for(wait_task, timeout=2.0)
            assert isinstance(status, InterpreterStatus)


@pytest.mark.asyncio
async def test_multiple_waiters_all_wake():
    """多个并发 wait, 一次 on_interpreter_stopped 全部唤醒."""
    shell = new_ctml_shell("its_multi_wait")
    chan = PyChannel(name="chan")

    started = asyncio.Event()
    release = asyncio.Event()

    @chan.build.command()
    async def blocker() -> str:
        started.set()
        await release.wait()
        return "released"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            i = await shell.interpreter("clear")
            async with i:
                i.feed("<chan:blocker />")
                i.commit()
                await i.wait_compiled()
                await started.wait()

                waits = [asyncio.create_task(toolset.wait_interpreter_done()) for _ in range(3)]
                await asyncio.sleep(0.05)
                for w in waits:
                    assert not w.done()

                release.set()
                await i.wait_stopped()

            results = await asyncio.wait_for(asyncio.gather(*waits), timeout=2.0)
            assert len(results) == 3
            for r in results:
                assert isinstance(r, InterpreterStatus)


@pytest.mark.asyncio
async def test_close_wakes_pending_waiters():
    """toolset.close 应立即唤醒所有 pending waiter."""
    shell = new_ctml_shell("its_close_wake")
    chan = PyChannel(name="chan")

    started = asyncio.Event()
    release = asyncio.Event()

    @chan.build.command()
    async def blocker() -> str:
        started.set()
        await release.wait()
        return "done"

    shell.main_channel.import_channels(chan)

    async with shell:
        toolset = InterleavedThinkingToolset.new_from_shell(shell)
        i = await shell.interpreter("clear")
        async with i:
            i.feed("<chan:blocker />")
            i.commit()
            await i.wait_compiled()
            await started.wait()

            wait_task = asyncio.create_task(toolset.wait_interpreter_done())
            await asyncio.sleep(0.05)
            assert not wait_task.done()

            # close 应唤醒 waiter (即使 interpreter 还没 stop)
            await toolset.close()
            status = await asyncio.wait_for(wait_task, timeout=1.0)
            assert isinstance(status, InterpreterStatus)

            # cleanup
            release.set()
            await i.wait_stopped()


@pytest.mark.asyncio
async def test_closed_toolset_stops_receiving_events():
    """close 后 shell 的后续 fire 都被跳过 (is_closed=True)."""
    shell = new_ctml_shell("its_closed_skip")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        toolset = InterleavedThinkingToolset.new_from_shell(shell)
        await toolset.close()
        assert toolset.is_closed()

        async with shell.interpreter_in_ctx() as i:
            i.feed("<chan:hello />")
            i.commit()
            await i.wait_tasks(timeout=2)

        # closed toolset 不应有新事件
        assert toolset.drain() == []


@pytest.mark.asyncio
async def test_interpreter_stopped_only_buffers_on_exception():
    """InterpreterStopped event 只在有 parsing_exception 时进 buffer."""
    shell = new_ctml_shell("its_stop_no_exc")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def hello() -> str:
        return "world"

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:hello />")
                i.commit()
                await i.wait_tasks(timeout=2)

            events = toolset.drain()
            # 只有 TaskDone, 没有 InterpreterStopped (清洁停止)
            interp_stops = [e for e in events if isinstance(e, InterpreterStopped)]
            assert interp_stops == [], "clean stop should not create InterpreterStopped event"


@pytest.mark.asyncio
async def test_interpreter_stopped_buffered_on_parse_error():
    """编译期错误应产生 InterpreterStopped 事件."""
    shell = new_ctml_shell("its_stop_with_exc")

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            # 触发编译期错误: 一个不存在的命令
            i = await shell.interpreter("clear")
            try:
                async with i:
                    i.feed("<totally.fake:command/>")
                    i.commit()
                    try:
                        await i.wait_compiled()
                    except Exception:
                        pass
            except Exception:
                pass

            events = toolset.drain()
            stops = [e for e in events if isinstance(e, InterpreterStopped)]
            assert len(stops) >= 1, f"parse error should yield InterpreterStopped, got events: {events}"
            assert stops[0].exception  # non-empty
            msg = stops[0].as_message()[0].to_content_string()
            assert stops[0].exception in msg or len(msg) > 0
