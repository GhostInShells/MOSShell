"""Smoke tests for InterleavedThinkingToolset.

覆盖:
- buffered / drain 语义 (drain 后清空, buffered 保留)
- status 反映当下 shell.interpreting() 活指针
- wait_interpreter_done 的时间语义 (阻塞 → interpreter stop → 唤醒)
- K9 兜底作用域收窄: 只对 observe=True 的空 outcome 做占位, 其他空成功折叠成计数
- InterpreterStopped 只在有 parsing_exception 时进 buffer
- close 唤醒所有 pending waiter
- async with 生命周期
- project_events 投影: 分桶聚合 + 时间戳
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
    project_events,
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
async def test_task_done_as_message_k9_scope_narrowed():
    """K9 作用域收窄: observe=True 空 outcome 才有占位, observe=False 空成功返 []."""
    shell = new_ctml_shell("its_k9")
    chan = PyChannel(name="chan")

    @chan.build.command(always_observe=True)
    async def watched() -> None:
        return None  # observe=True 空 outcome — 存在性必须占位

    @chan.build.command()
    async def silent() -> None:
        return None  # observe=False 空成功 — as_message 返 [], 交聚合层计数

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:watched />")
                i.feed("<chan:silent />")
                i.commit()
                await i.wait_tasks(timeout=2)
            events = toolset.drain()
            assert len(events) == 2
            done_by_caller = {e.result.caller: e for e in events if isinstance(e, TaskDone)}

            # observe=True: 占位不蒸发
            watched_msgs = done_by_caller["chan:watched"].as_message()
            assert len(watched_msgs) == 1, "observe=True 空 outcome 必须占位"
            assert "(no output)" in watched_msgs[0].to_content_string()

            # observe=False: as_message 返空, 交由 project_events 聚合
            silent_msgs = done_by_caller["chan:silent"].as_message()
            assert silent_msgs == [], "observe=False 空成功不占位, 走计数聚合"


@pytest.mark.asyncio
async def test_project_events_folds_silent_success_into_tally():
    """K8 计数聚合: 多个 observe=False 空成功 → 一条 <shell_tally>success: N/>."""
    shell = new_ctml_shell("its_tally")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def noop() -> None:
        return None

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                for _ in range(3):
                    i.feed("<chan:noop />")
                i.commit()
                await i.wait_tasks(timeout=2)
            events = toolset.drain()
            status = toolset.status()
            assert len(events) == 3

            messages = project_events(events, status)
            # 3 个空成功 → 1 条 tally + 1 条 status
            tags = [m.meta.tag for m in messages]
            assert 'shell_tally' in tags, f"expected shell_tally, got tags: {tags}"
            assert 'shell_status' in tags
            # tally 里必须只有 success: 3, 不能有 caller name (K8 折叠纪律)
            tally = next(m for m in messages if m.meta.tag == 'shell_tally')
            content = tally.to_content_string()
            assert "success: 3" in content
            assert "chan:noop" not in content, "tally 不带 caller name (K8 token 纪律)"
            # 时间戳 attribute
            assert tally.meta.attributes and 'at' in tally.meta.attributes


@pytest.mark.asyncio
async def test_project_events_mixed_payload_and_tally():
    """有 payload 的 TaskDone 保留身份, 空 payload 折叠成 tally, 两者共存."""
    shell = new_ctml_shell("its_mixed")
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def loud() -> str:
        return "hello"  # 有 payload

    @chan.build.command()
    async def quiet() -> None:
        return None  # observe=False 空

    shell.main_channel.import_channels(chan)

    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            async with shell.interpreter_in_ctx() as i:
                i.feed("<chan:loud />")
                i.feed("<chan:quiet />")
                i.feed("<chan:quiet />")
                i.commit()
                await i.wait_tasks(timeout=2)
            events = toolset.drain()
            status = toolset.status()

            messages = project_events(events, status)
            tags = [m.meta.tag for m in messages]
            # 有 payload 的 loud → 保留 (走 CommandTaskResult.as_messages, 包在 <command> 里)
            assert 'command' in tags, f"loud payload must survive; got tags: {tags}"
            # 两个 quiet → 一条 tally success: 2
            assert 'shell_tally' in tags
            tally = next(m for m in messages if m.meta.tag == 'shell_tally')
            assert "success: 2" in tally.to_content_string()
            assert 'shell_status' in tags


@pytest.mark.asyncio
async def test_status_message_has_timestamp():
    """shell_status 消息必须带简化时间戳 at=HH:MM:SS."""
    shell = new_ctml_shell("its_ts")
    async with shell:
        async with InterleavedThinkingToolset.new_from_shell(shell) as toolset:
            status = toolset.status()
            msgs = status.as_message()
            assert len(msgs) == 1
            attrs = msgs[0].meta.attributes
            assert attrs is not None and 'at' in attrs
            at = attrs['at']
            # HH:MM:SS 格式, 8 字符
            assert len(at) == 8 and at.count(':') == 2, f"expected HH:MM:SS, got {at!r}"


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
