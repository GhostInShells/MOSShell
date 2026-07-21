"""shell.interpreter(kind='append') 跨帧延续协议.

测试风格: 协议风格 — CTML 是唯一的输入标尺, side effects + task 真实运行态是
唯一的可观察输出. 不窥视 ``interpreter._managing_tasks`` 之类内部细节.

关键认知 (协议事实):
    Command Task 在 MOSS 里是**编译态 (AST 节点)**, 不是 asyncio task. Shell 通过
    interpreter 生成的是流式 AST, 可以分布式卸载到别的进程, Shell 维护拓扑一致.
    一个编译态 task 完全可能直接被取消, 从未进入过运行态.

    所以"跨帧延续"测试必须用 ``asyncio.Event`` 钉死时序:
        1. command 函数内 ``started.set()`` 表明"我开始跑了" (运行态).
        2. command 函数 ``await release.wait()`` 模拟长任务挂起.
        3. 测试代码 ``await started.wait()`` 后才允许 interpreter 退出.
        4. 断言运行结束, 用 release.set() 控制. 不靠 ``asyncio.sleep`` 推测时序.

通道 FIFO 戒律:
    长 task occupy 通道时, 同通道排队任务会被阻塞. 测试若让第二帧任务跟长 task
    同通道, 第二帧 ``wait_stopped`` 会卡死. 故第二帧的对照任务一律放**异通道**.

所有 await 加 ``asyncio.wait_for(timeout=...)`` 容错: 协议失守时 fail-fast,
不让测试变成挂起.

两层契约对齐:

    1. **kind='append' 不调 shell.clear()** (ctml_shell.py:342-349):
       旧 interpreter 用 ``close(cancel_executing=False)`` 关掉, 已进入运行态的
       task 留在 channel runtime 跑.

    2. **clear_after_exit 控制 interpreter __aexit__ 时是否 cancel 跑着的 task**:
       - ``clear_after_exit=False`` (default): 退出 with-block 旧 task 继续跑.
       - ``clear_after_exit=True``: 退出 with-block 旧 task 被 fail (INTERRUPTED).

       mindflow 体系总是用 ``clear_after_exit=False`` 配 ``kind='append'``
       (见 ghost_runtime.py:409).
"""
import asyncio
from typing import AsyncIterable

import pytest

from ghoshell_moss.core import new_ctml_shell
from ghoshell_moss.core.blueprint.channel_builder import new_channel


@pytest.mark.asyncio
async def test_append_cross_frame_long_task_survives_interpreter_exit():
    """append + clear_after_exit=False (默认): 上一帧运行态 task 跨帧仍在跑.

    协议契约:
        - 帧 1 CTML: ``<slow:long_task/>``. long_task 内置 started + release 双 event,
          测试代码 await ``started`` 才退出 with-block, 钉死"运行态已建立".
        - clear_after_exit 默认 False → __aexit__ 走 ``cancel_executing=False``,
          运行态 task 不被 cancel.
        - 帧 2 跑 ``<fast:short_task/>`` (异通道, 立刻完成).
        - 释放 release event, long_task 自然跑完.
    """
    slow = new_channel(name='slow')
    fast = new_channel(name='fast')
    log: list[str] = []
    long_started = asyncio.Event()
    long_release = asyncio.Event()

    @slow.build.command()
    async def long_task() -> None:
        long_started.set()
        await long_release.wait()
        log.append('long_done')

    @fast.build.command()
    async def short_task() -> None:
        log.append('short_done')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(slow, fast)
    async with shell:
        # 帧 1.
        async with await shell.interpreter(kind='append') as interp1:
            interp1.feed("<slow:long_task/>")
            interp1.commit()
            await asyncio.wait_for(interp1.wait_compiled(), timeout=1.0)
            await asyncio.wait_for(long_started.wait(), timeout=1.0)
        # 帧 1 退出: long_task 仍 await release.

        # 帧 2: 异通道 short_task.
        async with await shell.interpreter(kind='append') as interp2:
            interp2.feed("<fast:short_task/>")
            interp2.commit()
            await asyncio.wait_for(interp2.wait_stopped(), timeout=1.0)

        assert 'short_done' in log
        assert 'long_done' not in log, (
            "long_task 在 release 前不该完成 — 这是跨帧延续协议的前提"
        )

        long_release.set()
        await asyncio.wait_for(shell.wait_until_idle(), timeout=2.0)

    assert 'long_done' in log
    assert log.index('short_done') < log.index('long_done')


@pytest.mark.asyncio
async def test_append_clear_after_exit_true_cancels_running_task():
    """append + clear_after_exit=True: interpreter 退出时运行态 task 被打断.

    协议契约 (CTMLInterpreter.close):
        close cancel 的触发条件是 ``cancel_executing or self._clear_after_exit``.
        ``clear_after_exit=True`` 让 __aexit__ 走 cancel 路径,
        managing_tasks 里运行态 task 被 fail (INTERRUPTED), 协程收到 CancelledError.
    """
    slow = new_channel(name='slow')
    log: list[str] = []
    long_started = asyncio.Event()
    long_release = asyncio.Event()

    @slow.build.command()
    async def long_task() -> None:
        long_started.set()
        try:
            await long_release.wait()
            log.append('long_done')
        except asyncio.CancelledError:
            log.append('long_cancelled')
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(slow)
    async with shell:
        async with await shell.interpreter(kind='append', clear_after_exit=True) as interp:
            interp.feed("<slow:long_task/>")
            interp.commit()
            await asyncio.wait_for(interp.wait_compiled(), timeout=1.0)
            await asyncio.wait_for(long_started.wait(), timeout=1.0)
        # 退出: clear_after_exit=True → close 走 cancel 分支, long_task 被取消.

        # 让 cancel 传播到协程.
        await asyncio.sleep(0.05)

    assert 'long_done' not in log
    assert 'long_cancelled' in log


@pytest.mark.asyncio
async def test_append_does_not_clear_shell_state():
    """append 模式创建动作本身不调 shell.clear() — 异通道运行态 task 不被清.

    协议契约 (ctml_shell.py:342-349 vs 333-338):
        kind='clear' 先 ``await self.clear()`` 再建 interpreter;
        kind='append' 直接关闭旧 interpreter, 不动 shell 整体状态.

    与 cross_frame_long_task 的区别:
        那条聚焦 ``clear_after_exit=False`` 让 interpreter __aexit__ 不杀 task;
        这条聚焦 ``shell.interpreter(kind='append')`` **创建动作本身**不杀任何东西
        (区别于 kind='clear' 的协议级清扫).

    用异通道 fast_cmd 验证, 避免同通道 FIFO deadlock.
    """
    slow = new_channel(name='slow')
    fast = new_channel(name='fast')
    log: list[str] = []
    slow_started = asyncio.Event()
    slow_release = asyncio.Event()

    @slow.build.command()
    async def slow_cmd() -> None:
        slow_started.set()
        await slow_release.wait()
        log.append('slow_done')

    @fast.build.command()
    async def fast_cmd() -> None:
        log.append('fast_done')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(slow, fast)
    async with shell:
        async with await shell.interpreter(kind='append') as interp1:
            interp1.feed("<slow:slow_cmd/>")
            interp1.commit()
            await asyncio.wait_for(interp1.wait_compiled(), timeout=1.0)
            await asyncio.wait_for(slow_started.wait(), timeout=1.0)
        # 帧 1 退出, slow_cmd 在运行态等 release.

        # 帧 2: 新 append interpreter 创建动作本身不应触发 shell.clear() —
        # 不然 slow_cmd 会被取消, slow_release.set() 后也跑不完.
        async with await shell.interpreter(kind='append') as interp2:
            interp2.feed("<fast:fast_cmd/>")
            interp2.commit()
            await asyncio.wait_for(interp2.wait_stopped(), timeout=1.0)

        assert 'fast_done' in log
        assert 'slow_done' not in log

        slow_release.set()
        await asyncio.wait_for(shell.wait_until_idle(), timeout=2.0)

    # 异通道并行: fast 不被 slow 阻塞, 时序上 fast 先 done.
    assert log.index('fast_done') < log.index('slow_done')


@pytest.mark.asyncio
async def test_clear_kind_cancels_running_task_from_prior_frame():
    """对照: kind='clear' 在创建时 ``await self.clear()``, 取消运行态 task.

    这是 ``append`` 的对偶协议. 帧 2 用 kind='clear' 时, 帧 1 还在 slow channel
    占据运行态的 slow_cmd 应被 clear 路径取消.

    fast_cmd 放异通道, 避免同通道 FIFO 干扰判断.
    """
    slow = new_channel(name='slow')
    fast = new_channel(name='fast')
    log: list[str] = []
    slow_started = asyncio.Event()
    slow_release = asyncio.Event()

    @slow.build.command()
    async def slow_cmd() -> None:
        slow_started.set()
        try:
            await slow_release.wait()
            log.append('slow_done')
        except asyncio.CancelledError:
            log.append('slow_cancelled')
            raise

    @fast.build.command()
    async def fast_cmd() -> None:
        log.append('fast_done')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(slow, fast)
    async with shell:
        async with await shell.interpreter(kind='append') as interp1:
            interp1.feed("<slow:slow_cmd/>")
            interp1.commit()
            await asyncio.wait_for(interp1.wait_compiled(), timeout=1.0)
            await asyncio.wait_for(slow_started.wait(), timeout=1.0)
        # 帧 1 退出, slow_cmd 在运行态等 release.
        assert 'slow_done' not in log

        # 帧 2 用 kind='clear': 创建时 self.clear() → 取消 slow_cmd.
        async with await shell.interpreter(kind='clear') as interp2:
            interp2.feed("<fast:fast_cmd/>")
            interp2.commit()
            await asyncio.wait_for(interp2.wait_stopped(), timeout=1.0)

        await asyncio.wait_for(shell.wait_until_idle(), timeout=1.0)

    assert 'slow_cancelled' in log, "kind='clear' must cancel running task from prior frame"
    assert 'slow_done' not in log
    assert 'fast_done' in log


@pytest.mark.asyncio
async def test_append_chunks_command_continues_across_frames():
    """跨帧延续在 chunks__ (流式) 命令上的协议表现.

    流式命令是 mindflow 三循环里典型的 "长运行态 task" — speech 输出、视觉持续输出.
    它们必须能跨帧存活, 否则模型每次进新 articulator 帧时, 上一帧还没说完的话
    都会被截断.

    通道选择 (避开 CTML 父子分发规则):
        CTML 协议: 父通道 occupy 时, 所有子通道的新命令 pending.
        若 speak 放在 main channel (走主 ``__content__``), main occupy 会让 other
        子通道命令 pending → 帧 2 deadlock.
        故 speak 放子通道 a (occupy a 而非 main), other 与 a 是兄弟,
        a occupy 不影响 other → 帧 2 自由执行.

    场景:
        - 帧 1: ``<a:_>hello world</a:_>`` — a 通道作用域内非命令文本走 a 的 ``__content__`` = speak.
          speak 在收到首 chunk 后挂起等 release_first, 钉死 "运行态已建立".
        - 帧 2: ``<other:done/>`` — 与 a 是兄弟通道, 不受 a occupy 影响, 立刻完成.
        - 释放 release_first, speak 继续消费剩余 chunks.
        - 等 shell idle, 验证整段 text 全部消费完毕.
    """
    a = new_channel(name='a')
    other = new_channel(name='other')
    chunks_received: list[str] = []
    first_chunk_seen = asyncio.Event()
    release_first = asyncio.Event()
    other_done = asyncio.Event()

    async def speak(chunks__: AsyncIterable[str]) -> None:
        first = True
        async for chunk in chunks__:
            chunks_received.append(chunk)
            if first:
                first_chunk_seen.set()
                await release_first.wait()
                first = False

    a.build.content_command(speak)

    @other.build.command()
    async def done() -> None:
        other_done.set()

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a, other)
    async with shell:
        async with await shell.interpreter(kind='append') as interp1:
            interp1.feed("<a:_>hello world</a:_>")
            interp1.commit()
            await asyncio.wait_for(interp1.wait_compiled(), timeout=1.0)
            await asyncio.wait_for(first_chunk_seen.wait(), timeout=1.0)
        # 帧 1 退出: speak 仍 await release_first, occupy a channel.

        # 帧 2: other (a 的兄弟通道), 不受 a occupy 影响.
        async with await shell.interpreter(kind='append') as interp2:
            interp2.feed("<other:done/>")
            interp2.commit()
            await asyncio.wait_for(interp2.wait_stopped(), timeout=1.0)
        assert other_done.is_set()

        # 释放 chunks 消费, 等所有事跑完.
        release_first.set()
        await asyncio.wait_for(shell.wait_until_idle(), timeout=2.0)

    full_text = ''.join(chunks_received)
    assert full_text == 'hello world', (
        f"chunks 跨帧消费完整性失败: 期望 'hello world', 实得 {full_text!r}"
    )