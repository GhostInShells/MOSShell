"""Mindflow + Shell 集成 — 协议级三循环拓扑测试.

定位:
    ``MindflowInShell`` (``core.mindflow.mindflow_in_shell``) 是 mindflow 三循环
    的标准装线逻辑, 测试目标就是它本身. 本文件用 ``MindflowInShellTestSuite`` —
    它的测试专用具体子类 — 复用真实三循环, 只注入 logos 来源 / signal 与观测,
    验证抽象层的协议契约, 不依赖 ghost / matrix / session.

    参照关系 (重构方向): 这套测试是规范 — atom / dolores 等 ghost 原型在跑通后
    才知道如何装线; 测试不反向复刻 ghost 的实现.

设计约束 (来自 ``core.mindflow.mindflow_in_shell``):
    - ``_thinking_loop``: 从 ``mindflow.thinking_loop()`` 取 ``Thinking``, 先预发
      ``moment.command_logos``, 再走 ``_articulate_from_thinking`` (可拆卸 logos
      来源). ``thinking.effort() == 'none'`` 时 early return (reflex 反射弧).
    - ``_action_loop``: 从 ``mindflow.action_loop()`` 取 ``Action``, ``wait_ready``
      后用 ``action.logos()`` 作为 interpreter 输入流, 执行 logos.
    - ``interrupt == True`` 的新 attention 起步先 ``shell.clear()`` 停旧 logos.
"""
import asyncio
from typing import AsyncIterable

import pytest

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode,
    Impulse,
    Priority,
    Thinking,
)
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.mindflow.command_nucleus import new_command_signal
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.message import Message

from .mindflow_in_shell_test_suite import (
    MindflowInShellTestSuite,
    input_signal,
)


@pytest.mark.asyncio
async def test_loop_baseline():
    """裸基线: 三循环协作跑通一条 logos.

    验证 ``signal → mindflow → thinking → articulate → action → interpreter``
    全链路通畅, 是后续抢占/中断测试的健康基线.
    """
    suite = MindflowInShellTestSuite()

    content = ''

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        nonlocal content
        async for chunk in chunks__:
            content += chunk

    suite.shell.main_channel.build.content_command(content_func)
    suite.articulate = suite.text_articulator("hello world")

    async with suite:
        suite.add_signal(input_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert content == 'hello world'
    assert suite.attention_count == 1
    assert suite.thinking_count == 1
    assert suite.articulation_done_count == 1
    assert suite.action_count == 1
    assert suite.action_done_count == 1
    assert not suite.exceptions
    # impulse 来自 InputSignalNucleus, 非 interrupt.
    impulse = suite.impulses[0]
    assert impulse.source == 'input_signal_nucleus'
    assert impulse.interrupt is False
    assert impulse.thinking_effort == ''


@pytest.mark.asyncio
async def test_command_signal_skips_articulate_runs_logos():
    """thinking_effort='none' 协议: CommandNucleus 产 impulse 走 reflex 路径.

    协议契约:
        - CommandNucleus(signal) → impulse with thinking_effort='none' + logos=command_logos
        - thinking 检查 thinking_effort=='none' → early return, **不调 articulate**
        - action 仍然 wait_ready / logos: moment.command_logos 已被预填到 logos 流,
          action 自然消费并交给 interpreter
        - 模型 articulate 完全没参与, 但 shell 执行了命令
    """
    suite = MindflowInShellTestSuite()

    executed = []

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            executed.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    articulate_called = 0

    async def articulate(_thinking: Thinking) -> None:
        nonlocal articulate_called
        articulate_called += 1

    suite.articulate = articulate

    async with suite:
        suite.add_signal(new_command_signal('reflex_logos'))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    # 协议字段.
    impulse = suite.impulses[0]
    assert impulse.source == 'command_nucleus'
    assert impulse.thinking_effort == 'none'
    assert impulse.logos == 'reflex_logos'
    # thinking 走 early-return 分支, articulate 不应被触发.
    assert articulate_called == 0
    # 但 action 跑了, content_command 收到 logos.
    assert ''.join(executed) == 'reflex_logos'
    assert suite.action_done_count == 1
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_single_input_signal_yields_exactly_one_percept():
    """一条 InputSignal 产生的第一帧 moment.percepts 不应有重复消息.

    这是 percepts 改为 source-keyed dict 的前置基线 — 确保当前正常路径
    下不产生重复, dict 迁移时零行为变化.
    """
    suite = MindflowInShellTestSuite()

    captured_percepts = []

    async def articulate(thinking: Thinking) -> None:
        captured_percepts.extend(list(thinking.moment.percepts_messages()))
        art = thinking.articulator()
        async with art:
            art.send_nowait("ok")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for _ in chunks__:
            pass

    suite.shell.main_channel.build.content_command(content_func)

    async with suite:
        suite.add_signal(input_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    # 一条信号只产生一组 percepts, 不应重复.
    assert len(captured_percepts) == 1
    assert captured_percepts[0].contents[0]["text"] == "hello"
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_articulate_streams_multiple_chunks():
    """边界: articulate 分多段 send, content_command 收到拼接后的完整流.

    钉住 logos 流式喂入的语义 — 多个 delta 不丢、不重、顺序保持.
    """
    suite = MindflowInShellTestSuite()

    content = ''

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        nonlocal content
        async for chunk in chunks__:
            content += chunk

    suite.shell.main_channel.build.content_command(content_func)

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            for piece in ("hello", " ", "world"):
                art.send_nowait(piece)
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("hi"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert content == 'hello world'
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_notify_signal_quiet_creates_attention():
    """边界: quiet 系统 (无 attention) 下 notify 走 default 路径创建 attention.

    钉住 ``ChallengeMode.notify`` 在"抢占成功侧"不偏离 default — 只有抢占失败侧
    才 buffer;quiet 时没有 defender, notify 正常创建 attention.
    """
    suite = MindflowInShellTestSuite()

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for _ in chunks__:
            pass

    suite.shell.main_channel.build.content_command(content_func)
    suite.articulate = suite.text_articulator("done")

    async with suite:
        suite.add_signal(new_notify_signal(Message.new().with_content("user_msg")))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert suite.attention_count == 1
    impulse = suite.impulses[0]
    assert impulse.source == 'notify_nucleus'
    assert impulse.mode == ChallengeMode.notify.value
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_interrupt_signal_stops_running_interpreter():
    """interrupt 协议: action 跑 long task 中, interrupt signal 抢占 + shell.clear.

    协议契约:
        - InterruptNucleus 产 impulse: FATAL + notify + thinking_effort='none' + interrupt=True
        - FATAL 必抢占, attention1 被 abort
        - 新 attention 起步先调 shell.clear() (interrupt 协议)
        - attention1 action 里运行的长任务被 CancelledError 取消
    """
    suite = MindflowInShellTestSuite()

    long_task_started = asyncio.Event()
    long_task_outcome = []  # 'cancelled' | 'completed'

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        # 消费 logos 流后, 模拟一个长任务.
        async for _ in chunks__:
            pass
        long_task_started.set()
        try:
            await asyncio.sleep(10.0)
            long_task_outcome.append('completed')
        except asyncio.CancelledError:
            long_task_outcome.append('cancelled')
            raise

    suite.shell.main_channel.build.content_command(content_func)

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            art.send_nowait("long_running_logos")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        # 第一帧: input signal 起 attention, 跑 long task.
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(long_task_started.wait(), timeout=1)
        # 长任务确实没结束.
        assert not long_task_outcome

        attention1 = suite.last_attention

        # 第二帧: interrupt signal 抢占.
        suite.add_signal(new_interrupt_signal(Message.new().with_content("halt")))

        # attention1 被 abort, attention2 起.
        await asyncio.wait_for(attention1.wait_abort(), timeout=2)
        # 等 attention2 也走完 (interrupt effort='none', 无 logos → 自然结束).
        for _ in range(100):
            if suite.attention_count >= 2 and suite.attention_stopped.is_set():
                break
            await asyncio.sleep(0.02)

    # 协议字段 (interrupt impulse 是第二个).
    assert suite.attention_count == 2
    impulse2 = suite.impulses[1]
    assert impulse2.source == 'interrupt_nucleus'
    assert impulse2.priority == Priority.FATAL
    assert impulse2.mode == ChallengeMode.notify.value
    assert impulse2.thinking_effort == 'none'
    assert impulse2.interrupt is True

    # attention 起步触发了 shell.clear (interrupt 协议).
    assert suite.interrupt_clear_calls == 1

    # 长任务收到 CancelledError.
    assert long_task_outcome == ['cancelled']
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_interrupt_during_articulate_aborts_logos_stream():
    """interrupt 时序切片 (2): 中断发生在 articulator 还在 send_logos 流式喂的阶段.

    协议契约:
        articulate 阶段被打断时, articulator 收到 abort, send_nowait 之后的剩余流不再
        传到 action; action 的 logos() 应该看到截断的流, interpreter
        不会执行后续命令.

    与 test_interrupt_signal_stops_running_interpreter 的区别: 上一个测中断打在
    action 里, 这里打在 articulate 里, 验证两个时序切片都能正确收线.
    """
    suite = MindflowInShellTestSuite()

    after_articulate_check = []

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            after_articulate_check.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    articulate_started = asyncio.Event()
    articulate_continue = asyncio.Event()
    articulate_finished = []  # 是否走到了 send 第二段.

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            art.send_nowait("first_chunk ")
            articulate_started.set()
            # 卡住, 等待外部触发 interrupt.
            try:
                await asyncio.wait_for(articulate_continue.wait(), timeout=3.0)
            except asyncio.CancelledError:
                articulate_finished.append('cancelled')
                raise
            art.send_nowait("second_chunk")
            articulate_finished.append('completed')

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(articulate_started.wait(), timeout=1)
        attention1 = suite.last_attention

        # 中断打在 articulate 卡住阶段.
        suite.add_signal(new_interrupt_signal(Message.new().with_content("halt")))
        await asyncio.wait_for(attention1.wait_abort(), timeout=2)

        # 等第二帧 attention 收线.
        for _ in range(100):
            if suite.attention_count >= 2 and suite.attention_stopped.is_set():
                break
            await asyncio.sleep(0.02)

    # articulate 阶段被 abort, 没走到 send 第二段.
    assert articulate_finished == ['cancelled']
    # content_command 只收到第一段 (并可能根本没收到, 取决于 attention 转接时序).
    seen = ''.join(after_articulate_check)
    assert 'second_chunk' not in seen
    # interrupt 协议字段.
    assert suite.attention_count == 2
    assert suite.impulses[1].interrupt is True
    assert suite.interrupt_clear_calls == 1
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_action_aborted_triggers_shell_clear_cancels_pending():
    """action abort 路径: shell.clear() 取消 pending command.

    协议契约:
        - action loop 在 feed/compile/execute 各阶段后检查 ``act.is_aborted()``
        - 一旦发现 abort, 调 ``shell.clear()`` 显式取消 pending command (而非
          仅靠 ``clear_after_exit=False`` 的 interpreter 自然退出 — 那条路径
          不会取消运行中的 task).
        - 被取消的 task 进 ``interpretation.cancelled_tasks``.

    关键设计: interpreter 用 ``kind='append', clear_after_exit=False`` 保证
    interpreter 退出本身不清 shell command (跨帧延续语义), 取消运行中 task
    是 action loop 的责任 — 通过 ``shell.clear()`` 显式触发.
    """
    suite = MindflowInShellTestSuite()

    long_task_started = asyncio.Event()
    long_task_outcome = []

    chan = new_channel(name="slow")

    @chan.build.command()
    async def long_task() -> str:
        long_task_started.set()
        try:
            await asyncio.sleep(5.0)
            long_task_outcome.append('completed')
            return 'done'
        except asyncio.CancelledError:
            long_task_outcome.append('cancelled')
            raise

    suite.shell.main_channel.import_channels(chan)

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            # 发完命令后继续喂字串, 制造 logos 循环还在转的窗口, 给外部 abort 时序点.
            art.send_nowait("<slow:long_task/>")
            for _ in range(50):
                await asyncio.sleep(0.05)
                art.send_nowait(" ")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(long_task_started.wait(), timeout=2)
        assert not long_task_outcome
        attention = suite.last_attention

        # 直接 abort attention — 不走 interrupt 协议, 测纯 action abort 路径.
        attention.abort('test abort')

        # 等 action 收尾 (会触发 shell.clear).
        for _ in range(100):
            if suite.action_done_count >= 1:
                break
            await asyncio.sleep(0.02)

    # long_task 被取消.
    assert long_task_outcome == ['cancelled']
    # action loop 触发了 shell.clear (至少一次, 时序可能 feed 阶段或 execute 阶段命中).
    assert suite.shell_clear_calls >= 1
    # interpretation 里有 cancelled task.
    interp_done = suite.interpretations[0]
    assert len(interp_done.cancelled_tasks) >= 1
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_observe_loop_runs_two_frames_in_one_attention():
    """attention 多帧循环: echo(observe=True) 触发第二帧, 一个 attention 跑两组 logos.

    协议契约 (点 1/5 — thinking 不能先于 last action 退出):
        - attention 不是"每个 signal 一个", 而是"持续观察直到自然结束".
        - 每帧 yield 一个 Thinking, 跑完后检查 need_observe.
        - 命令返回 ``CommandUtil.observe(...)`` → shell trajectory 通知 mindflow
          ``add_echoes(need_observe=True)`` → 进下一帧.
        - 关键: articulate 必须 ``await art.wait_action_done()`` — 否则 thinking 先于
          action 退出, observe 还没落盘, mindflow 会误判 attention 自然结束, 签发
          新 attention 而非同 attention 的第二帧.
    """
    suite = MindflowInShellTestSuite()

    chan = new_channel(name="probe")
    frames_seen: list[str] = []

    @chan.build.command()
    async def frame1() -> str:
        frames_seen.append('frame1')
        return CommandUtil.observe('observe_me')

    @chan.build.command()
    async def frame2() -> str:
        frames_seen.append('frame2')
        return 'done'

    suite.shell.main_channel.import_channels(chan)

    frame_idx = 0

    async def articulate(thinking: Thinking) -> None:
        nonlocal frame_idx
        art = thinking.articulator()
        async with art:
            if frame_idx == 0:
                art.send_nowait("<probe:frame1/>")
            else:
                art.send_nowait("<probe:frame2/>")
            frame_idx += 1
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        # 等两帧都跑完 + attention 自然结束.
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=3)

    # 一个 attention, 两次 articulation, 两段 logos 都执行.
    assert suite.attention_count == 1
    assert suite.thinking_count == 2
    assert suite.articulation_done_count == 2
    assert frames_seen == ['frame1', 'frame2']
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_mindflow_channel_help_not_duplicated_across_frames():
    """连续两帧 moment 不得重复携带同一份 mindflow channel help.

    协议契约 (mindflow channel help 的去重):
        - mindflow 反身 channel 被挂为 shell 的 virtual child. 它第一次进入 shell metas
          时 (frame1), facade 全量展示其 instruction/help/interface; 之后内容无变化时
          (frame2), ``ShellKeyFrame.facade_delta`` 应经 ``diff_facade`` 去重为空 —
          不得把同一份 help 再次全量 emit 给模型, 否则模型连续两帧看到同一份 help.
        - 本测试直接断言: "mindflow nuclei" (help 头部) 只在某一帧的 echo 里出现一次.
          若两帧都出现, 即重复 — 失败, 由我们 debug.
    """
    suite = MindflowInShellTestSuite()

    # 用 probe 命令驱动两帧: frame1 返回 observe → 触发第二帧.
    chan = new_channel(name="probe")

    @chan.build.command()
    async def frame1() -> str:
        return CommandUtil.observe('observe_me')

    @chan.build.command()
    async def frame2() -> str:
        return 'done'

    suite.shell.main_channel.import_channels(chan)

    frame_moments: list = []

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            idx = len(frame_moments)
            frame_moments.append(thinking.moment)
            if idx == 0:
                art.send_nowait("<probe:frame1/>")
            else:
                art.send_nowait("<probe:frame2/>")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=3)

    # 一个 attention, 两帧 (两次 articulation).
    assert suite.attention_count == 1
    assert len(frame_moments) == 2

    def _help_present(moment) -> bool:
        return any(
            "mindflow nuclei" in m.to_content_string()
            for m in moment.previous_echoes_messages()
        )

    hits = [i for i, m in enumerate(frame_moments) if _help_present(m)]

    # 前置: mindflow help 确实会流经 moment 的 echo (frame1 首次进入 metas 时全量).
    assert hits, "mindflow help 未出现在任何 frame 的 echo 中 — 测试作用面可能变了"
    # 核心契约: help 不得在两帧重复出现.
    assert len(hits) == 1, f"mindflow help 在 frame(s) {hits} 重复出现"


@pytest.mark.asyncio
async def test_notify_buffer_drains_to_next_attention_percepts():
    """notify 抢占失败 → mindflow buffer → 下一帧 attention 的 moment.percepts.

    协议契约 (点 3 — notify 保护期不丢):
        - ``ChallengeMode.notify`` 抢占失败时 (含保护期内), messages 进 mindflow
          buffer 而非 suppress (notify 偏离侧).
        - mindflow 在下一帧 observe 时把 buffer drain 到 moment.percepts
          (source = "MomentsInjectedPercepts").
    """
    suite = MindflowInShellTestSuite()

    captured_percepts: list[list] = []
    notify_text = "notify_payload_xyz"

    attention1_running = asyncio.Event()
    release_attention1 = asyncio.Event()
    attention2_articulate_done = asyncio.Event()

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            if not attention1_running.is_set():
                # 第一帧 (attention1): 卡住, 给主测试时间发 notify 并验 buffer.
                attention1_running.set()
                await asyncio.wait_for(release_attention1.wait(), timeout=3.0)
                art.send_nowait("frame1_done")
            else:
                # 第二帧 (attention2): observe 已把 buffer drain 到 percepts.
                captured_percepts.append(list(thinking.moment.percepts_messages()))
                art.send_nowait("frame2_done")
                attention2_articulate_done.set()
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for _ in chunks__:
            pass

    suite.shell.main_channel.build.content_command(content_func)

    async with suite:
        # attention1: 用 add_impulse 带 protection_time 注入.
        suite.add_impulse(Impulse(
            priority=Priority.NOTICE,
            protection_time=10.0,
            messages=[Message.new().with_content("user_first")],
        ))
        await asyncio.wait_for(attention1_running.wait(), timeout=2)

        # 发 notify signal — 保护期内 NOTICE 同优先级, 抢占失败 → buffer.
        suite.add_signal(new_notify_signal(
            Message.new().with_content(notify_text),
            priority=Priority.NOTICE,
        ))
        await asyncio.sleep(0.2)

        # attention1 仍在跑 (notify 没抢占成功).
        assert suite.last_attention is not None
        assert not suite.last_attention.is_aborted()

        # buffer 里应该有 notify 内容.
        buffered = suite.mindflow.moments.peek()
        buffered_text = ''.join(
            c['text'] for m in buffered.percepts_messages() for c in m.contents if 'text' in c
        )
        assert notify_text in buffered_text

        # 放 attention1 跑完.
        release_attention1.set()
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

        # attention2: 再发一个 input signal.
        suite.add_signal(input_signal("user_second"))
        await asyncio.wait_for(attention2_articulate_done.wait(), timeout=2)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

    assert suite.attention_count == 2
    assert captured_percepts, "frame2 articulate 没拿到 percepts"
    frame2_percepts_text = ''.join(
        c['text'] for m in captured_percepts[0] for c in m.contents if 'text' in c
    )
    assert notify_text in frame2_percepts_text
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_interrupt_preempts_protected_attention():
    """FATAL 必抢占, 无视保护期 (点 1 — challenge 结果与 mode 正交).

    协议契约:
        - ``priority == FATAL`` 是挑战结果轴 (win), ``mode`` 是处置轴 (buffer/preempt).
        - 保护期只应让"同/低优先级"的挑战失败, 不该压过 FATAL.
        - interrupt = FATAL + notify + interrupt=True → 即使当前 attention 在保护期内,
          也必须抢占成功 (abort 当前 attention, 建新 attention), 而非被 buffer.
    """
    suite = MindflowInShellTestSuite()

    release_attention1 = asyncio.Event()

    async def articulate(thinking: Thinking) -> None:
        art = thinking.articulator()
        async with art:
            art.send_nowait("frame1")
            # 卡住保持 attention1 活着, 让 interrupt 有机会抢它.
            await asyncio.wait_for(release_attention1.wait(), timeout=3.0)
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for _ in chunks__:
            pass

    suite.shell.main_channel.build.content_command(content_func)

    async with suite:
        # attention1: 带保护期, 让同优先级挑战必败.
        suite.add_impulse(Impulse(
            priority=Priority.NOTICE,
            protection_time=10.0,
            messages=[Message.new().with_content("user_first")],
        ))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        attention1 = suite.last_attention
        assert attention1 is not None

        # interrupt 抢占 — FATAL 必须无视保护期.
        suite.add_signal(new_interrupt_signal(Message.new().with_content("halt")))
        await asyncio.wait_for(attention1.wait_abort(), timeout=2)

        # 等 attention2 (interrupt) 收线.
        for _ in range(100):
            if suite.attention_count >= 2:
                break
            await asyncio.sleep(0.02)

    release_attention1.set()

    # FATAL 抢占了保护期内的 attention: 两个 attention, 第二个是 interrupt.
    assert suite.attention_count == 2
    assert suite.impulses[1].interrupt is True
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_three_loop_cycles_20_rounds():
    """三循环拓扑 20 轮: thinking / attention / action 交互.

    协议验证 (预测先行):
        - 每轮 thinking 产两帧 articulator: "hello" wait_action_done (同步),
          "world" wait_compiled (interleaved 超速), 两者之间 observe 一次 moment.
        - action 侧: "hello" 快退; "world" set_compiled 后 sleep 5s.
        - attention 旁路 (subscription): 每次 event_k 置位就 abort 当前 attention.
        - abort 级联砍断 "world" 的 5s.

    预测: 20 个 attention 被 abort, 20 个 hello 完成, 20 个 world 被 abort (0 完成),
    20 次 articulate observe (moment).
    """
    suite = MindflowInShellTestSuite()

    event_k = asyncio.Event()

    hello_done = 0
    world_done = 0
    world_aborted = 0
    attention_aborted = 0
    moment_observed = 0

    async def articulate(thinking: Thinking) -> None:
        nonlocal moment_observed
        event_k.clear()
        # articulator 1: hello (同步)
        art1 = thinking.articulator()
        async with art1:
            art1.send_nowait('hello')
            await art1.wait_action_done()
        # 两次 articulator 之间 observe 一次 moment.
        thinking.observe()
        moment_observed += 1
        # articulator 2: world (interleaved, 只等 compiled)
        art2 = thinking.articulator()
        async with art2:
            art2.send_nowait('world')
            await art2.wait_compiled()
        event_k.set()

    suite.articulate = articulate

    # 覆写 action: 手动 hello / world 语义.
    async def _run_action(action) -> None:
        nonlocal hello_done, world_done, world_aborted
        try:
            async with action:
                await action.wait_ready()
                if action.is_aborted():
                    return

                async def _action_func() -> None:
                    nonlocal hello_done, world_done, world_aborted
                    async for delta in action.logos():
                        if delta == 'hello':
                            hello_done += 1
                        elif delta == 'world':
                            action.set_compiled()
                            try:
                                await asyncio.sleep(5.0)
                                world_done += 1
                            except asyncio.CancelledError:
                                world_aborted += 1
                                raise

                await action.wait_until_done(asyncio.ensure_future(_action_func()))
        except asyncio.CancelledError:
            raise

    suite._run_action = _run_action

    # attention 旁路: subscription, 每次 event_k 置位 abort 当前 attention.
    async def attention_bypass() -> None:
        nonlocal attention_aborted
        async for attention in suite.mindflow.attention_loop():
            await event_k.wait()
            attention.abort('bypass')
            attention_aborted += 1
            event_k.clear()

    async with suite:
        bypass_task = asyncio.create_task(attention_bypass())
        # InputSignalNucleus 是 FIFO 聚合, 一次性发 20 个会合成一个 impulse;
        # 这里逐个发, 每个 cycle (attention 被旁路 abort) 完成后才发下一个.
        for i in range(20):
            suite.add_signal(input_signal(f"msg_{i}"))
            for _ in range(1000):
                if attention_aborted >= i + 1:
                    break
                await asyncio.sleep(0.005)
        # 退出前捕获, mindflow.__aexit__ 会 _clear() 清空 moments.
        moment_count = len(suite.mindflow.moments.moments())
        bypass_task.cancel()

    assert attention_aborted == 20
    assert hello_done == 20
    assert world_done == 0
    assert world_aborted == 20
    assert moment_observed == 20
    assert moment_count == 40
