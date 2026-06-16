"""Mindflow + Shell 集成 — 协议级三循环拓扑测试.

定位:
    GhostRuntimeImpl 把 ``mindflow.py:1404 __example__`` 的规范三循环 (main_loop /
    articulate_loop / action_loop, 通过两个 janus.Queue 解耦, 用 Logos 流耦合
    articulator 和 action) 工程化封装. 直接测 GhostRuntimeImpl 需要 mock 整个
    MossRuntime, 成本高且失真. 本文件用 **真 BaseMindflow + 真 CTMLShell**, 手写
    一份最小三循环编排, 验证抽象层的协议契约 — 不依赖 ghost / matrix / session.

设计约束 (来自 ``core.blueprint.mindflow``):
    - ``main_loop``: ``async for attention in mindflow.loop()`` → ``async for
      (articulator, action) in attention.loop()`` → 两 queue put_nowait.
      ``impulse.interrupt = True`` 时进 attention.loop 前先 ``shell.clear()`` —
      shell.clear 是 stop_interpretation 的超集 (清 speech + tree + interpreter),
      与 GhostRuntimeImpl._main_loop 对齐.
    - ``articulate_loop``: 从 articulator queue 取 articulator, 在它的生命周期里
      调 ``send_logos(...)`` 把模型 logos 流送出. 测试里直接灌固定字串.
    - ``action_loop``: 从 action queue 取 action, ``await action.wait_ready()``,
      用 ``action.received_logos()`` 作为 interpreter 的输入流, ``action.outcome(...)``
      回写结果. 这是 logos 流被 shell 真正执行的地方.

非目标:
    - 不测 ghost.articulate() / on_articulate_exit (没有 ghost).
    - 不测 moss_dynamic / refresh_metas 时序 (独立测试组).
    - 不测 GhostRuntimeImpl 的生命周期编排 (independent).
"""
import asyncio
import contextlib
from collections.abc import Coroutine
from typing import AsyncIterator, Awaitable, Callable, AsyncIterable

import janus
import pytest

from ghoshell_moss.core import CommandTask
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.concepts.interpreter import Interpretation
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.mindflow import (
    Action,
    Articulator,
    Attention,
    ChallengeMode,
    Impulse,
    InputSignal,
    Priority,
    Mindflow,
)
from ghoshell_moss.core.mindflow import (
    BaseMindflow,
    CommandNucleus,
    InputSignalNucleus,
    InterruptNucleus,
    NotifyNucleus,
)
from ghoshell_moss.core.mindflow.command_nucleus import new_command_signal
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.message import Message


# ============================================================
# Helpers
# ============================================================


def _build_mindflow() -> BaseMindflow:
    """Input + Interrupt + Command + Notify — 覆盖四种 nucleus 的最小集合."""
    return BaseMindflow(
        InputSignalNucleus(),
        InterruptNucleus(suppress_seconds=0.05),
        CommandNucleus(),
        NotifyNucleus(),
    )


def _input_signal(text: str, *, priority: Priority = Priority.NOTICE):
    return InputSignal().to_signal(
        Message.new().with_content(text),
        priority=priority,
    )


LogosProvider = Callable[[Articulator], Awaitable[AsyncIterator[str]]]


class ThreeLoopSuite:
    """三循环句柄, 供测试观察 + 关停."""

    def __init__(
            self,
            *,
            mindflow: Mindflow | None = None,
            shell: MOSShell | None = None,
    ):
        self.mindflow = mindflow or _build_mindflow()
        self.shell = shell or new_ctml_shell()
        self.observed_attentions: list[Attention] = []
        self.main_task: asyncio.Task | None = None
        self.articulate_task: asyncio.Task | None = None
        self.action_task: asyncio.Task | None = None
        self.attention_count: int = 0
        self.articulation_count: int = 0
        self.moments = []
        self.impulses = []
        self.articulation_done_count: int = 0
        self.action_count: int = 0
        self.action_done_count: int = 0
        self.attention_callback: Callable[[Attention], Coroutine] | None = None
        self.articulate_func: Callable[[Articulator], Coroutine] | None = None
        self.action_callback: Callable | None = None
        self.interpretations: list[Interpretation] = []
        # 记录 interrupt 协议触发的 shell.clear 调用次数 —
        # interrupt 协议 (main_loop 入口) 要求停止所有执行中的 logos.
        # shell.clear 是 stop_interpretation 的超集 — 关闭 interpreter +
        # 清 speech 缓冲 + 取消 runtime tree pending command tasks.
        # 与 shell_clear_calls (action abort 三阶段触发的 clear) 分开计,
        # 让 interrupt 协议和 abort 协议的断言彼此独立.
        self.interrupt_clear_calls: int = 0
        # 记录 shell.clear 调用次数 — action abort → clear 协议的反推依据.
        self.shell_clear_calls: int = 0
        # 思维奔逸 (mind wandering) 预留 flag.
        # False (current default): action 等所有 task 跑完 (wait_stopped) 才结束本帧,
        #   articulator 必须等执行完毕才能进下一帧 — 严格"思考→执行→观察→再思考".
        # True (future): action 在 wait_compiled 后立刻结束 (CTML 已解析, task 后台跑),
        #   articulator 立刻进下一帧思考, 真正实现全双工 — 执行的同时思考下一步.
        #   未来 ghost runtime 也应该把这个开关暴露给应用层, 配合 action.is_aborted /
        #   shell.clear 协议管控奔逸状态下的中断.
        self.action_returns_at_compiled: bool = False

        self.attention_started = asyncio.Event()
        self.attention_stopped = asyncio.Event()
        self.attention_stopped.set()
        self.last_attention: Attention | None = None
        self._art_q = janus.Queue[Articulator]()
        self._act_q = janus.Queue[Action]()
        self.exceptions: list[Exception] = []
        self._exit_stack = contextlib.AsyncExitStack()

    async def main_loop(self):
        try:
            await self.mindflow.wait_started()
            async for attention in self.mindflow.loop():
                self.attention_count += 1
                self.observed_attentions.append(attention)
                # 回调探知.
                if self.attention_callback is not None:
                    await self.attention_callback(attention)
                # 执行预设逻辑.
                impulse = attention.draw_from()
                self.impulses.append(impulse)
                if impulse.interrupt:
                    # interrupt 协议: 停止所有执行中的 logos.
                    # 与 GhostRuntimeImpl._main_loop 对齐 — shell.clear() 而非
                    # stop_interpretation, 是后者的超集 (清 speech + tree + interpreter).
                    self.interrupt_clear_calls += 1
                    await self.shell.clear()
                # 开启上下文.
                async with attention:
                    self.last_attention = attention
                    self.attention_started.set()
                    self.attention_stopped.clear()
                    async for art, act in attention.loop():
                        self._art_q.sync_q.put_nowait(art)
                        self._act_q.sync_q.put_nowait(act)

                self.attention_stopped.set()
                self.attention_started.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.exceptions.append(e)

    async def articulate_loop(self):
        try:
            while True:
                art = await self._art_q.async_q.get()
                # 从 queue 收到 articulator 才算一次 articulation. 修正原版 +1 时序.
                self.articulation_count += 1
                async with art:
                    self.moments.append(art.moment)
                    # 时序契约 (与 GhostRuntimeImpl._run_articulator 对齐):
                    # 1. refresh_metas 阻塞 — 拿实时 perspectives, timeout/stale_time 等值 0.5s
                    #    (人类感知阈值内, 慢通道理论上应自行改推模式).
                    await self.shell.refresh_metas(0.5, stale_time=0.5)
                    # 2. command_logos 预发送给 action.
                    if art.moment.command_logos:
                        art.send_nowait(art.moment.command_logos)
                    if art.thinking_effort() == 'none':
                        # 模拟 _run_articulator early return: 不调 ghost.articulate.
                        # 注意: early return 路径不拼 moss_dynamic — 不思考就不需要.
                        continue
                    # 3. moss_dynamic 注入 perspective — 复用 articulator 入口刚刷的缓存
                    #    (stale_time=0.5 保证命中, 零阻塞代价).
                    art.moment.with_perspective(
                        'moss_dynamic',
                        self.shell.dynamic_messages(available_only=True, stale_time=0.5),
                    )
                    # 4. articulate.
                    if self.articulate_func:
                        await art.create_task(
                            self.articulate_func(art)
                        )
                self.articulation_done_count += 1
        except janus.AsyncQueueShutDown:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.exceptions.append(e)

    async def action_loop(self):
        try:
            while True:
                act = await self._act_q.async_q.get()
                self.action_count += 1
                async with act:
                    if self.action_callback:
                        await act.create_task(self._run_action(act))
                    else:
                        await self._run_action(act)
                # 时序契约 (与 GhostRuntimeImpl._run_action 对齐):
                # action 结束触发 fire-and-forget refresh_metas, 预热下一轮
                # articulator 入口的 stale_time 检查. 不 await — 让 action_loop
                # 立即进下一轮. 异常吞掉记 warning, 不影响主循环.
                asyncio.create_task(self._post_action_refresh())
                self.action_done_count += 1

        except janus.AsyncQueueShutDown:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.exceptions.append(e)

    async def _post_action_refresh(self) -> None:
        """fire-and-forget refresh, 内部捕获异常防 task 静默崩溃."""
        try:
            await self.shell.refresh_metas()
        except Exception as e:
            # 与 GhostRuntimeImpl 异常分级一致: 非关键路径异常不中断主循环.
            self.exceptions.append(e)

    async def _run_action(self, act: Action) -> None:
        """对齐 ``GhostRuntimeImpl._stream_execute`` 的标准三阶段实现.

        协议契约:
            - 三阶段 (feed → compile → execute) 各自结束后 check ``act.is_aborted()``,
              发现 abort 立刻 ``shell.clear()`` 取消 pending command, 返回.
            - ``InterpretError`` 是一等控制流 feature: 模型错 CTML / 命令异常时,
              interpreter 内部已保留 partial results + 标记 observe=True, 调用方
              捕获后让 attention 进下一帧自我纠正. **不 swallow 成静默**.
            - 不捕获 ``Exception`` 兜底 — ``CancelledError`` 继承 ``BaseException``
              本就不被 ``except Exception`` 拦, 其他异常应该 bubble 让 action loop
              的统一 handler 进 ``self.exceptions``.
        """
        await act.wait_ready()
        if act.is_aborted():
            return
        interp = await self.shell.interpreter(kind='append', clear_after_exit=False)
        # 提前拿出 interpretation.
        interpretation = interp.interpretation()
        self.interpretations.append(interpretation)

        def _on_task(task: CommandTask) -> None:
            r = task.task_result()
            act.outcome(*r.as_messages(), observe=r.observe)

        async def _check_abort_and_clear() -> bool:
            if not act.is_aborted():
                return False
            self.shell_clear_calls += 1
            await self.shell.clear()
            return True

        async with interp:
            interp.on_task_done(_on_task)
            try:
                # 阶段 1: feed. received_logos 自带 is_aborted 检查, abort 时自然 break.
                async for delta in act.received_logos():
                    interp.feed(delta)
                if await _check_abort_and_clear():
                    return
                # 阶段 2: commit + wait_compiled.
                interp.commit()
                await interp.wait_compiled()
                if await _check_abort_and_clear():
                    return
                # 思维奔逸切入点: 若开关打开, CTML 解析完毕即结束本帧, task 后台继续跑,
                # articulator 可立即进下一帧. 默认关闭 — 严格等所有 task 跑完.
                if self.action_returns_at_compiled:
                    return
                # 阶段 3: wait_stopped — 所有 task 跑完.
                await interp.wait_stopped()
                if await _check_abort_and_clear():
                    return
            except InterpretError:
                # 控制流 feature — interpretation 已保留 partial results, observe=True
                # 让 attention 进下一帧, 模型自我纠正. 真生产里还会 session.output('error', ...).
                pass

    async def __aenter__(self):
        await self._exit_stack.__aenter__()
        await self._exit_stack.enter_async_context(self.shell)
        await self._exit_stack.enter_async_context(self.mindflow)
        self.action_task = asyncio.create_task(self.action_loop())
        self.articulate_task = asyncio.create_task(self.articulate_loop())
        self.main_task = asyncio.create_task(self.main_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.action_task.cancel()
        self.articulate_task.cancel()
        self.main_task.cancel()
        await self._exit_stack.__aexit__(exc_type, exc, tb)


# ============================================================
# Tests
# ============================================================


@pytest.mark.asyncio
async def test_loop_baseline():
    """裸基线: 三循环协作跑通一条 logos.

    验证 ``signal → mindflow → attention → (articulator, action) → shell.interpreter``
    全链路通畅, 是后续抢占/中断测试的健康基线.
    """
    suite = ThreeLoopSuite()

    content = ''

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        nonlocal content
        async for chunk in chunks__:
            content += chunk

    suite.shell.main_channel.build.content_command(content_func)

    async def articulate(art: Articulator) -> None:
        art.send_nowait("hello world")

    suite.articulate_func = articulate

    async with suite:
        suite.mindflow.add_signal(InputSignal().to_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert content == 'hello world'
    assert suite.attention_count == 1
    assert suite.articulation_count == 1
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

    协议契约 (06-12 设计决策 #3):
        - CommandNucleus(signal) → impulse with thinking_effort='none' + logos=command_logos
        - main_loop 仍然展开 (articulator, action) 配对
        - articulator 检查 thinking_effort=='none' → early return, **不调 send_logos**
        - action 仍然 wait_ready / received_logos: moment.command_logos 已被 attention
          预填到 logos 流, action 自然消费并交给 interpreter
        - 模型 articulate 完全没参与, 但 shell 执行了命令

    这把 "reflex 反射弧" 的协议层钉住: 命令绕过模型思考但保留完整 shell 执行链.
    """
    suite = ThreeLoopSuite()

    executed = []

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            executed.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    articulate_called = 0

    async def articulate(_art: Articulator) -> None:
        nonlocal articulate_called
        articulate_called += 1

    suite.articulate_func = articulate

    async with suite:
        suite.mindflow.add_signal(new_command_signal('reflex_logos'))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    # 协议字段.
    impulse = suite.impulses[0]
    assert impulse.source == 'command_nucleus'
    assert impulse.thinking_effort == 'none'
    assert impulse.logos == 'reflex_logos'
    # articulator 走 early-return 分支, articulate_func 不应被触发.
    assert articulate_called == 0
    # 但 action 跑了, content_command 收到 logos.
    assert ''.join(executed) == 'reflex_logos'
    assert suite.action_done_count == 1
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_interrupt_signal_stops_running_interpreter():
    """interrupt 协议: action 跑 long task 中, interrupt signal 抢占 + shell.clear.

    协议契约 (GhostRuntimeImpl._main_loop 与本套件对齐):
        - InterruptNucleus 产 impulse: priority=FATAL + mode=notify + thinking_effort='none' + interrupt=True
        - FATAL 必抢占, attention1 被 abort
        - main_loop 在进入 attention2.loop() 前调 shell.stop_interpretation()
        - attention1 action 里运行的长任务被 CancelledError 取消
    """
    suite = ThreeLoopSuite()

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

    async def articulate(art: Articulator) -> None:
        art.send_nowait("long_running_logos")

    suite.articulate_func = articulate

    async with suite:
        # 第一帧: input signal 起 attention, 跑 long task.
        suite.mindflow.add_signal(InputSignal().to_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(long_task_started.wait(), timeout=1)
        # 长任务确实没结束.
        assert not long_task_outcome

        attention1 = suite.last_attention

        # 第二帧: interrupt signal 抢占.
        suite.mindflow.add_signal(new_interrupt_signal(Message.new().with_content("halt")))

        # attention1 被 abort, attention2 起.
        await asyncio.wait_for(attention1.wait_aborted(), timeout=2)
        # 等 attention2 也走完 (interrupt impulse thinking_effort='none', 没有 logos → 自然结束).
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

    # main_loop 触发了 shell.clear (interrupt 协议).
    assert suite.interrupt_clear_calls == 1

    # 长任务收到 CancelledError.
    assert long_task_outcome == ['cancelled']
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_interrupt_during_articulate_aborts_logos_stream():
    """interrupt 时序切片 (2): 中断发生在 articulator 还在 send_logos 流式喂的阶段.

    协议契约:
        articulate 阶段被打断时, articulator 收到 abort, send_nowait 之后的剩余流不再
        传到 action; action 的 received_logos() 应该看到截断的流, interpreter
        不会执行后续命令.

    与 test_interrupt_signal_stops_running_interpreter 的区别: 上一个测中断打在
    action 里, 这里打在 articulate 里, 验证两个时序切片都能正确收线.
    """
    suite = ThreeLoopSuite()

    after_articulate_check = []

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            after_articulate_check.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    articulate_started = asyncio.Event()
    articulate_continue = asyncio.Event()
    articulate_finished = []  # 是否走到了 send 第二段.

    async def articulate(art: Articulator) -> None:
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

    suite.articulate_func = articulate

    async with suite:
        suite.mindflow.add_signal(InputSignal().to_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(articulate_started.wait(), timeout=1)
        attention1 = suite.last_attention

        # 中断打在 articulate 卡住阶段.
        suite.mindflow.add_signal(new_interrupt_signal(Message.new().with_content("halt")))
        await asyncio.wait_for(attention1.wait_aborted(), timeout=2)

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

    协议契约 (GhostRuntimeImpl._stream_execute 三阶段 abort 检查):
        - action loop 在 feed/compile/execute 各阶段后检查 ``act.is_aborted()``
        - 一旦发现 abort, 调 ``shell.clear()`` 显式取消 pending command (而非
          仅靠 ``clear_after_exit=False`` 的 interpreter 自然退出 — 那条路径
          不会取消运行中的 task).
        - 被取消的 task 进 ``interpretation.cancelled_tasks``.

    关键设计: interpreter 用 ``kind='append', clear_after_exit=False`` 保证
    interpreter 退出本身不清 shell command (跨帧延续语义), 取消运行中 task
    是 action loop 的责任 — 通过 ``shell.clear()`` 显式触发.

    场景:
        - 注册一个长任务 ``slow:long_task`` (sleep 5s)
        - action loop 开始消费 logos, feed 进 interpreter, long_task 开始跑
        - 在 long_task 进入 sleep 后, 外部直接调 ``attention.abort('test')``
        - action loop 在 received_logos 循环 (feed 阶段) 检查到 is_aborted,
          调 shell.clear(), return.
        - 验证: long_task 收到 CancelledError, interpretation 里有 cancelled task,
          suite.shell_clear_calls == 1.
    """
    suite = ThreeLoopSuite()

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

    async def articulate(art: Articulator) -> None:
        # 发完命令后还要继续喂一些字串, 制造 received_logos 循环还在转的窗口,
        # 给外部 abort 一个时序点.
        art.send_nowait("<slow:long_task/>")
        # 多 send 几次小片让 feed 循环活跃一段时间. 每片之间 sleep 让出 control.
        for _ in range(50):
            await asyncio.sleep(0.05)
            art.send_nowait(" ")

    suite.articulate_func = articulate

    async with suite:
        suite.mindflow.add_signal(InputSignal().to_signal("user_msg"))
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
async def test_notify_buffer_drains_to_next_attention_percepts():
    """notify 抢占失败 → mindflow buffer → 下一帧 attention 的 moment.percepts.

    协议契约:
        - ``ChallengeMode.notify`` 抢占失败时, messages 进 ``mindflow._buffered_messages``
          而非 suppress (notify 偏离侧).
        - mindflow 在 ``_set_attention`` 时给 attention 注册 ``"MindflowBuffer"``
          percepts_func (= ``_pop_buffer``).
        - 下一帧 attention 的 ``_prepare_moment`` 调 percepts_funcs, buffer
          被 drain 到 ``moment.percepts``.

    场景:
        - attention1 起 (input signal, ``protection_time=10.0`` 让它无法被同优先级抢占)
        - 在 attention1 活的时候发 notify signal (NOTICE) → 抢占失败 (保护期内) → buffer
        - 验证 attention1 期间 ``mindflow.get_buffered(pop=False)`` 能看见 notify 内容
        - attention1 结束后 (用 abort), 注入第二个 input signal → attention2
        - 在 attention2 的 articulate_func 里检查 ``art.moment.percepts`` 包含 notify 文本
    """
    suite = ThreeLoopSuite()

    # 用 add_impulse 注入带保护期的 impulse. 普通 InputSignal 不带 protection_time.
    captured_percepts: list[list[Message]] = []
    notify_text = "notify_payload_xyz"

    attention1_running = asyncio.Event()
    release_attention1 = asyncio.Event()
    attention2_articulate_done = asyncio.Event()

    async def articulate(art: Articulator) -> None:
        if not attention1_running.is_set():
            # 第一帧.
            attention1_running.set()
            # 卡住, 给主测试时间发 notify signal 并验 buffer.
            await asyncio.wait_for(release_attention1.wait(), timeout=3.0)
            art.send_nowait("frame1_done")
        else:
            # 第二帧 — 此时 _prepare_moment 已经把 buffer drain 到 percepts.
            captured_percepts.append(list(art.moment.percepts_messages()))
            art.send_nowait("frame2_done")
            attention2_articulate_done.set()

    suite.articulate_func = articulate

    async def content_func(chunks__: AsyncIterable[str]) -> None:
        async for _ in chunks__:
            pass

    suite.shell.main_channel.build.content_command(content_func)

    async with suite:
        # attention1: 用 add_impulse 带 protection_time 注入.
        suite.mindflow.add_impulse(Impulse(
            priority=Priority.NOTICE,
            protection_time=10.0,
            messages=[Message.new().with_content("user_first")],
        ))
        await asyncio.wait_for(attention1_running.wait(), timeout=2)

        # 发 notify signal — 保护期内 NOTICE 同优先级, 抢占失败 → buffer.
        suite.mindflow.add_signal(new_notify_signal(
            Message.new().with_content(notify_text),
            priority=Priority.NOTICE,
        ))
        # 给 mindflow consume 时间.
        await asyncio.sleep(0.2)

        # attention1 仍在跑 (notify 没抢占成功).
        assert suite.last_attention is not None
        assert not suite.last_attention.is_aborted()

        # buffer 里应该有 notify 内容.
        buffered = suite.mindflow.get_buffered(pop=False)
        buffered_text = ''.join(
            c['text'] for m in buffered for c in m.contents if 'text' in c
        )
        assert notify_text in buffered_text

        # 放 attention1 跑完.
        release_attention1.set()
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

        # attention2: 再发一个 input signal.
        suite.mindflow.add_signal(InputSignal().to_signal("user_second"))
        await asyncio.wait_for(attention2_articulate_done.wait(), timeout=2)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

    assert suite.attention_count == 2
    # attention2 的 moment.percepts 应该 drain 到 notify buffer.
    assert captured_percepts, "frame2 articulate 没拿到 percepts"
    frame2_percepts_text = ''.join(
        c['text'] for m in captured_percepts[0] for c in m.contents if 'text' in c
    )
    assert notify_text in frame2_percepts_text
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_observe_loop_runs_two_frames_in_one_attention():
    """attention 多帧循环: outcome(observe=True) 触发第二帧, 一个 attention 跑两组 logos.

    协议契约 (base_attention.py:654-697 ``_loop``):
        - attention 不是"每个 signal 一个", 而是"持续观察直到自然结束".
        - 每帧 yield (Articulator, Action) 配对, 跑完后检查 observe_messages.
        - 若 action 调了 ``outcome(..., observe=True)`` (或抛 ObserveError),
          observe_messages 非 None → 进下一帧, 用 ``self._ctx.next_frame()`` 更新.
        - 若 observe_messages is None → 自然结束.

    这是 mindflow 抽象的核心 — Re-Act 循环里"行动 → 观察 → 再行动"的语义.

    场景:
        - 一个 attention, 两帧.
        - 第一帧: articulate 发 logos1 (``<probe:frame1/>``), action 跑 frame1,
          ``outcome(observe=True)`` 触发下一帧.
        - 第二帧: articulate 又被调一次 (新 articulator 实例), 发 logos2
          (``<probe:frame2/>``), action 跑 frame2, ``outcome(observe=False)``,
          attention 自然结束.
        - 验证: attention_count == 1 (单 attention), articulation_count == 2
          (两次 articulator), 两段 logos 都执行了.
    """
    suite = ThreeLoopSuite()

    chan = new_channel(name="probe")
    frames_seen: list[str] = []

    @chan.build.command()
    async def frame1() -> str:
        frames_seen.append('frame1')
        return 'observe_me'

    @chan.build.command()
    async def frame2() -> str:
        frames_seen.append('frame2')
        return 'done'

    suite.shell.main_channel.import_channels(chan)

    frame_idx = 0

    async def articulate(art: Articulator) -> None:
        nonlocal frame_idx
        if frame_idx == 0:
            art.send_nowait("<probe:frame1/>")
        else:
            art.send_nowait("<probe:frame2/>")
        frame_idx += 1

    suite.articulate_func = articulate

    # 自定 _run_action: frame1 outcome 带 observe=True, frame2 不带.
    # 用 ActionCallback hack 不方便, 直接 monkey-patch suite._run_action.

    call_idx = 0

    async def patched_run_action(act: Action) -> None:
        nonlocal call_idx
        await act.wait_ready()
        if act.is_aborted():
            return
        interp = await suite.shell.interpreter(kind='append', clear_after_exit=False)
        interpretation = interp.interpretation()
        suite.interpretations.append(interpretation)

        local_idx = call_idx
        call_idx += 1

        def _on_task(task: CommandTask) -> None:
            r = task.task_result()
            # 第一次 action observe=True (触发第二帧), 第二次 observe=False.
            observe = (local_idx == 0)
            act.outcome(*r.as_messages(), observe=observe)

        async with interp:
            interp.on_task_done(_on_task)
            try:
                async for delta in act.received_logos():
                    interp.feed(delta)
                interp.commit()
                await interp.wait_compiled()
                await interp.wait_stopped()
            except InterpretError:
                pass

    suite._run_action = patched_run_action

    async with suite:
        suite.mindflow.add_signal(InputSignal().to_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        # 等两帧都跑完 + attention 自然结束.
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=3)

    # 一个 attention, 两次 articulation, 两段 logos 都执行.
    assert suite.attention_count == 1
    assert suite.articulation_count == 2
    assert suite.articulation_done_count == 2
    assert frames_seen == ['frame1', 'frame2']
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_single_input_signal_yields_exactly_one_percept():
    """一条 InputSignal 产生的第一帧 moment.percepts 不应有重复消息.

    这是 percepts 改为 source-keyed dict 的前置基线 — 确保当前正常路径
    下不产生重复, dict 迁移时零行为变化.
    """
    suite = ThreeLoopSuite()

    captured_percepts: list[Message] = []

    async def articulate(art: Articulator) -> None:
        nonlocal captured_percepts
        captured_percepts.extend(list(art.moment.percepts_messages()))
        art.send_nowait("ok")

    suite.articulate_func = articulate

    async with suite:
        suite.mindflow.add_signal(_input_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    # 一条信号只产生一组 percepts, 不应重复.
    assert len(captured_percepts) == 1
    assert captured_percepts[0].contents[0]["text"] == "hello"
    assert not suite.exceptions
