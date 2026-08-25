"""MindflowInShellTestSuite — ``MindflowInShell`` 的测试专用具体子类.

定位:
    ``MindflowInShell`` (``core.mindflow.mindflow_in_shell``) 是三循环的标准
    装线逻辑: 把 ``Mindflow`` + ``MOSShell`` + ``MShellTrajectory`` 装线起来,
    跑 thinking / action 两个生产循环, 并用 interpreter 执行 logos. 它是 host
    (``GhostInShellDrivenByMindflow``) 复用的抽象基类 — host 只补环境 accessor
    (matrix / moss_runtime / session / container), 三循环本身不动.

    本套件就是 ``MindflowInShell`` 的 **测试专用具体子类**: 复用它的三循环,
    只补最小 accessor (无 matrix / moss_runtime / ghost), 把两个可拆卸单元
    (logos 来源 / signal 路由) 暴露给测试, 并挂上行为观测钩子. 它不是测试
    用例, 而是被 ``test_mindflow_shell_integration`` 各测试共用的参照装线.

可拆卸单元:
    - ``articulate``: ``Callable[[Thinking], Awaitable[None]]`` — logos 来源.
      对应 ``_articulate_from_thinking``, 主体 (模型 / ghost) 在此被抽掉;
      测试在此挂任何东西 (包括人类交互界面) 都合法.

      **articulate 自保证**: 需要同步的那一段, articulate 内部自行
      ``await art.wait_action_done()`` (见 ``text_articulator``) — 这是主路径:
      中间段可以 interleaved 超速, 最后一段必须等到 action 跑完才能拿到 observe.

      框架侧另有兜底: ``BaseThinking.__aexit__`` 的 ``_wait_last_action_done()``
      在正常退出时等最后一个 action, 即便 articulate 忘了 wait 也不丢最后一帧.
    - ``add_signal`` / ``add_impulse``: signal / impulse 注入入口.

观测钩子 (通过覆写 ``MindflowInShell`` 的 hook 方法实现, 不改三循环):
    - attention / thinking / action 计数与事件
    - ``interrupt_clear_calls`` / ``shell_clear_calls`` 区分两种 shell.clear
    - ``interpretations`` / ``impulses`` / ``exceptions`` 观测
"""
import asyncio
import contextlib
from typing import Awaitable, Callable, Iterable

from ghoshell_container import Container

from ghoshell_moss.contracts import get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import (
    Impulse,
    InputSignalMeta,
    Mindflow,
    NucleusMeta,
    Priority,
    Signal,
    Thinking,
)
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.concepts.interpreter import Interpretation
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.core.mindflow import (
    BaseMindflow,
    CommandNucleus,
    InputSignalNucleus,
    InterruptNucleus,
    NotifyNucleus,
)
from ghoshell_moss.core.mindflow.mindflow_in_shell import MindflowInShell
from ghoshell_moss.message import Message

__all__ = [
    'MindflowInShellTestSuite',
    'build_mindflow',
    'input_signal',
    'ArticulateFunc',
]


def build_mindflow() -> BaseMindflow:
    """Input + Interrupt + Command + Notify — 覆盖四种 nucleus 的最小集合."""
    return BaseMindflow(
        InputSignalNucleus(),
        InterruptNucleus(suppress_seconds=0.05),
        CommandNucleus(),
        NotifyNucleus(),
    )


def input_signal(text: str, *, priority: Priority = Priority.NOTICE) -> Signal:
    return InputSignalMeta().to_signal(
        Message.new().with_content(text),
        priority=priority,
    )


ArticulateFunc = Callable[[Thinking], Awaitable[None]]


class MindflowInShellTestSuite(MindflowInShell):
    """MindflowInShell 的测试具体子类 — 复用三循环, 补 accessor + 可拆卸单元 + 观测."""

    def __init__(
            self,
            *,
            mindflow: Mindflow | None = None,
            shell: MOSShell | None = None,
            articulate: ArticulateFunc | None = None,
    ):
        self._mindflow = mindflow or build_mindflow()
        self._shell = shell or new_ctml_shell()
        self._shell_trajectory = MShellTrajectory(self._shell)
        self._logger = get_moss_logger()
        self._container = Container(name="mindflow_in_shell_test_suite")

        # 可拆卸单元.
        self.articulate: ArticulateFunc | None = articulate
        self._signal_route: Callable[[Signal], None] | None = None

        # 观测.
        self.observed_attentions: list = []
        self.impulses: list[Impulse] = []
        self.interpretations: list[Interpretation] = []
        self.exceptions: list[BaseException | str] = []
        self.attention_count: int = 0
        self.thinking_count: int = 0
        self.articulation_done_count: int = 0
        self.action_count: int = 0
        self.action_done_count: int = 0
        self.interrupt_clear_calls: int = 0
        self.shell_clear_calls: int = 0
        self.action_returns_at_compiled: bool = False

        self.attention_started = asyncio.Event()
        self.attention_stopped = asyncio.Event()
        self.attention_stopped.set()
        self.last_attention = None

        self._seen_att_id: str | None = None
        self._loop_tasks: list[asyncio.Task] = []
        self._exit_stack = contextlib.AsyncExitStack()

        self._bind_shell_clear_counter()

    def _bind_shell_clear_counter(self) -> None:
        """观测 shell.clear() 调用 — action abort → clear 协议的反推依据.

        interrupt 协议的 clear 已由 interrupt_clear_calls 语义级计数, 这里只数
        原始 shell.clear() 调用次数 (涵盖 action abort / 异常兜底两条路径).
        """
        original_clear = self._shell.clear

        def counting_clear() -> asyncio.Future[None]:
            self.shell_clear_calls += 1
            return original_clear()

        self._shell.clear = counting_clear

    # ── MindflowInShell 抽象 accessor ──────────────

    @property
    def mindflow(self) -> Mindflow:
        return self._mindflow

    @property
    def shell(self) -> MOSShell:
        return self._shell

    @property
    def shell_trajectory(self) -> MShellTrajectory:
        return self._shell_trajectory

    @property
    def logger(self):
        return self._logger

    @property
    def container(self) -> Container:
        return self._container

    def _collect_nuclei_metas(self) -> Iterable[NucleusMeta]:
        # 测试直接往 mindflow 里注入 concrete nuclei (build_mindflow), 不走 meta 工厂.
        return []

    def _when_signal_added(self, callback: Callable[[Signal], None]):
        self._signal_route = callback

    async def _refresh_shell(self) -> None:
        await self.shell.refresh_metas(timeout=0.5, stale_time=1.0)

    def _is_thinking_gated(self) -> bool:
        return False

    async def _articulate_from_thinking(self, thinking: Thinking) -> None:
        if self.articulate is not None:
            await self.articulate(thinking)

    async def _fire_interpreter_result(self, interpretation: Interpretation) -> None:
        self.interpretations.append(interpretation)

    async def _enter_async_context(self, manager: contextlib.AbstractAsyncContextManager) -> None:
        await self._exit_stack.enter_async_context(manager)

    # ── 观测钩子 (覆写 hook, 不改三循环) ───────────

    def _on_mindflow_loop_task(self, future: asyncio.Future, *, name: str | None = None) -> None:
        self._loop_tasks.append(future)

    def _on_thinking_start(self, thinking: Thinking) -> None:
        self.thinking_count += 1
        att = thinking.attention
        if att.id != self._seen_att_id:
            self._seen_att_id = att.id
            self.attention_count += 1
            self.observed_attentions.append(att)
            impulse = att.draw_from()
            self.impulses.append(impulse)
            self.last_attention = att
            self.attention_started.set()
            self.attention_stopped.clear()
            asyncio.create_task(self._attention_monitor(att))
            if impulse.interrupt:
                self.interrupt_clear_calls += 1

    def _on_thinking_exited(self, thinking: Thinking, err: BaseException | None) -> None:
        self.articulation_done_count += 1

    def _on_mindflow_error(self, error: BaseException | str) -> None:
        self.exceptions.append(error)

    async def _attention_monitor(self, att) -> None:
        try:
            await att.wait_abort()
        except asyncio.CancelledError:
            return
        self.attention_stopped.set()

    async def _run_action(self, action) -> None:
        self.action_count += 1
        try:
            await super()._run_action(action)
        finally:
            self.action_done_count += 1

    # ── 可拆卸单元: signal / impulse 注入 ──────────

    def add_signal(self, signal: Signal) -> None:
        if self._signal_route is not None:
            self._signal_route(signal)
        else:
            self.mindflow.add_signal(signal)

    def add_impulse(self, impulse: Impulse) -> None:
        self.mindflow.add_impulse(impulse)

    def text_articulator(self, text: str) -> ArticulateFunc:
        """便捷 logos 来源: 发送固定文本并等 action 跑完."""

        async def _talker(thinking: Thinking) -> None:
            art = thinking.articulator()
            async with art:
                art.send_nowait(text)
                if not thinking.is_aborted():
                    await art.wait_action_done()

        return _talker

    # ── 生命周期 ──────────────────────────────────

    async def __aenter__(self) -> 'MindflowInShellTestSuite':
        await self._exit_stack.__aenter__()
        await self._exit_stack.enter_async_context(self.shell)
        await self._exit_stack.enter_async_context(self.shell_trajectory)
        await self._wire_mindflow()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self._exit_stack.__aexit__(exc_type, exc, tb)
        finally:
            for task in self._loop_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._loop_tasks, return_exceptions=True)
