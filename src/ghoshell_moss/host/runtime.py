from typing import Literal, Self

import janus

from ghoshell_moss import Message, MOSShell
from ghoshell_moss.host.abcd.host_interface import (
    MossRuntime, ToolSet, Perception, MossMode,
    Conceive,
)
from ghoshell_moss.host.abcd.app import AppStore
from ghoshell_moss.host.abcd.matrix import Matrix
from ghoshell_moss.core.concepts.mindflow import Mindflow, Signal, InputSignal
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.contracts import Workspace
from .abcd import OutputItem
from .app_store import HostAppStore
from .matrix import HostMatrix
from ghoshell_moss.host.abcd.environment import Environment
import contextlib
import asyncio


class Logos:

    def __aiter__(self):
        return self

    async def __anext__(self):
        pass


class HostMossRuntime(MossRuntime, ToolSet):

    def __init__(
            self,
            env: Environment,
            workspace: Workspace,
            mode: MossMode,
            matrix: HostMatrix,
            mindflow: Mindflow | None = None,
            as_toolset: bool = False,
            conceive: Conceive | None = None,
    ):
        env.bootstrap()
        self._env = env
        self._workspace = workspace
        self._matrix = matrix
        self._mode = mode
        self._as_toolset = as_toolset
        self._ctml_shell = new_ctml_shell(
            name="MOSS." + self._mode.name,
            description=self._mode.description,
            container=self.matrix.container,
            experimental=False,
        )
        self._app_store = HostAppStore(
            env=self._env,
            workspace=self._workspace,
            namespace="MOSS/app_store/main",
            runnable=True,
            include=self._mode.apps,
            bringup=self._mode.bringup,
        )
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        self._paused = False
        self._close_event = ThreadSafeEvent()
        self._log_prefix = f"<HostMossRuntime mode={self._mode.name} session_id={self._env.session_scope}>"

        self._mindflow: Mindflow | None = mindflow

        self._interpreting_future: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._conceive_func: Conceive | None = None

        self._action_task: asyncio.Task | None = None

        # --- shell action loop --- #
        self._shell_logos_queue: janus.Queue = janus.Queue()

    @property
    def mode(self) -> str:
        return self._mode.name

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError('Moss is not running.')

    def moss_instruction(self) -> str:
        self._check_running()
        instructions = []
        if meta_instruction := self._env.meta_instruction.get_meta_instruction().strip():
            instructions.append(meta_instruction)
        if mode_instruction := self._mode.instruction.strip():
            instructions.append(mode_instruction)
        if static_messages := self._ctml_shell.static_messages().strip():
            instructions.append(static_messages)
        return "\n".join(instructions)

    def moss_dynamic_messages(self) -> list[Message]:
        return self._ctml_shell.dynamic_messages()

    async def moss_observe(
            self,
            timeout: float | None = None,
            priority: int = 0,
            with_dynamic: bool = True,
    ) -> list[Message]:
        self._check_running()
        if timeout and timeout > 0:
            await asyncio.wait_for(self._observe(timeout), timeout=timeout)
        else:
            await self._observe(timeout=timeout)
        # 返回最新的 perception.
        return list(self._pop_perception().as_messages())

    async def _observe(self, timeout: float | None = None) -> None:
        """
        一次观察包含两个语义.
        1. 躯体运行正常结束, 或者异常结束.
        2. 预热了 refresh metas, 拿到最新的 meta.
        在这个过程中, 也会新的数据积累.
        """
        refresh = self._ctml_shell.refresh_metas(timeout=timeout)
        if self._action_task is not None and not self._action_task.done():
            await self._action_task
        await refresh

    def _pop_perception(self) -> Perception:
        """
        perception 由三部分组成:
        1. buffer 的外部世界输入, 通过 mindflow 进行加工和过滤.
        2. 已经运行结束的命令.
        3. 正在执行中的命令.
        4. dynamic
        """
        pass

    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
            with_dynamic: bool = True,
            priority: int = 0,
    ) -> list[Message]:
        pass

    async def moss_interrupt(self) -> str:
        pass

    def is_running(self) -> bool:
        pass

    def snapshot(self, new: bool = False, ack: bool = False) -> Perception:
        self._check_running()
        pass

    def ack_snapshot(self, snapshot: Perception) -> bool:
        pass

    def wait_close_sync(self, timeout: float | None = None) -> bool:
        return self._close_event.wait_sync(timeout)

    async def wait_close(self) -> None:
        await self._close_event.wait()

    def close(self) -> None:
        self._close_event.set()

    def pause(self, toggle: bool = True) -> None:
        self._check_running()
        self._ctml_shell.pause(toggle)
        self._paused = toggle

    @property
    def apps(self) -> AppStore:
        return self._app_store

    @property
    def shell(self) -> MOSShell:
        return self._ctml_shell

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        await self._async_exit_stack.__aenter__()
        # 启动 matrix.
        await self._async_exit_stack.enter_async_context(self._matrix)
        # 启动 app 并且 bringup
        await self._async_exit_stack.enter_async_context(self._app_store)
        # 启动 ctml shell
        await self._async_exit_stack.enter_async_context(self._ctml_shell)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self.logger.exception("%s failed to aexit %s", self._log_prefix, e)
        finally:
            self._close_event.set()
