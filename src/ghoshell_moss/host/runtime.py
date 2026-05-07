from typing_extensions import Self

import janus

from ghoshell_moss import Message, MOSShell, CTMLShell
from ghoshell_moss.host.abcd.host_design import (
    MossRuntime, MossMode,
)
from ghoshell_moss.host.abcd.app import AppStore
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.contracts import Workspace
from .app_store import HostAppStore
from .matrix import MatrixImpl
from ghoshell_moss.host.abcd.environment import Environment
from ghoshell_moss.host.channels.app_store_channel import AppStoreChannel
import contextlib
import asyncio

__all__ = ['MossRuntimeImpl']


class MossRuntimeImpl(MossRuntime):

    def __init__(
            self,
            env: Environment,
            workspace: Workspace,
            mode: MossMode,
            matrix: MatrixImpl,
    ):
        env.bootstrap()
        self._env = env
        self._workspace = workspace
        self._matrix = matrix
        self._mode = mode
        self._ctml_shell: CTMLShell | None = None
        self._app_store: HostAppStore | None = None
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        self._paused = False
        self._close_event = ThreadSafeEvent()
        self._log_prefix = f"<HostMossRuntime mode={self._mode.name} session_id={self._env.session_scope}>"
        self._interpreting_future: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._action_task: asyncio.Task | None = None
        self._started = False
        # --- shell action loop --- #
        self._shell_logos_queue: janus.Queue = janus.Queue()
        # --- prepare shell --- #
        system_prompt = self._matrix.moss_system_prompter()
        self._ctml_shell = new_ctml_shell(
            name="MOSS." + self._mode.name,
            description=self._mode.description,
            parent_container=self.matrix.container,
            experimental=False,
            meta_instruction=system_prompt.instruction(),
            # 只用环境发现的原语. 不做任何隐式原语.
            primitives=list(self._matrix.manifests.primitives().values()),
        )

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError('Moss is not running.')

    def moss_instruction(self, with_static: bool = True) -> str:
        self._check_running()
        instructions = [self._ctml_shell.meta_instruction()]

        if with_static:
            if static_messages := self._ctml_shell.static_messages().strip():
                instructions.append("# MOSS static\n\n" + static_messages)
        return "\n\n".join(instructions)

    async def moss_dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        self._check_running()
        await self._ctml_shell.refresh_metas(max_wait)
        return self._ctml_shell.dynamic_messages()

    def moss_static_messages(self) -> str:
        return self._ctml_shell.static_messages()

    async def moss_observe(
            self,
            timeout: float | None = None,
            priority: int = 0,
            with_dynamic: bool = True,
    ) -> list[Message]:
        self._check_running()
        # 返回最新的 perception.
        return []

    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        self._check_running()
        interpreter = await self._ctml_shell.interpreter(
            kind='clear' if call_soon else 'append',
            clear_after_exit=False,
        )
        interpretation = interpreter.interpretation()
        async with interpreter:
            interpreter.feed(logos)
            await interpreter.wait_compiled()
            if wait_done:
                await interpreter.wait_stopped()
        return interpretation.executed_messages()

    async def moss_interrupt(self) -> list[Message]:
        self._check_running()
        # 清空状态.
        await self._ctml_shell.clear()
        interpreter = self._ctml_shell.interpreting()
        if interpreter is None:
            return [Message.new().with_content('no logos are executing')]
        else:
            return interpreter.interpretation().executed_messages()

    def is_running(self) -> bool:
        return self._started and not self._close_event.is_set()

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
        self._check_running()
        return self._app_store

    @property
    def shell(self) -> MOSShell:
        self._check_running()
        return self._ctml_shell

    @property
    def matrix(self) -> Matrix:
        return self._matrix

    def _bootstrap_after_matrix(self) -> None:
        self._app_store = HostAppStore(
            env=self._env,
            workspace=self._workspace,
            namespace="MOSS/app_store/main",
            runnable=True,
            include=self._mode.apps,
            bringup=self._mode.bringup_apps,
            logger=self.matrix.logger,
        )
        # 注册 Apps
        self._ctml_shell.main_channel.import_channels(
            AppStoreChannel(name='apps')
        )
        self._matrix.container.set(AppStore, self._app_store)
        self._matrix.container.set(MOSShell, self._ctml_shell)
        self._matrix.container.set(CTMLShell, self._ctml_shell)

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError('Host Toolset is already started')
        self._started = True
        await self._async_exit_stack.__aenter__()
        # 启动 matrix.
        await self._async_exit_stack.enter_async_context(self._matrix)
        # 启动 app 并且 bringup
        self._bootstrap_after_matrix()
        await self._async_exit_stack.enter_async_context(self._app_store)
        # 启动 ctml shell
        await self._async_exit_stack.enter_async_context(self._ctml_shell)
        await self._ctml_shell.refresh_metas()
        # 注册日志到当前 app store 里.
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self._matrix.logger.exception("%s failed to aexit %s", self._log_prefix, e)
            raise e
        finally:
            self._close_event.set()
