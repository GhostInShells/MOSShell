"""
MossRuntime concrete — 编排 CTMLShell + Matrix + Mode 生命周期.

wire-up 契约 (§ZZ):
- Matrix 层承接 SystemPrompter 基础契约 (contracts/system_prompter.py),
  MossRuntime 承接 MossSystemPrompter (blueprint/host.py, ctml/project/mode/static
  4 slot 命名访问器) — 跨 cell 展示 feature 过重, 本期不上.
- mode 参数化: mode 是 MossRuntime 关注点, 不进 matrix. matrix 无 .manifests/.mode.
- main channel 从 mode.manifests().channel() 单 Manifest 拿; 无声明用 new_shell_main_channel 兜底.
- static slot 在 shell 起来后接 ctml_shell.static_messages callable (dynamic leaf).
"""
from typing_extensions import Self

import janus

from ghoshell_moss.message.message import Message
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.ctml.shell.ctml_shell import CTMLShell
from ghoshell_moss.core.blueprint.host import (
    MossRuntime, MossSystemPrompter,
)
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import HostMode
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.states_channel import new_shell_main_channel
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import Workspace, SystemPrompter, BaseSystemPrompter

from ghoshell_moss.matrix.matrix_impl import MatrixImpl

import contextlib
import asyncio

__all__ = ['MossRuntimeImpl']


class _MossSystemPrompterImpl(BaseSystemPrompter, MossSystemPrompter):
    """具象化 MossSystemPrompter — BaseSystemPrompter 提供 tree/组装, MossSystemPrompter
    提供命名 slot API. 钻石继承合流, 一实例既作 SystemPrompter 又作 MossSystemPrompter.
    """
    pass


class MossRuntimeImpl(MossRuntime):

    def __init__(
            self,
            *,
            env: Environment,
            workspace: Workspace,
            mode: HostMode,
            matrix: MatrixImpl,
            run_shell_on_start: bool = True,
            name: str | None = None,
            description: str | None = None,
    ):
        # env 已由 Host 侧 seal (Host.__init__ 承担). 这里假设 env 已 sealed.
        # 二次 seal 会抛 EnvironmentSealedError.
        self._env = env
        self._workspace = workspace
        self._mode = mode
        self._matrix = matrix
        self._name = name or env.moss_meta.name
        # 描述发现三级优先: 传参 > mode > env moss_meta.
        self._description = (
            description
            or mode.meta.description
            or env.moss_meta.description
        )
        self._run_shell_on_start = run_shell_on_start

        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        self._paused = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._log_prefix = (
            f"<HostMossRuntime mode={self._mode.name} "
            f"session_id={self._env.network_scope}>"
        )
        self._interpreting_future: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._action_task: asyncio.Task | None = None

        # --- shell action loop --- #
        self._shell_logos_queue: janus.Queue = janus.Queue()

        # --- system prompter (matrix 层不构造, MossRuntime 层持有全 4 slot tree) --- #
        # ctml/project/mode 三 slot 在此填充; static slot 在 _bootstrap_after_matrix 补
        # (依赖 ctml_shell 起来).
        self._system_prompter: _MossSystemPrompterImpl = self._build_system_prompter()

        # --- prepare shell --- #
        # main channel 从 mode.manifests().channel() 单 Manifest 拿 (§ZZ-1 ModeManifests
        # 由 MossRuntime 承接). 无声明用默认空白 main 兜底.
        manifests_main = self._discover_main_channel()
        if manifests_main is None:
            manifests_main = new_shell_main_channel(
                description=f"Default main channel for {self._description or self._name}",
            )
        self._ctml_shell = new_ctml_shell(
            name=self._name,
            description=self._description,
            parent_container=self.matrix.container,
            main_channel=manifests_main,
            experimental=False,
            meta_instruction=self._system_prompter.instruction(),
        )

    def _build_system_prompter(self) -> _MossSystemPrompterImpl:
        """构造 MossSystemPrompter tree 的 ctml/project/mode 三 slot.

        static slot 依赖 ctml_shell.static_messages, 在 _bootstrap_after_matrix 补.
        """
        prompter = _MossSystemPrompterImpl(
            description=(
                "MOSS system instruction — assembled from ctml/project/mode/static layers."
            ),
        )
        # ctml slot: 版本优先 mode > moss_meta
        ctml_version = self._mode.meta.ctml_version or self._env.moss_meta.ctml_version
        ctml_prompt = self._load_ctml_prompt(ctml_version)
        prompter.with_prompter(
            MossSystemPrompter.CTML_SLOT,
            BaseSystemPrompter(
                own_instruction=ctml_prompt,
                description=f"CTML grammar prompt (version {ctml_version}).",
            ),
        )
        # project slot: workspace 根 MOSS.md 声明的 project instruction
        # (MossMeta.system_project 字段, 老代码写 system_prompt 是遗迹 bug —
        # 老 host/matrix.py 里同一 typo, 只是从未跑通过).
        prompter.with_prompter(
            MossSystemPrompter.PROJECT_SLOT,
            BaseSystemPrompter(
                own_instruction=self._env.moss_meta.system_project,
                description="Workspace root MOSS.md project instruction.",
            ),
        )
        # mode slot: 模式内 HOST.md 声明的 instruction
        prompter.with_prompter(
            MossSystemPrompter.MODE_SLOT,
            BaseSystemPrompter(
                own_instruction=self._mode.meta.system_prompt,
                description=f"Mode '{self._mode.name}' instruction.",
            ),
        )
        return prompter

    def _load_ctml_prompt(self, ctml_version: str) -> str:
        """从 project.ctml_versions() 加载指定版本的 CTML meta instruction 全文."""
        versions = self._matrix.project.ctml_versions()
        ctml_file = versions.get(ctml_version)
        if ctml_file is None:
            raise KeyError(
                f"CTML version {ctml_version!r} not found. "
                f"Available: {sorted(versions.keys())}"
            )
        return ctml_file.read_text(encoding='utf-8')

    def _discover_main_channel(self):
        """从 mode.manifests().channel() 拿 main channel Manifest.

        新 ABC (§ZZ-1 ModeManifests): channel() -> Manifest[PrimeChannel] 单值.
        老 API .channels().values() 已废, 老代码 next(...) 迭代形态一并作废.
        """
        try:
            manifests = self._mode.manifests()
        except Exception:
            return None
        try:
            channel_manifest = manifests.channel()
        except Exception:
            return None
        if channel_manifest is None or channel_manifest.is_error():
            return None
        try:
            return channel_manifest.value()
        except Exception:
            return None

    @property
    def name(self) -> str:
        return self._name or self._env.moss_meta.name

    @property
    def description(self) -> str:
        return self._description or self._env.moss_meta.description

    @property
    def mode(self) -> HostMode:
        return self._mode

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError('MossRuntime is not running.')

    def _check_shell_running(self):
        if not self.is_running() or not self._ctml_shell.is_running():
            raise RuntimeError('MossRuntime Shell is not running.')

    @property
    def env(self) -> Environment:
        return self._env

    def moss_instruction(self, with_static: bool = True) -> str:
        self._check_shell_running()
        instructions = [self._ctml_shell.meta_instruction()]

        if with_static:
            if static_messages := self._ctml_shell.static_messages().strip():
                instructions.append("# MOSS static\n\n" + static_messages)
        return "\n\n".join(instructions)

    async def moss_dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        self._check_shell_running()
        await self._ctml_shell.refresh_metas(max_wait)
        return self._ctml_shell.dynamic_messages()

    def moss_static_messages(self) -> str:
        return self._ctml_shell.static_messages()

    async def moss_refresh_metas(self, timeout: float = 2.0) -> None:
        self._check_shell_running()
        await self._ctml_shell.refresh_metas(timeout)

    async def moss_observe(
            self,
            timeout: float | None = None,
            with_dynamic: bool = True,
    ) -> list[Message]:
        self._check_shell_running()
        if interpreter := self._ctml_shell.interpreting():
            messages = interpreter.interpretation().status_messages()
        else:
            messages = []

        if with_dynamic:
            await self._ctml_shell.refresh_metas()
            dynamic_messages = self._ctml_shell.dynamic_messages()
            messages.extend(dynamic_messages)
        return messages

    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        self._check_shell_running()
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
        return interpretation.as_messages()

    async def moss_interrupt(self) -> list[Message]:
        self._check_running()
        await self._ctml_shell.clear()
        interpreter = self._ctml_shell.interpreting()
        if interpreter is None:
            return [Message.new().with_content('no logos are executing')]
        return interpreter.interpretation().executed_messages()

    def is_running(self) -> bool:
        return self._started and not (
                self._closing_event.is_set() or self._closed_event.is_set()
        )

    def wait_close_sync(self, timeout: float | None = None) -> bool:
        return self._closing_event.wait_sync(timeout)

    async def wait_close(self) -> None:
        await self._closing_event.wait()

    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        return self._closed_event.wait_sync(timeout)

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def close(self) -> None:
        self._closing_event.set()

    def pause(self, toggle: bool = True) -> None:
        self._check_running()
        self._ctml_shell.pause(toggle)
        self._paused = toggle

    @property
    def shell(self) -> MOSShell:
        self._check_running()
        return self._ctml_shell

    @property
    def matrix(self) -> Matrix:
        return self._matrix

    def _bootstrap_after_matrix(self) -> None:
        # ctml_shell 起来后, 补 static slot (dynamic leaf: ctml_shell.static_messages
        # 是 callable, 每次读取时动态计算); 注册 SystemPrompter / MossSystemPrompter
        # 两个 IoC key 指向同一实例 (钻石继承).
        self._matrix.container.set(MOSShell, self._ctml_shell)
        self._matrix.container.set(CTMLShell, self._ctml_shell)
        self._system_prompter.with_prompter(
            MossSystemPrompter.MOSS_STATIC_SLOT,
            self._ctml_shell.static_messages,
        )
        self._matrix.container.set(SystemPrompter, self._system_prompter)
        self._matrix.container.set(MossSystemPrompter, self._system_prompter)

    @contextlib.asynccontextmanager
    async def _manager_shell_lifecycle(self):
        if self._run_shell_on_start:
            await self._ctml_shell.__aenter__()
            # kick off first round refresh_meta
            await self._ctml_shell.refresh_metas(0.5)
        try:
            yield
        finally:
            if self._ctml_shell.is_running():
                await self._ctml_shell.__aexit__(None, None, None)

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError('MossRuntime is already started')
        self._started = True
        await self._async_exit_stack.__aenter__()
        # 启动 matrix
        await self._async_exit_stack.enter_async_context(self._matrix)
        # 补 IoC 注册 (system prompter / MOSShell) — 之前挂 _app_store 的位置
        self._bootstrap_after_matrix()
        # 启动 ctml shell
        await self._async_exit_stack.enter_async_context(self._manager_shell_lifecycle())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 进入即标记 closing, 通知依赖方提前结束运行时逻辑.
        self._closing_event.set()
        self._matrix.close()
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._matrix.logger.exception("%s failed to aexit %s", self._log_prefix, e)
            raise e
        finally:
            self._closed_event.set()
