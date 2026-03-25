from abc import ABC, abstractmethod

from pydantic_ai import ToolReturn
from typing_extensions import Self

from ghoshell_moss.core.concepts.moss import (
    MOSS, MOSSRuntime, IdleHook, RespondHook, MOSSToolSet, PriorityLevel,
    IgnorePolicy, Snapshot,
)
from ghoshell_moss.core.concepts.speech import Speech
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.core.ctml.v1_0_0.prompts import (
    make_interfaces,
    make_context_messages,
    make_instruction_messages,
)
from ghoshell_container import IoCContainer, Container
from ghoshell_common.contracts import LoggerItf
import logging


class BaseMOSSToolset(MOSSToolSet):

    def __init__(self, runtime: MOSSRuntime):
        self._main_runtime = runtime
        self._entered = False
        self._exited = False

    def meta_instruction(self) -> str:
        return self._main_runtime.shell.meta_instruction()

    @property
    def runtime(self) -> MOSSRuntime:
        return self._main_runtime

    async def moss_instructions(self) -> str:
        pass

    async def moss_context_messages(self) -> ToolReturn:
        pass

    async def moss_add(self, commands: str) -> ToolReturn:
        pass

    async def moss_call_soon(self, commands: str) -> ToolReturn:
        await self._main_runtime.call_soon(commands)
        snapshot = await self._main_runtime.pop_snapshot()
        return self._snapshot_to_tool_return(snapshot)

    async def moss_interrupt(self) -> ToolReturn:
        await self._main_runtime.interrupt()
        snapshot = await self._main_runtime.pop_snapshot()
        return self._snapshot_to_tool_return(snapshot)

    async def moss_observe(self, timeout: float | None = None) -> ToolReturn:
        await self._main_runtime.observe(timeout)
        snapshot = await self._main_runtime.pop_snapshot()
        return self._snapshot_to_tool_return(snapshot)

    async def moss_focus(self, level: PriorityLevel, policy: IgnorePolicy = 'buffer') -> None:
        await self._main_runtime.focus(level, policy)

    @staticmethod
    def _snapshot_to_tool_return(
            snapshot: Snapshot,
    ) -> ToolReturn:
        return ToolReturn(
            return_value=None,
            content=list(snapshot.to_user_contents(with_meta=True)),
        )

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError('MOSS is already entered')
        self._entered = True
        await self._main_runtime.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exited:
            raise RuntimeError('MOSS is already exited')
        self._exited = True
        await self._main_runtime.__aexit__(exc_type, exc_val, exc_tb)


class BaseMOSSImpl(MOSS, ABC):

    def __init__(
            self,
            *,
            name: str = "MOSS",
            container: IoCContainer | None = None,
            description: str = '',
            logger: LoggerItf | None = None,
            speech: Speech = None,
            primitives: list[str] | None = None,
    ):
        self._name = name
        self._container = container or Container(name=name)
        self._shell = new_ctml_shell(
            name=name,
            container=self._container,
            description=description,
            speech=speech,
            logger=logger,
            primitives=primitives,
        )
        self._respond_hooks: list[RespondHook] = []
        self._idle_hooks: list[IdleHook] = []

    @property
    def container(self) -> IoCContainer:
        return self._container

    def run(self) -> MOSSRuntime:
        pass

    def run_as_toolset(self) -> MOSSToolSet:
        runtime = self.run()
        return BaseMOSSToolset(runtime)

    @property
    def shell(self) -> MOSShell:
        return self._shell

    def on_respond(self, hook: RespondHook) -> Self:
        self._respond_hooks.append(hook)
        return self

    def on_idle(self, hook: IdleHook) -> Self:
        self._idle_hooks.append(hook)
        return self
