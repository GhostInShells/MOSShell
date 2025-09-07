from __future__ import annotations
from typing import Callable, Coroutine, Optional, Iterable
from ghoshell_moss.concepts.command import Command, CommandTask, CommandMeta, RESULT, CommandType
from ghoshell_moss.concepts.channel import Channel, ChannelRuntime, ChannelMeta
from ghoshell_container import set_container
import anyio


class PyCommand(Command[RESULT]):

    def __init__(self, meta: CommandMeta, func: Callable[..., Coroutine[RESULT]]):
        self._meta = meta
        self._func = func

    def meta(self) -> CommandMeta:
        return self._meta

    def __prompt__(self) -> str:
        return self._meta.interface

    async def __call__(self, *args, **kwargs) -> RESULT:
        return await self._func(*args, **kwargs)


class PyChannelRuntime(ChannelRuntime):

    def __init__(
            self,
    ):
        self._task_stream: anyio.create_memory_object_stream()

    def meta(self) -> ChannelMeta:
        pass

    def append(self, *commands: CommandTask) -> None:
        pass

    def prepend(self, *commands: CommandTask) -> None:
        pass

    def clear(self) -> None:
        pass

    def cancel(self) -> bool:
        pass

    def defer_clear(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def is_idle(self) -> bool:
        pass

    def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    def new_task(self, name: str, *args, **kwargs) -> CommandTask:
        pass

    def get_command_metas(self, types: Optional[CommandType] = None) -> Iterable[CommandMeta]:
        pass

    def get_commands(self, types: Optional[CommandType] = None) -> Iterable[Command]:
        pass