from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union, Callable, Coroutine, List, Type, TypeVar, Dict, ClassVar, Any, AsyncIterable
from typing_extensions import Self
from .command import Command, CommandCall, CommandType, CommandMeta, CommandTask
from .states import StateStore, State
from ghoshell_container import IoCContainer, INSTANCE, Provider, BINDING
from ghoshell_common.helpers import generate_import_path
from pydantic import BaseModel, Field

FunctionCommand = Callable[..., Coroutine]
PolicyCommand = Callable[..., Coroutine[None]]

R = TypeVar('R')


class ChannelMeta(BaseModel):
    name: str = Field(description="The name of the channel.")
    available: bool = Field(description="Whether the channel is available.")
    description: str = Field(description="The description of the channel.")
    stats: List[State] = Field(default_factory=list, description="The list of state objects.")
    commands: List[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    children: List[str] = Field(default_factory=list, description="the children channel names")


#
# class ChannelRuntime(ABC):
#
#     #
#     #     # --- commands --- #
#     #
#     @abstractmethod
#     def append(self, *commands: CommandTask) -> None:
#         pass
#
#     @abstractmethod
#     def prepend(self, *commands: CommandTask) -> None:
#         pass
#
#     # --- control --- #
#     @abstractmethod
#     def clear(self) -> None:
#         """
#         clear the channel's commands, include executing command and pending command.
#         after clear, the channel will rerun the policy.
#         """
#         pass
#
#     #
#     #     @abstractmethod
#     #     def cancel(self) -> bool:
#     #         """
#     #         cancel the running command task
#     #         """
#     #         pass
#     #
#     @abstractmethod
#     def defer_clear(self) -> None:
#         """
#         clear when any new command is pushed into this channel
#         """
#         pass


#
#     @abstractmethod
#     def reset(self) -> None:
#         """
#         返回初始状态. 包括 policy 也会重置回原始状态.
#         """
#         pass
#
#     # --- status --- #
#     @abstractmethod
#     def is_idle(self) -> bool:
#         pass
#
#     @abstractmethod
#     def wait_until_idle(self, timeout: float | None = None) -> None:
#         pass
#
#     # --- call --- #
#     @abstractmethod
#     def new_task(self, name: str, *args, **kwargs) -> CommandTask:
#         pass
#
#     @abstractmethod
#     def get_command_metas(self, types: Optional[CommandType] = None) -> Iterable[CommandMeta]:
#         pass
#
#     @abstractmethod
#     def get_commands(self, types: Optional[CommandType] = None) -> Iterable[Command]:
#         pass

class ChannelRuntime(ABC):

    @abstractmethod
    def metas(self) -> List[ChannelMeta]:
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        提供依赖注入容器, 可以在被 command 标记的函数中用.
        """
        pass

    @abstractmethod
    def available(self) -> bool:
        """
        channel is available.
        """
        pass

    @abstractmethod
    def set_available(self, available: bool):
        pass

    @property
    @abstractmethod
    def states(self) -> StateStore:
        """
        the states store
        """
        pass

    @abstractmethod
    def commands(self) -> Iterable[Command]:
        pass

    @abstractmethod
    def get_command(self, name: str) -> Optional[Command]:
        pass

    @abstractmethod
    async def make_prompt(self, pml: str) -> str:
        """
        基于 prompt 函数, 和 prompt marked language 生成一份 prompt.
        """
        pass


class Channel(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        pass

    # --- children --- #

    @abstractmethod
    def with_children(self, *children: "Channel") -> Self:
        pass

    @abstractmethod
    def children(self) -> Dict[str, "Channel"]:
        """
        register children channel.
        """
        pass

    @abstractmethod
    def descendants(self) -> Dict[str, "Channel"]:
        pass

    @abstractmethod
    def get_child(self, name: str) -> Optional[Self]:
        pass

    # --- decorators --- #

    @abstractmethod
    def with_description(self, callback: Callable[..., str]) -> Callable[..., str]:
        pass

    @abstractmethod
    def with_function(
            self,
            *,
            name: str = "",
            doc: Optional[Callable[..., str] | str] = None,
            interface: Optional[str | Callable[..., str]] = None,
            to_thread: bool = False,
            tags: Optional[List[str]] = None,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        """
        wrap an async function
        """
        pass

    @abstractmethod
    def with_policy(
            self,
            *,
            name: str = "",
            doc: Optional[Union[Callable[..., str] | str]] = None,
            interface: Optional[str] = None,
            tags: Optional[List[str]] = None,
    ) -> Callable[[PolicyCommand], PolicyCommand]:
        """
        register policy functions
        """
        pass

    @abstractmethod
    def with_providers(self, *providers: Provider) -> None:
        """
        register default providers for the contracts
        """
        pass

    @abstractmethod
    def with_binding(self, contract: Type[INSTANCE], binding: Optional[BINDING] = None) -> None:
        """
        register default bindings for the given contract.
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    async def start(self, container: Optional[IoCContainer] = None) -> Self:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass
