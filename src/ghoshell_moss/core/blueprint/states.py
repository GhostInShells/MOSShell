from abc import ABC, abstractmethod
from typing_extensions import Self
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.builder import Builder

__all__ = [
    'ChannelState', 'ChannelStateBuilder', 'StatefulChannel',
    'new_state_builder', 'new_channel_from_state', 'new_stateful_channel',
]

"""

"""

_ChannelName = str

__description__ = "How to build stateful channel"

class ChannelState(ABC):
    """
    Channel 的运行时状态, 用来快速构建一个 StateChannel.
    """

    @abstractmethod
    def name(self) -> str:
        """
        return name of the state
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        return description of the state
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        if the state is available
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> bool:
        """
        if the state is dynamic, need to refresh each time.
        """
        pass

    @abstractmethod
    async def get_instruction(self) -> str:
        """
        return instruction provided by the state
        """
        pass

    @abstractmethod
    async def get_context_messages(self) -> list[Message]:
        """
        return the context messages from the state.
        """
        pass

    @abstractmethod
    async def on_startup(self) -> None:
        """
        when channel startup.
        """
        pass

    @abstractmethod
    async def on_close(self) -> None:
        """
        when channel close.
        """
        pass

    @abstractmethod
    async def on_running(self) -> None:
        """
        when channel is running.
        """
        pass

    @abstractmethod
    async def on_idle(self) -> None:
        """
        when channel is idle, all the commands are done and the children are idle as well
        """
        pass

    @abstractmethod
    def own_commands(self) -> dict[str, Command]:
        """
        return the commands mapping by name
        """
        pass

    @abstractmethod
    def get_own_command(self, name: str) -> Command | None:
        """
        get a command by name
        """
        pass

    @abstractmethod
    def bootstrap(self, container: IoCContainer) -> None:
        """
        register something to the container. or get some contracts from it.
        """
        pass

    @abstractmethod
    def get_children(self) -> dict[_ChannelName, Channel]:
        """
        return the sustain children channel
        """
        pass

    @abstractmethod
    def get_virtual_children(self) -> dict[_ChannelName, Channel]:
        """
        return the virtual children that may be changed during runtime
        """
        pass


class ChannelStateBuilder(Builder, ChannelState, ABC):
    """
    Channel State which is mutable.
    """

    @abstractmethod
    def add_virtual_channel(self, channel: Channel, alias: _ChannelName | None = None) -> None:
        """
        add virtual channel during runtime.
        wrap this method into a command
        """
        pass

    @abstractmethod
    def remove_virtual_channel(self, name: str) -> None:
        """
        remove virtual channel during runtime.
        wrap this method into a command
        """
        pass


def new_state_builder(name: str, description: str = "") -> ChannelStateBuilder:
    """
    new state builder
    """
    from ghoshell_moss.core.py_channel import PyChannelBuilder
    return PyChannelBuilder(name=name, description=description)


class StatefulChannel(Channel, ABC):

    @abstractmethod
    def main_state(self) -> ChannelState:
        """
        return the main state of the channel
        """
        pass

    @abstractmethod
    def new_state(self, name: str, description: str) -> ChannelStateBuilder:
        """
        create new substate of the channel
        """
        pass

    @abstractmethod
    def states(self) -> dict[str, ChannelState]:
        """
        return the switchable states, without main states.
        """
        pass

    @abstractmethod
    def with_state(self, state: ChannelState, alias: str | None = None) -> Self:
        """
        register a named substate to the channel.
        """
        pass


def new_channel_from_state(state: ChannelState) -> StatefulChannel:
    """
    create new channel by state object
    """
    from ghoshell_moss.core.py_channel import BaseStateChannel
    return BaseStateChannel(state)


def new_stateful_channel(name: str, description: str = "") -> StatefulChannel:
    """
    create new stateful channel with builders.
    """
    from ghoshell_moss.core.py_channel import PyChannel
    return PyChannel(name=name, description=description)
