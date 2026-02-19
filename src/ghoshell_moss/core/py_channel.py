import asyncio
import contextvars
import inspect
from typing import Optional, Callable

from ghoshell_container import BINDING, INSTANCE, Container, IoCContainer
from typing_extensions import Self

from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import (
    Builder,
    Channel,
    MutableChannel,
    ChannelBroker,
    ChannelMeta,
    CommandFunction,
    MessageFunction,
    LifecycleFunction,
    ChannelCtx,
    StringType,
    ChannelPaths,
)
from ghoshell_moss.core.concepts.broker import AbsChannelTreeBroker
from ghoshell_moss.core.concepts.command import Command, PyCommand, CommandWrapper
from ghoshell_moss.core.concepts.states import BaseStateStore, StateModel, StateStore
from ghoshell_common.helpers import uuid

__all__ = ["PyChannel", "PyChannelBroker", "PyChannelBuilder"]


class PyChannelBuilder(Builder):
    def __init__(self, name: str, blocking: bool):
        self._name = name
        self._blocking = blocking
        self._description_fn: Optional[StringType] = None
        self._available_fn: Optional[Callable[[], bool]] = None
        self._on_idle_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_start_up_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_stop_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_running_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_pause_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._context_messages_function: Optional[MessageFunction] = None
        self._instruction_messages_function: Optional[MessageFunction] = None
        self._state_models: list[StateModel] = []
        self._commands: dict[str, Command] = {}
        self._container_instances = {}
        self._dynamic = False

    def description(self) -> Callable[[StringType], StringType]:
        """
        todo: 移除这个函数.
        """

        def wrapper(func: StringType) -> StringType:
            self._dynamic = True
            self._description_fn = func
            return func

        return wrapper

    def is_dynamic(self) -> bool:
        return self._dynamic

    def available(self) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        def wrapper(func: Callable[[], bool]) -> Callable[[], bool]:
            self._dynamic = True
            self._available_fn = func
            return func

        return wrapper

    def is_available(self) -> bool:
        if self._available_fn is not None:
            return self._available_fn()
        return True

    def state_model(self, model: type[StateModel] | StateModel) -> type[StateModel] | StateModel:
        saving = model
        if isinstance(model, type):
            saving = model()
        self._state_models.append(saving)
        return model

    def get_states(self, owner: str, parent: StateStore | None = None) -> StateStore:
        store = BaseStateStore(owner=owner, parent=parent)
        store.register(*self._state_models)
        return store

    def context_messages(self, func: MessageFunction) -> MessageFunction:
        self._context_messages_function = func
        self._dynamic = True
        return func

    async def get_context_message(self) -> list[Message]:
        if self._context_messages_function is None:
            return []
        if inspect.iscoroutinefunction(self._context_messages_function):
            return await self._context_messages_function()
        return self._context_messages_function()

    def instruction_messages(self, func: MessageFunction) -> MessageFunction:
        self._instruction_messages_function = func
        self._dynamic = True
        return func

    async def get_instruction_messages(self) -> list[Message]:
        if self._instruction_messages_function is None:
            return []
        if inspect.iscoroutinefunction(self._instruction_messages_function):
            return await self._instruction_messages_function()
        return self._instruction_messages_function()

    def command(
            self,
            *,
            name: str = "",
            chan: str | None = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType] = None,
            available: Optional[Callable[[], bool]] = None,
            blocking: Optional[bool] = None,
            call_soon: bool = False,
            return_command: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:

        def wrapper(func: CommandFunction) -> CommandFunction:
            command = PyCommand(
                func,
                name=name,
                chan=chan if chan is not None else self._name,
                doc=doc,
                comments=comments,
                tags=tags,
                interface=interface,
                available=available,
                blocking=blocking if blocking is not None else self._blocking,
                call_soon=call_soon,
            )
            self._commands[command.name()] = command
            if return_command:
                return command
            return func

        return wrapper

    def commands(self) -> list[Command]:
        return list(self._commands.values())

    def get_command(self, name: str) -> Command | None:
        return self._commands.get(name)

    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_idle_funcs.append((func, is_coroutine))
        return func

    async def on_idle(self):
        await self._run_funcs(self._on_idle_funcs)

    def start_up(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_start_up_funcs.append((func, is_coroutine))
        return func

    async def on_start_up(self) -> None:
        await self._run_funcs(self._on_start_up_funcs)

    def close(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_stop_funcs.append((func, is_coroutine))
        return func

    @classmethod
    async def _run_funcs(cls, funcs: list[tuple[LifecycleFunction, bool]]) -> None:
        if len(funcs) == 0:
            return

        tasks = []
        for func, is_coroutine in funcs:
            if is_coroutine:
                cor = func()
            else:
                cor = asyncio.to_thread(func)
            t = asyncio.create_task(cor)
            tasks.append(t)
        await asyncio.gather(*tasks)

    async def on_close(self) -> None:
        await self._run_funcs(self._on_stop_funcs)

    def running(self, running_func: LifecycleFunction) -> LifecycleFunction:
        self._on_running_funcs.append((running_func, inspect.iscoroutinefunction(running_func)))
        return running_func

    def pause(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_pause_funcs.append((func, is_coroutine))
        return func

    async def on_pause(self) -> None:
        await self._run_funcs(self._on_pause_funcs)

    async def on_running(self) -> None:
        await self._run_funcs(self._on_running_funcs)

    def with_binding(self, contract: type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        self._container_instances[contract] = binding
        return self

    def update_container(self, container: IoCContainer) -> None:
        for contract, instance in self._container_instances.items():
            container.set(contract, instance)


class PyChannel(MutableChannel):
    def __init__(
            self,
            *,
            name: str,
            description: str = "",
            # todo: block 还是叫 blocking 吧.
            blocking: bool = True,
            dynamic: bool | None = None,
    ):
        """
        :param name: channel 的名称.
        :param description: channel 的静态描述, 给模型看的.
        :param blocking: channel 里默认的 command 类型, 是阻塞的还是非阻塞的.
        :param dynamic: 这个 channel 对大模型而言是否是动态的.
                        如果是动态的, 大模型每一帧思考时, 都会从 channel 获取最新的状态.
        """
        self._name = name
        self._id = uuid()
        self._description = description
        self._broker: Optional[ChannelBroker] = None
        self._children: dict[str, Channel] = {}
        self._block = blocking
        self._dynamic = dynamic
        # decorators
        self._builder = PyChannelBuilder(
            name=name,
            blocking=blocking,
        )

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    @property
    def build(self) -> Builder:
        return self._builder

    @property
    def broker(self) -> ChannelBroker | None:
        return self._broker

    def import_channels(self, *children: "Channel") -> Self:
        for child in children:
            self._children[child.name()] = child
        return self

    def new_child(
            self,
            name: str,
            description: str = "",
            blocking: bool = True,
    ) -> Self:
        """
        语法糖, 用来做单元测试.
        """
        child = PyChannel(name=name, description=description, blocking=blocking)
        self._children[name] = child
        return child

    def children(self) -> dict[str, "Channel"]:
        return self._children

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelBroker":
        if self._broker is not None and self._broker.is_running():
            raise RuntimeError("Server already running")
        self._broker = PyChannelBroker(
            channel=self,
            container=container,
            dynamic=self._dynamic,
        )
        return self._broker

    def is_running(self) -> bool:
        return self._broker is not None and self._broker.is_running()


class PyChannelBroker(AbsChannelTreeBroker):
    def __init__(
            self,
            channel: PyChannel,
            container: Optional[IoCContainer] = None,
            *,
            dynamic: bool | None = None,
    ):
        super().__init__(channel=channel, container=container)
        self._builder = channel.build
        self._dynamic = dynamic

    def is_connected(self) -> bool:
        # always true
        return True

    async def wait_connected(self) -> None:
        # always ready
        return

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Channel {self} not running")

    def children(self) -> dict[str, Channel]:
        result = self._channel.children()
        return result

    async def _generate_self_meta(self) -> ChannelMeta:
        dynamic = self._dynamic or False
        command_metas = []
        commands = self._builder.commands()

        refreshing_commands = []
        refreshing_command_tasks = []
        for command in commands:
            # 只添加需要动态更新的 command.
            if command.meta().dynamic:
                refreshing_commands.append(command)
                refreshing_command_tasks.append(command.refresh_meta())
                dynamic = True

        # 更新所有的 动态 commands.
        if len(refreshing_commands) > 0:
            done = await asyncio.gather(*refreshing_command_tasks, return_exceptions=True)
            idx = 0
            for refreshed in done:
                if isinstance(refreshed, Exception):
                    command = commands[idx]
                    self.logger.exception("Refresh command meta failed on command %s", command)
                idx += 1

        for command in commands:
            command_metas.append(command.meta())

        name = self._name
        instruction_message_task = asyncio.create_task(self._builder.get_instruction_messages())
        context_message_task = asyncio.create_task(self._builder.get_context_message())
        new_context_messages = await context_message_task
        new_instruction_messages = await instruction_message_task

        meta = ChannelMeta(
            name=name,
            channel_id=self.id,
            available=self._builder.is_available(),
            description=self.channel.description(),
            context=new_context_messages,
            instructions=new_instruction_messages,
        )
        meta.dynamic = dynamic
        meta.commands = command_metas
        return meta

    # ---- commands ---- #

    def _is_available(self) -> bool:
        return self._builder.is_available()

    def self_commands(self, available_only: bool = True) -> dict[str, Command]:
        if not self.is_available():
            return {}
        result = {}
        for command in self._builder.commands():
            if not available_only or command.is_available():
                result[command.name()] = self._wrap_origin_command(command)
        return result

    def _wrap_origin_command(self, command: Command | None) -> Command | None:
        """
        确保函数被单独调用时也拥有自己的 ctx
        """
        if command is None:
            return None
        ctx = contextvars.copy_context()
        ChannelCtx.init(self)
        return CommandWrapper.wrap(command, ctx)

    def get_self_command(
            self,
            name: str,
    ) -> Optional[Command]:
        return self._wrap_origin_command(self._builder.get_command(name))

    async def on_running(self) -> None:
        await self._builder.on_running()

    async def on_idle(self) -> None:
        try:
            if not self.is_running():
                return
            await self._builder.on_idle()

        except asyncio.CancelledError:
            self.logger.info(f"{self.log_prefix} on_idle done")
            return
        except Exception as e:
            self.logger.exception(e)
            raise

    async def on_start_up(self) -> None:
        # 准备 start up 的运行.
        await self._builder.on_start_up()

    async def on_close(self) -> None:
        await self._builder.on_close()

    def prepare_container(self, container: IoCContainer | None) -> IoCContainer:
        self._builder.update_container(container)
        container = super().prepare_container(container)
        return container
