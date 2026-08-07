import asyncio
import inspect
import logging
from typing import Optional, Callable, Iterable

from ghoshell_container import BINDING, INSTANCE, IoCContainer, Provider, provide
from typing_extensions import Self

from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import (
    Channel,
    ChannelMeta,
    ChannelNamePattern,
    ChannelName,
    ChannelCtx,
)
from ghoshell_moss.core.runtime import AbsChannelTreeRuntime
from ghoshell_moss.core.concepts.errors import CommandError
from ghoshell_moss.message import unique_id
from ghoshell_common.contracts import LoggerItf
from PIL.Image import Image
from ghoshell_moss.core.concepts.command import Command, PyCommand, CommandWrapper, CommandUniqueName
from ghoshell_moss.core.blueprint.states_channel import (
    MutableChannelState, ChannelState, StatefulChannel, PrimeChannel,
    StatefulChannelRuntime,
)
from ghoshell_moss.core.blueprint.channel_builder import (
    CommandFunction,
    MessageFunction,
    MessageType,
    LifecycleFunction,
    StringType,
    ChannelFactory,
)
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
import time
import re

__all__ = ["PyChannel", "StatefulChannelRuntimeImpl", "PyChannelBuilder", "BaseStateChannel"]

_ChannelNamePattern = re.compile(ChannelNamePattern)
_ChannelName = str


class PyChannelBuilder(MutableChannelState, ChannelState):

    def __init__(self, name: str, blocking: bool = True, description: str = "", uid: str = '') -> None:
        self._name = name
        self._description = description
        self._uid = uid or unique_id()
        self._blocking = blocking
        self._available_fn: Optional[Callable[[], bool]] = None
        self._on_idle_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_start_up_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_stop_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_running_funcs: list[tuple[LifecycleFunction, bool]] = []
        self._on_refresh_meta_funcs: list[tuple[LifecycleFunction, bool]] = []

        self._context_messages_functions: list[MessageFunction] = []
        self._instruction_functions: StringType | None = None
        self._sustain_children: dict[str, Channel | ChannelFactory] = {}
        self._sustain_children_factories: list[Callable] = []
        self._virtual_children: dict[str, Channel] = {}
        self._providers: list[tuple[Provider, bool]] = []

        self._commands: dict[str, Command] = {}
        self._container_instances = {}
        self._dynamic = False
        self._logger = logging.getLogger("moss")

    def copy(self) -> Self:
        from copy import copy
        copied = copy(self)
        update = {}
        for k, v in self.__dict__.items():
            if k.startswith("_") and not k.startswith("__"):
                if isinstance(v, dict) or isinstance(v, list):
                    update[k] = v.copy()
        copied.__dict__.update(update)
        return copied

    def id(self) -> str:
        return self._uid

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        """
        返回 state 的 description.
        """
        return self._description

    def with_logger(self, logger: LoggerItf) -> None:
        self._logger = logger

    def is_dynamic(self) -> bool:
        return self._dynamic or len(self._virtual_children) > 0

    def available(self, func: Callable[[], bool]) -> Callable[[], bool]:
        self._dynamic = True
        self._available_fn = func
        return func

    def is_available(self) -> bool:
        if self._available_fn is not None:
            return self._available_fn()
        return True

    def context_messages(self, func: MessageFunction, reset: bool = False) -> MessageFunction:
        if reset:
            self._context_messages_functions.clear()
        self._context_messages_functions.append(func)
        self._dynamic = True
        return func

    async def get_context_messages(self) -> list[Message]:
        """
        使用所有的 context messages 函数生成
        """
        if not self._context_messages_functions:
            return []
        message_cor = []
        for func in self._context_messages_functions:
            if inspect.iscoroutinefunction(func):
                message_cor.append(func())
            else:
                message_cor.append(asyncio.to_thread(func))
        messages = []
        # 并发生成 messages.
        if len(message_cor) > 0:
            done = await asyncio.gather(*message_cor, return_exceptions=True)
            for result in done:
                if isinstance(result, Exception):
                    self._logger.error(
                        'refresh channel %s failed with message func error: %s',
                        self._name, result,
                    )
                    continue
                context_messages = result
                messages.extend(context_messages)
        return self._wrap_messages(messages)

    @staticmethod
    def _wrap_messages(messages: list[MessageType]):
        last = None
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                if last is not None:
                    result.append(last)
                last = msg
            else:
                if last is not None:
                    last.with_content(msg)
                else:
                    last = Message.new().with_content(msg)
        if last is not None:
            result.append(last)
        return result

    def instruction(self, func: StringType) -> StringType:
        self._instruction_functions = func
        if callable(func):
            self._dynamic = True
        return func

    async def get_instruction(self) -> str:
        if self._instruction_functions is None:
            return ''
        if isinstance(self._instruction_functions, str):
            return self._instruction_functions
        if inspect.iscoroutinefunction(self._instruction_functions):
            return await self._instruction_functions()
        return self._instruction_functions()

    def add_command(
            self,
            command: Command,
            *,
            override: bool = True,
            name: Optional[str] = None,
    ) -> None:
        if not isinstance(command, Command):
            raise ValueError("Command must be of type Command, not {}".format(type(command)))
        name = name or command.name()
        if override or name not in self._commands:
            self._commands[command.name()] = command
            if command.is_dynamic():
                self._dynamic = True

    def command(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType] = None,
            available: Optional[Callable[[], bool]] = None,
            override: bool = True,
            blocking: Optional[bool] = None,
            priority: int = 0,
            call_soon: bool = False,
            return_command: bool = False,
            always_observe: bool = False,
            timeout: Optional[float] = None,
            visible: bool = True,
            macro: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:

        def wrapper(func: CommandFunction) -> CommandFunction:
            command = PyCommand(
                func,
                name=name,
                chan=self._name,
                doc=doc,
                comments=comments,
                tags=tags,
                interface=interface,
                available=available,
                blocking=blocking if blocking is not None else self._blocking,
                priority=priority,
                call_soon=call_soon,
                always_observe=always_observe,
                timeout=timeout,
                visible=visible,
                macro=macro,
            )
            self.add_command(command, override=override)
            if return_command:
                return command
            return func

        return wrapper

    def add_virtual_channel(self, channel: Channel, alias: ChannelName | None = None) -> None:
        name = alias or channel.name()
        self._virtual_children[name] = channel

    def remove_virtual_channel(self, name: str) -> None:
        if name in self._virtual_children:
            self._virtual_children.pop(name)

    def with_contract_factory(
            self,
            contract: type[INSTANCE],
            factory: Callable[[...], INSTANCE],
            *,
            singleton: bool = True,
            override: bool = False,
    ) -> Self:
        provider = provide(contract, singleton)(factory)
        self._providers.append((provider, override))
        return self

    def import_channels(
            self,
            *children: Channel | ChannelFactory | tuple[Channel | ChannelFactory, _ChannelName],
    ) -> Self:
        for value in children:
            if isinstance(value, tuple):
                channel, name = value
            elif callable(value):
                self._sustain_children_factories.append(value)
                continue
            else:
                channel = value
                name = channel.name()
            self._sustain_children[name] = channel
        return self

    def get_children(self) -> dict[_ChannelName, Channel]:
        return self._sustain_children

    def get_virtual_children(self) -> dict[_ChannelName, Channel]:
        return self._virtual_children

    def own_commands(self) -> dict[str, Command]:
        return self._commands

    def get_own_command(self, name: str) -> Command | None:
        return self._commands.get(name)

    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_idle_funcs.append((func, is_coroutine))
        return func

    async def on_idle(self):
        await self._run_funcs(self._on_idle_funcs)

    def startup(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_start_up_funcs.append((func, is_coroutine))
        return func

    async def on_startup(self) -> None:
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

    async def on_running(self) -> None:
        await self._run_funcs(self._on_running_funcs)

    def refresh_meta(self, func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(func)
        self._on_refresh_meta_funcs.append((func, is_coroutine))
        self._dynamic = True
        return func

    async def on_refresh_meta(self) -> None:
        await self._run_funcs(self._on_refresh_meta_funcs)

    def with_binding(self, contract: type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        self._container_instances[contract] = binding
        return self

    def bootstrap(self, container: IoCContainer) -> None:
        if len(self._container_instances) > 0:
            for contract, instance in self._container_instances.items():
                container.set(contract, instance)
        if len(self._providers) > 0:
            for provider, override in self._providers:
                if override or not container.bound(provider.contract(), recursively=True):
                    container.register(provider)
        # 支持 ChannelFactory 的实现方式.
        channel_from_factory = {}
        for name, channel in self._sustain_children.items():
            if callable(channel):
                # create channel.
                channel_from_factory[name] = channel(container)
        for channel_factory in self._sustain_children_factories:
            channel = channel_factory(container)
            if isinstance(channel, Channel):
                channel_from_factory[channel.name()] = channel

        if channel_from_factory:
            # 反向更新.
            self._sustain_children.update(channel_from_factory)
        validated_sustain_children = {}
        for name, channel in self._sustain_children.items():
            if isinstance(channel, Channel):
                validated_sustain_children[name] = channel

        # 保证没有错误行为.
        self._sustain_children = validated_sustain_children

    def to_channel(self, uid: str | None = None) -> Channel:
        return BaseStateChannel(self, uid=uid)


class BaseStateChannel(StatefulChannel):

    def __init__(
            self,
            main: ChannelState,
            uid: str | None = None,
            states: dict[str, ChannelState] | None = None,
            modules: dict[str, ChannelModule] | None = None,
            default_state_name: str = '',
            bootstrap_callbacks: list[Callable[[Self, IoCContainer], None]] | None = None,
    ) -> None:
        self._uid = uid or unique_id()
        self._main: ChannelState = main
        self._states: dict[str, ChannelState] = states or {}
        self._default_state_name: str = default_state_name
        self._modules: dict[str, ChannelModule] = modules or {}
        self._boostrap_callbacks: list[Callable[[Self, IoCContainer], None]] = bootstrap_callbacks or []

    def on_bootstrap(self, bootstrapper: Callable[[StatefulChannel, IoCContainer], None]) -> None:
        self._boostrap_callbacks.append(bootstrapper)

    def copy(self) -> Self:
        from copy import copy
        copied = copy(self)
        # main 不 copy, 因为就是表示是有副作用的.
        copied._main = copied._main
        copied._states = {k: v.copy() for k, v in copied._states.items()}
        copied._modules = copied._modules.copy()
        copied._boostrap_callbacks = copied._boostrap_callbacks.copy()
        return copied

    def main_state(self) -> ChannelState:
        return self._main

    def new_state(self, name: str, description: str) -> MutableChannelState:
        new_state = PyChannelBuilder(name=name, description=description)
        self._states[name] = new_state
        return new_state

    def states(self) -> dict[str, ChannelState]:
        return self._states

    def with_state(self, state: ChannelState, alias: str | None = None, is_default: bool = False) -> Self:
        """注册为可切换的模式。同一时刻只有一个 state 激活，通过 switch_state() 切换。"""
        name = alias or state.name()
        self._states[name] = state
        if is_default:
            self._default_state_name = name
        return self

    def default_state_name(self) -> str:
        return self._default_state_name

    def with_module(self, module: ChannelModule) -> Self:
        """注册为永久能力模块。所有 module 同时激活、累积叠加 — 与 with_state() 的排他切换正交。"""
        self._modules[module.name()] = module
        return self

    def modules(self) -> dict[str, ChannelModule]:
        """返回所有已注册的永久能力模块。"""
        return self._modules

    def children(self) -> dict[_ChannelName, Channel]:
        return self._main.get_children()

    def virtual_children(self) -> dict[_ChannelName, Channel]:
        return self._main.get_virtual_children()

    def name(self) -> str:
        return self._main.name()

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._main.description()

    def materialize(self, container: IoCContainer) -> "StatefulChannelRuntimeImpl":
        # 所有 state 的启动都是在 StatefulChannelRuntime 中.
        if len(self._boostrap_callbacks) > 0:
            container = container or StatefulChannelRuntimeImpl.create_raw_container(
                self.name(),
                self.id(),
            )
            for callback in self._boostrap_callbacks:
                callback(self, container)

        return StatefulChannelRuntimeImpl(self, container=container)


class PyChannel(PrimeChannel, BaseStateChannel):
    """
    一个 Prime Channel.
    """

    def __init__(
            self,
            *,
            name: str,
            description: str = "",
            blocking: bool = True,
            uid: str | None = None,
    ):
        """
        :param name: channel 的名称.
        :param description: channel 的静态描述, 给模型看的.
        :param blocking: 默认所有 command 序列执行 (blocking=True)。此参数是设计不佳的语法糖——阻塞语义应由 command 自身声明，而非 channel 统一施加。未来版本应移除。
        """
        matched = _ChannelNamePattern.fullmatch(name)
        if matched is None:
            raise ValueError("Channel name '%s' is not valid" % name)
        state = PyChannelBuilder(name=name, description=description, blocking=blocking, uid=uid)
        super().__init__(state, uid=uid)
        self._builder = state

    @property
    def build(self) -> PyChannelBuilder:
        return self._builder

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
        self.build.import_channels(child)
        return child


class StatefulChannelRuntimeImpl(StatefulChannelRuntime, AbsChannelTreeRuntime[StatefulChannel]):
    """
    实现标准的, 支持各种 State 的 ChannelRuntime.
    """

    def __init__(
            self,
            channel: StatefulChannel,
            container: Optional[IoCContainer] = None,
    ):

        self._main_state = channel.main_state()
        self._dynamic_states = channel.states()
        self._modules: dict[str, ChannelModule] = channel.modules()
        self._static_meta_cache: Optional[ChannelMeta] = None
        self._current_state: ChannelState | None = None
        self._current_state_name: str | None = None
        self._current_state_running_task: asyncio.Task | None = None
        self._default_state_name: str = channel.default_state_name()
        self._switch_state_command = PyCommand(
            self.switch_state,
            available=lambda: len(self._dynamic_states) > 0,
        )
        self._stop_current_command = PyCommand(
            self.stop_current_state,
            available=lambda: self._current_state_name is not None and self._current_state_name != self._default_state_name,
        )
        self._on_startup_instruction: str = ''
        super().__init__(
            channel=channel,
            container=container,
        )

    def is_connected(self) -> bool:
        # always true
        return True

    async def wait_connected(self) -> None:
        # always ready
        return

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Channel `{self}` is not running")

    async def switch_state(self, name: str) -> str:
        """
        switch current state into existing state by name.
        """
        if name == self._current_state_name:
            return f'{self._current_state_name} is already running'
        states = self._dynamic_states
        if name not in states:
            return f'state `{name}` not found.'
        stop_any = await self.stop_current_state()
        try:
            new_state = states[name]
            await new_state.on_startup()
            self._current_state_name = name
            self._current_state_running_task = asyncio.create_task(new_state.on_running())
            self._current_state = new_state
            return f"{stop_any}started current state `{name}`"
        finally:
            if self._current_state is None:
                await self.stop_current_state()

    async def add_virtual_channel(self, channel: Channel, wait_started: bool = False) -> bool:
        if isinstance(self._main_state, MutableChannelState):
            self._main_state.add_virtual_channel(channel)
            if wait_started:
                await self.refresh_metas()
                if runtime := self.tree.get_channel_runtime(channel):
                    await runtime.wait_started()
                    return True
        return False

    async def remove_virtual_channel(self, name: str, wait_refreshed: bool = False) -> bool:
        if isinstance(self._main_state, MutableChannelState):
            self._main_state.remove_virtual_channel(name)
            if wait_refreshed:
                await self.refresh_metas()
                if _ := self.fetch_sub_runtime(name):
                    return False
                return True
        return False

    async def stop_current_state(self) -> str:
        """
        stop current running state and return to default.
        """
        try:
            if self._current_state_running_task is not None and not self._current_state_running_task.done():
                self._current_state_running_task.cancel()
                try:
                    await self._current_state_running_task
                except asyncio.CancelledError:
                    pass
            self._current_state_running_task = None
            current_state_name = self._current_state_name
            self._current_state_name = None
            if not self._current_state:
                return "no current state is running. "
            await self._current_state.on_close()
            return f'{current_state_name} is stopped. '
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise
        except CommandError:
            raise
        except Exception as e:
            return f"stop current state error: {e}. "
        finally:
            self._current_state = None
            self._current_state_name = None
            self._current_state_running_task = None

    def sub_channels(self) -> dict[str, Channel]:
        result = self._main_state.get_children()
        return result

    def virtual_sub_channels(self) -> dict[str, Channel]:
        virtual_channels = self._main_state.get_virtual_children().copy()
        if self._current_state is not None:
            for name, child in self._current_state.get_children().copy().items():
                # new virtual children.
                virtual_channels[name] = child
            for name, child in self._current_state.get_virtual_children().copy().items():
                virtual_channels[name] = child
        return virtual_channels

    def is_dynamic(self) -> bool:
        states = self._dynamic_states
        if len(states) > 0:
            return True
        return self._main_state.is_dynamic()

    async def _generate_own_metas(self) -> dict[str, ChannelMeta]:
        if self.is_available() and self._static_meta_cache:
            # 返回缓存.
            return {'': self._static_meta_cache}

        # 通知所有 state/module: 即将重新生成 metas, 先做 async 状态同步
        await self.on_refresh_meta()

        dynamic = self.is_dynamic()
        name = self._name
        description = self.channel.description()
        main_state = self._main_state
        states_data = {}
        states = self._dynamic_states
        if len(states) > 0:
            states_data = {name: state.description() for name, state in states.items()}
            dynamic = True
        try:
            command_metas = []
            commands = self.own_commands()
            for command in commands.values():
                # 只添加需要动态更新的 command.
                if command.is_dynamic():
                    command.refresh_meta()
                cmd_meta = command.meta()
                if cmd_meta.dynamic:
                    dynamic = True
                command_metas.append(cmd_meta.model_copy())

            context_message_task = asyncio.create_task(self._get_context_messages())
            new_context_messages = await context_message_task

            meta = ChannelMeta(
                name=name,
                channel_id=self.channel.id(),
                available=main_state.is_available(),
                description=description,
                states=states_data,
                current_state=self._current_state_name or '',
                modules=list(self._modules.keys()),
                context=new_context_messages,
                instruction=self._on_startup_instruction,
            )
            meta.dynamic = dynamic
            meta.commands = command_metas
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            meta = ChannelMeta(
                name=name,
                description=description,
                available=False,
                failure="channel not available with system failure: %s" % e,
                dynamic=True,
            )
        if not meta.dynamic:
            self._static_meta_cache = meta
        return {"": meta}

    async def _get_context_messages(self) -> list[Message]:
        funcs = [self._main_state.get_context_messages()]
        if len(self._modules) > 0:
            for module in self._modules.values():
                if hasattr(module, 'get_context_messages'):
                    funcs.append(module.get_context_messages())
        # TODO: 考虑用 XML tag 包裹每个 module 的 context messages，
        # 避免自由合并产生的割裂感（模型不知道哪些内容来自哪个模块）。
        if current_state := self._get_current_state():
            funcs.append(current_state.get_context_messages())
        result = []
        done = await asyncio.gather(*funcs, return_exceptions=True)
        for t in done:
            if isinstance(t, list):
                result.extend(t)
            else:
                self.logger.error("%r get context messages receive invalid result %r", self, t)
        return list(self._wrap_messages(result))

    def _wrap_messages(self, messages: Iterable[Message | str | Image]) -> Iterable[Message]:
        for msg in messages:
            if isinstance(msg, Message):
                yield msg
            else:
                yield Message.new(tag='').with_content(msg)

    def _get_current_state(self) -> ChannelState | None:
        if self._current_state is None:
            return None
        if not self._current_state.is_available():
            self._current_state = None
            self._current_state_name = None
            if self._current_state_running_task is not None:
                self._current_state_running_task.cancel()
            self._current_state_running_task = None
            return None
        return self._current_state

    # ---- commands ---- #

    def _is_available(self) -> bool:
        return self._main_state.is_available()

    def has_own_command(self, name: CommandUniqueName) -> bool:
        path, name = Command.split_unique_name(name)
        if path:
            return False
        command = self._get_own_command(name)
        return command is not None

    def own_commands(self, available_only: bool = True) -> dict[str, Command]:
        if not self.is_available():
            return {}
        result = {}
        for name, command in self._own_commands().items():
            if not available_only or command.is_available():
                result[name] = self._wrap_origin_command(command)
        return result

    def _own_commands(self) -> dict[str, Command]:
        commands = self._main_state.own_commands().copy()
        if self._current_state is not None:
            commands[self._stop_current_command.name()] = self._stop_current_command
        if len(self._dynamic_states) > 0:
            commands[self._switch_state_command.name()] = self._switch_state_command

        # modules — 永久能力模块，累积叠加。main_state 的命令优先。
        if len(self._modules) > 0:
            for module in self._modules.values():
                for name, command in module.own_commands().items():
                    if name not in commands:
                        commands[name] = command

        if self._current_state is not None:
            for name, command in self._current_state.own_commands().items():
                if name not in commands:
                    commands[name] = command
        return commands

    def _wrap_origin_command(self, command: Command | None) -> Command | None:
        """
        确保函数被单独调用时也拥有自己的 ctx
        """
        if command is None:
            return None

        ctx = ChannelCtx(self, None)
        return CommandWrapper.wrap(command, ctx_fn=ctx.in_ctx)

    def get_own_command(
            self,
            name: CommandUniqueName,
    ) -> Optional[Command]:
        if self._current_state is not None and name == self._stop_current_command.name():
            return self._stop_current_command
        if len(self._dynamic_states) > 0 and name == self._switch_state_command.name():
            return self._switch_state_command

        path, name = Command.split_unique_name(name)
        if path:
            return None
        return self._wrap_origin_command(self._get_own_command(name))

    def _get_own_command(
            self,
            name: CommandUniqueName,
    ) -> Optional[Command]:
        if self._current_state is not None and name == self._stop_current_command.name():
            return self._stop_current_command
        if len(self._dynamic_states) > 0 and name == self._switch_state_command.name():
            return self._switch_state_command
        command = self._main_state.get_own_command(name)
        if command is not None:
            return command
        if len(self._modules) > 0:
            for module in self._modules.values():
                cmd = module.own_commands().get(name)
                if cmd is not None:
                    return cmd
        if self._current_state is None:
            return None
        return self._current_state.get_own_command(name)

    async def on_running(self) -> None:
        await self._main_state.on_running()

    async def on_idle(self) -> None:
        try:
            if not self.is_running():
                return
            idle_func = [self._main_state.on_idle()]
            if self._current_state is not None:
                idle_func.append(self._current_state.on_idle())
            done = await asyncio.gather(*idle_func, return_exceptions=True)
            for r in done:
                if isinstance(r, Exception):
                    self.logger.error("%r run on_idle func failed: %s", self, r)

        except asyncio.CancelledError:
            self.logger.info(f"%r on_idle done", self)
            return
        except Exception as e:
            self.logger.exception("%r on idle failed: %s", self, e)
            raise

    async def on_refresh_meta(self) -> None:
        """通知 main state, modules, current state: 即将重新生成 metas。"""
        refresh_funcs = [self._main_state.on_refresh_meta()]
        if self._current_state is not None:
            refresh_funcs.append(self._current_state.on_refresh_meta())
        for module in self._modules.values():
            if hasattr(module, 'on_refresh_meta'):
                refresh_funcs.append(module.on_refresh_meta())
        done = await asyncio.gather(*refresh_funcs, return_exceptions=True)
        for r in done:
            if isinstance(r, Exception):
                self.logger.error("%r on_refresh_meta func failed: %s", self, r)

    def __repr__(self):
        return self.log_prefix

    async def on_startup(self) -> None:
        # 准备 start up 的运行.
        main_state = self._main_state
        await main_state.on_startup()
        self._on_startup_instruction = await main_state.get_instruction()

        # 启动所有永久能力模块。
        for module in self._modules.values():
            if hasattr(module, 'on_startup'):
                await module.on_startup()

        default = self.channel.default_state_name()
        # start the default state.
        if default in self._dynamic_states:
            await self.switch_state(default)

    async def on_close(self) -> None:
        # 先关闭 current_state，再关闭 module，最后关闭 main。
        await self.stop_current_state()
        for module in self._modules.values():
            if hasattr(module, 'on_close'):
                await module.on_close()
        await self._main_state.on_close()

    def prepare_container(self, container: IoCContainer) -> IoCContainer:
        # 只有这一个地方是 state 调用 bootstrap 的地方.
        # main state 调用 bootstrap.
        self._main_state.bootstrap(container)
        for state in self._dynamic_states.values():
            # 保证所有的状态的 bootstrap 都被调用了.
            state.bootstrap(container)
        container = super().prepare_container(container)
        return container
