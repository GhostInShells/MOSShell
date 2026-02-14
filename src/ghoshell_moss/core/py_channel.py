import asyncio
import contextvars
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import copy_context
from typing import Any, Optional

from ghoshell_common.helpers import uuid
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
    R,
    StringType,
)
from ghoshell_moss.core.concepts.command import Command, CommandTask, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode, FatalError
from ghoshell_moss.core.concepts.states import BaseStateStore, StateModel, StateStore
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent, ensure_tasks_done_or_cancel
from ghoshell_moss.core.helpers.func import unwrap_callable_or_value

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

    def on_idle(self, run_policy: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(run_policy)
        self._on_idle_funcs.append((run_policy, is_coroutine))
        return run_policy

    async def run_idling(self):
        await self._run_funcs(self._on_idle_funcs)

    def on_start_up(self, start_func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(start_func)
        self._on_start_up_funcs.append((start_func, is_coroutine))
        return start_func

    async def run_start_up(self) -> None:
        await self._run_funcs(self._on_start_up_funcs)

    def on_stop(self, stop_func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(stop_func)
        self._on_stop_funcs.append((stop_func, is_coroutine))
        return stop_func

    @classmethod
    async def _run_funcs(cls, funcs: list[tuple[LifecycleFunction, bool]]) -> None:
        if len(funcs) == 0:
            return

        cors = []
        for func, is_coroutine in funcs:
            if is_coroutine:
                cor = func()
            else:
                cor = asyncio.to_thread(func)
            cors.append(cor)
        done = await asyncio.gather(*cors, return_exceptions=False)
        for _ in done:
            pass

    async def run_stop(self) -> None:
        await self._run_funcs(self._on_stop_funcs)

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

    @property
    def build(self) -> Builder:
        return self._builder

    @property
    def broker(self) -> ChannelBroker:
        if self._broker is None:
            raise RuntimeError("Server not start")
        elif self._broker.is_running():
            return self._broker
        else:
            raise RuntimeError("Server not running")

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
            name=self._name,
            set_chan_ctx_fn=self.set_context_var,
            get_children_fn=self._get_children_names,
            container=container,
            builder=self._builder,
            dynamic=self._dynamic,
        )
        return self._broker

    def _get_children_names(self) -> list[str]:
        return list(self._children.keys())

    def is_running(self) -> bool:
        return self._broker is not None and self._broker.is_running()

    def __del__(self):
        self._children.clear()


class PyChannelBroker(ChannelBroker):
    def __init__(
            self,
            name: str,
            *,
            set_chan_ctx_fn: Callable[[], None],
            get_children_fn: Callable[[], list[str]],
            builder: PyChannelBuilder,
            container: Optional[IoCContainer] = None,
            uid: Optional[str] = None,
            dynamic: bool | None = None,
    ):
        # todo: 考虑移除 channel 级别的 container, 降低分形构建的理解复杂度. 也许不移除才是最好的.
        container = Container(parent=container, name=f"moss/py_channel/{name}/broker")
        # 下面这行赋值必须被 del 掉, 否则会因为互相持有导致垃圾回收失败.
        self._name = name
        self._set_chan_ctx_fn = set_chan_ctx_fn
        self._get_children_fn = get_children_fn
        self._container = container
        self._id = uid or uuid()
        self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self._state_store = self.container.get(StateStore)
        self._dynamic = dynamic
        if self._state_store is None:
            self._state_store = BaseStateStore(name)
            self.container.set(StateStore, self._state_store)
        self._builder = builder
        self._meta_cache: Optional[ChannelMeta] = None
        self._stop_event = ThreadSafeEvent()
        self._failed_exception: Optional[Exception] = None
        self._policy_is_running = ThreadSafeEvent()
        self._policy_tasks: list[asyncio.Task] = []
        self._policy_lock = threading.Lock()
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = threading.Event()

    def name(self) -> str:
        return self._name

    @property
    def container(self) -> IoCContainer:
        return self._container

    @property
    def id(self) -> str:
        return self._id

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def meta(self) -> ChannelMeta:
        if self._meta_cache is None:
            raise RuntimeError(f"Channel broker {self._name} not initialized")
        return self._meta_cache.model_copy()

    async def refresh_meta(self) -> None:
        self._meta_cache = await self._generate_meta_with_ctx()

    def is_connected(self) -> bool:
        return True

    async def wait_connected(self) -> None:
        # always ready
        return

    def description(self) -> str:
        # todo: redefine
        return ""

    def is_available(self) -> bool:
        if not self.is_running():
            return False
        if self._builder._available_fn is not None:
            return self._builder.is_available()
        return True

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Channel {self} not running")

    async def _generate_meta_with_ctx(self) -> ChannelMeta:
        ctx = contextvars.copy_context()
        self._set_chan_ctx_fn()
        # 保证 generate meta 运行在 channel 的 ctx 下.
        return await ctx.run(self._generate_meta)

    async def _generate_meta(self) -> ChannelMeta:
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
                    self._logger.exception("Refresh command meta failed on command %s", command)
                idx += 1

        for command in commands:
            command_metas.append(command.meta())

        name = self._name
        instruction_message_task = asyncio.create_task(self._builder.get_instruction_messages())
        context_message_task = asyncio.create_task(self._builder.get_context_message())
        await asyncio.gather(instruction_message_task, context_message_task)
        new_context_messages = await context_message_task
        new_instruction_messages = await instruction_message_task

        meta = ChannelMeta(
            name=name,
            channel_id=self.id,
            available=self.is_available(),
            description=self.description(),
            children=self._get_children_fn(),
            context=new_context_messages,
            instructions=new_instruction_messages,
        )
        meta.dynamic = dynamic
        meta.commands = command_metas
        return meta

    def commands(self, available_only: bool = True) -> dict[str, Command]:
        if not self.is_available():
            return {}
        result = {}
        for command in self._builder.commands():
            if not available_only or command.is_available():
                result[command.name()] = command
        return result

    def get_command(
            self,
            name: str,
    ) -> Optional[Command]:
        return self._builder.get_command(name)

    async def update_meta(self) -> ChannelMeta:
        self._check_running()
        self._meta_cache = await self._generate_meta_with_ctx()
        return self._meta_cache

    async def on_idle(self) -> None:
        ctx = contextvars.copy_context()
        self._set_chan_ctx_fn()
        await ctx.run(self._run_idling)

    async def _run_idling(self) -> None:
        try:
            self._check_running()
            with self._policy_lock:
                if self._policy_is_running.is_set():
                    return
                policy_tasks = []
                for policy_run_func, is_coroutine in self._builder._on_idle_funcs:
                    if is_coroutine:
                        task = asyncio.create_task(policy_run_func())
                    else:
                        task = asyncio.create_task(asyncio.to_thread(policy_run_func))
                    policy_tasks.append(task)
                self._policy_tasks = policy_tasks
                if len(policy_tasks) > 0:
                    self._policy_is_running.set()

        except asyncio.CancelledError:
            self._logger.info("Policy tasks cancelled")
            return
        except Exception as e:
            self._fail(e)

    async def _cancel_if_stopped(self) -> None:
        await self._stop_event.wait()

    async def _clear_running_policies(self) -> None:
        if len(self._policy_tasks) > 0:
            tasks = self._policy_tasks
            self._policy_tasks.clear()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await ensure_tasks_done_or_cancel(*tasks, cancel=self._stop_event.wait)
            except asyncio.CancelledError:
                return
            finally:
                self._policy_is_running.clear()

    async def policy_pause(self) -> None:
        pass

    def _fail(self, error: Exception) -> None:
        self._logger.exception("Channel failed: %s", error)
        self._starting = False
        self._stop_event.set()

    async def on_clear(self) -> None:
        pass

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        # 启动所有容器.
        await asyncio.to_thread(self._self_boostrap)
        self._state_store = self._builder.get_states(self.id, self.container.get(StateStore))
        await self._state_store.start()

        ctx = contextvars.copy_context()
        # prepare context var
        self._set_chan_ctx_fn()
        # startup with ctx.
        await ctx.run(self._run_start_up)
        self._started = True
        # 然后再更新 meta.
        await ctx.run(self.refresh_meta)

    async def _run_start_up(self) -> None:
        # 准备 start up 的运行.
        await self._builder.run_start_up()

    def _self_boostrap(self) -> None:
        # 自己的 container 自己才可以启动.
        self._builder.update_container(self.container)
        self.container.bootstrap()

    async def execute(self, task: CommandTask[R]) -> R:
        ctx = copy_context()
        self._set_chan_ctx_fn()
        return await ctx.run(self._execute, task.meta.name, task.args, task.kwargs)

    async def _execute(self, name: str, args, kwargs) -> Any:
        """
        直接在本地运行.
        """
        func = self._get_execute_func(name)
        # 必须返回的是一个 Awaitable 的函数.
        result = await func(*args, **kwargs)
        return result

    def _get_execute_func(self, name: str) -> Callable[..., Coroutine | Awaitable]:
        """重写这个函数可以重写调用逻辑实现."""
        command = self.get_command(name)
        if command is None:
            raise NotImplementedError(f"Command '{name}' is not implemented.")
        if not command.is_available():
            raise CommandErrorCode.NOT_AVAILABLE.error(
                f"Command '{name}' is not available.",
            )
        return command.__call__

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        ctx = copy_context()
        self._set_chan_ctx_fn()
        await ctx.run(self.policy_pause)
        await self.on_clear()
        await ctx.run(self._run_on_stop)
        if self._state_store:
            await self._state_store.close()
        self._stop_event.set()
        # 自己的 container 自己才可以关闭.
        await asyncio.to_thread(self.container.shutdown)

    async def _run_on_stop(self) -> None:
        await self._builder.run_stop()
        on_stop_calls = []
        # 准备 start up 的运行.
        if len(self._builder._on_start_up_funcs) > 0:
            for on_stop_func, is_coroutine in self._builder._on_stop_funcs:
                if is_coroutine:
                    task = asyncio.create_task(on_stop_func())
                else:
                    task = asyncio.to_thread(on_stop_func)
                on_stop_calls.append(task)
            # 并行启动.
            done = await asyncio.gather(*on_stop_calls, return_exceptions=True)
            for r in done:
                if isinstance(r, Exception):
                    self._logger.error("channel %s on stop function failed: %s", self._name, r)

    @property
    def states(self) -> StateStore:
        return self._state_store
