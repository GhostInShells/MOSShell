import asyncio
import contextvars
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import copy_context
from typing import Any, Optional

from ghoshell_common.helpers import uuid
from ghoshell_container import BINDING, INSTANCE, Container, IoCContainer, Provider, provide
from typing_extensions import Self

from ghoshell_moss.core.concepts.channel import (
    Builder,
    Channel,
    ChannelBroker,
    ChannelMeta,
    CommandFunction,
    ContextMessageFunction,
    LifecycleFunction,
    R,
    StringType,
)
from ghoshell_moss.core.concepts.command import Command, CommandTask, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode, FatalError
from ghoshell_moss.core.concepts.states import MemoryStateStore, StateModel, StateStore
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent, ensure_tasks_done_or_cancel
from ghoshell_moss.core.helpers.func import unwrap_callable_or_value

__all__ = ["PyChannel", "PyChannelBroker", "PyChannelBuilder"]


class PyChannelBuilder(Builder):
    def __init__(self, *, name: str, description: str, block: bool):
        self.name = name
        self.block = block
        self.description = description
        self.description_fn: Optional[StringType] = None
        self.available_fn: Optional[Callable[[], bool]] = None
        self.state_models: list[StateModel] = []
        self.policy_run_funcs: list[tuple[LifecycleFunction, bool]] = []
        self.policy_pause_funcs: list[tuple[LifecycleFunction, bool]] = []
        self.on_clear_funcs: list[tuple[LifecycleFunction, bool]] = []
        self.on_start_up_funcs: list[tuple[LifecycleFunction, bool]] = []
        self.on_stop_funcs: list[tuple[LifecycleFunction, bool]] = []
        self.providers: list[Provider] = []
        self.context_message_function: Optional[ContextMessageFunction] = None
        self.commands: dict[str, Command] = {}
        self.contracts: list = []
        self.container_instances = {}

    def with_description(self) -> Callable[[StringType], StringType]:
        def wrapper(func: StringType) -> StringType:
            self.description_fn = func
            return func

        return wrapper

    def with_available(self) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        def wrapper(func: Callable[[], bool]) -> Callable[[], bool]:
            self.available_fn = func
            return func

        return wrapper

    def state_model(self) -> Callable[[type[StateModel]], StateModel]:
        """
        注册一个状态模型.

        @chan.build.state_model()
        class DemoStateModel(StateBaseModel):
            state_name = "demo"
            state_desc = "demo state model"
        """

        def wrapper(model: type[StateModel]) -> StateModel:
            instance = model()
            self.state_models.append(instance)
            return instance

        return wrapper

    def with_context_messages(self, func: ContextMessageFunction) -> Self:
        self.context_message_function = func
        return self

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
        block: Optional[bool] = None,
        call_soon: bool = False,
        return_command: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:
        def wrapper(func: CommandFunction) -> CommandFunction:
            command = PyCommand(
                func,
                name=name,
                chan=chan if chan is not None else self.name,
                doc=doc,
                comments=comments,
                tags=tags,
                interface=interface,
                available=available,
                block=block if block is not None else self.block,
                call_soon=call_soon,
            )
            self.commands[command.name()] = command
            if return_command:
                return command
            return func

        return wrapper

    def on_policy_run(self, run_policy: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(run_policy)
        self.policy_run_funcs.append((run_policy, is_coroutine))
        return run_policy

    def on_policy_pause(self, pause_policy: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(pause_policy)
        self.policy_pause_funcs.append((pause_policy, is_coroutine))
        return pause_policy

    def on_clear(self, clear_func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(clear_func)
        self.on_clear_funcs.append((clear_func, is_coroutine))
        return clear_func

    def on_start_up(self, start_func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(start_func)
        self.on_start_up_funcs.append((start_func, is_coroutine))
        return start_func

    def on_stop(self, stop_func: LifecycleFunction) -> LifecycleFunction:
        is_coroutine = inspect.iscoroutinefunction(stop_func)
        self.on_stop_funcs.append((stop_func, is_coroutine))
        return stop_func

    def with_providers(self, *providers: Provider) -> Self:
        self.providers.extend(providers)
        return self

    def with_contracts(self, *contracts: type) -> Self:
        self.contracts.extend(contracts)
        return self

    def with_binding(self, contract: type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        if binding and isinstance(contract, type) and isinstance(binding, contract):
            self.container_instances[contract] = binding
            return self

        provider = provide(contract, singleton=True)(binding)
        self.providers.append(provider)
        return self


class PyChannel(Channel):
    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        # todo: block 还是叫 blocking 吧.
        block: bool = True,
        dynamic: bool | None = None,
    ):
        """
        :param name: channel 的名称.
        :param description: channel 的静态描述, 给模型看的.
        :param block: channel 里默认的 command 类型, 是阻塞的还是非阻塞的.
        :param dynamic: 这个 channel 对大模型而言是否是动态的.
                        如果是动态的, 大模型每一帧思考时, 都会从 channel 获取最新的状态.
        """
        self._name = name
        self._description = description
        self._broker: Optional[ChannelBroker] = None
        self._children: dict[str, Channel] = {}
        self._block = block
        self._dynamic = dynamic
        # decorators
        self._builder = PyChannelBuilder(
            name=name,
            description=description,
            block=block,
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

    def new_child(self, name: str) -> Self:
        child = PyChannel(name=name)
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
            self._state_store = MemoryStateStore(name)
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

    def is_none_block(self) -> bool:
        return self._builder.block

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
        if self._builder.description_fn is not None:
            return unwrap_callable_or_value(self._builder.description_fn)
        return self._builder.description

    def is_available(self) -> bool:
        if not self.is_running():
            return False
        if self._builder.available_fn is not None:
            return self._builder.available_fn()
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
        commands = list(self._builder.commands.values())
        # 刷新所有的 command 的 meta 信息.
        refresh_message_task = None
        if self._builder.context_message_function:
            dynamic = True
            if inspect.iscoroutinefunction(self._builder.context_message_function):
                refresh_message_task = asyncio.create_task(self._builder.context_message_function())
            else:
                refresh_message_task = asyncio.create_task(asyncio.to_thread(self._builder.context_message_function))

        refreshing_commands = []
        for command in commands:
            # 只添加需要动态更新的 command.
            if command.meta().dynamic:
                refreshing_commands.append(command.refresh_meta())
                dynamic = True

        # 更新所有的 动态 commands.
        if len(refreshing_commands) > 0:
            done = await asyncio.gather(*refreshing_commands, return_exceptions=True)
            idx = 0
            for refreshed in done:
                if isinstance(refreshed, Exception):
                    command = commands[idx]
                    self._logger.exception("Refresh command meta failed on command %s", command)
                idx += 1

        for command in commands:
            try:
                command_metas.append(command.meta())
            except Exception as exc:
                # 异常的命令直接不返回了.
                self._logger.exception("Exception on get meta from command %s", command.name())

        name = self._builder.name
        new_context_messages = []
        if refresh_message_task is not None:
            try:
                new_context_messages = await refresh_message_task
            except Exception as exc:
                self._logger.exception("Exception on refresh message task %s", refresh_message_task)
                raise

        meta = ChannelMeta(
            name=name,
            channel_id=self.id,
            available=self.is_available(),
            description=self.description(),
            children=self._get_children_fn(),
            context=new_context_messages,
        )
        meta.dynamic = dynamic
        meta.commands = command_metas
        return meta

    def commands(self, available_only: bool = True) -> dict[str, Command]:
        if not self.is_available():
            return {}
        result = {}
        for command in self._builder.commands.values():
            if not available_only or command.is_available():
                result[command.name()] = command
        return result

    def get_command(
        self,
        name: str,
    ) -> Optional[Command]:
        return self._builder.commands.get(name, None)

    async def update_meta(self) -> ChannelMeta:
        self._check_running()
        self._meta_cache = await self._generate_meta_with_ctx()
        return self._meta_cache

    async def policy_run(self) -> None:
        ctx = contextvars.copy_context()
        self._set_chan_ctx_fn()
        await ctx.run(self._policy_run)

    async def _policy_run(self) -> None:
        try:
            self._check_running()
            with self._policy_lock:
                if self._policy_is_running.is_set():
                    return
                policy_tasks = []
                for policy_run_func, is_coroutine in self._builder.policy_run_funcs:
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
        ctx = contextvars.copy_context()
        await ctx.run(self._policy_pause)

    async def _policy_pause(self) -> None:
        try:
            with self._policy_lock:
                await self._clear_running_policies()
                pause_tasks = []
                for policy_pause_func, is_coroutine in self._builder.policy_pause_funcs:
                    if is_coroutine:
                        task = asyncio.create_task(policy_pause_func())
                    else:
                        task = asyncio.to_thread(policy_pause_func)
                    pause_tasks.append(task)
                await ensure_tasks_done_or_cancel(*pause_tasks, cancel=self._stop_event.wait)

        except Exception as e:
            self._fail(e)

    def _fail(self, error: Exception) -> None:
        self._logger.exception("Channel failed")
        self._starting = False
        self._stop_event.set()

    async def clear(self) -> None:
        clear_tasks = []
        for clear_func, is_coroutine in self._builder.on_clear_funcs:
            if is_coroutine:
                task = asyncio.create_task(clear_func())
            else:
                task = asyncio.to_thread(clear_func)
            clear_tasks.append(task)
        try:
            await asyncio.gather(*clear_tasks, return_exceptions=False)
        except asyncio.CancelledError:
            self._logger.exception("Clear cancelled")
        except FatalError:
            self._logger.exception("Clear failed with fatal error")
            raise
        except Exception:
            self._logger.exception("Clear failed")

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        # 启动所有容器.
        await asyncio.to_thread(self._self_boostrap)
        ctx = contextvars.copy_context()
        # prepare context var
        self._set_chan_ctx_fn()
        # startup with ctx.
        await ctx.run(self._run_start_up)
        self._started = True
        # 然后再更新 meta.
        await ctx.run(self.refresh_meta)

    async def _run_start_up(self) -> None:
        startups = []
        # 准备 start up 的运行.
        if len(self._builder.on_start_up_funcs) > 0:
            for on_start_func, is_coroutine in self._builder.on_start_up_funcs:
                if is_coroutine:
                    task = asyncio.create_task(on_start_func())
                else:
                    task = asyncio.to_thread(on_start_func)
                startups.append(task)
            # 并行启动.
            await asyncio.gather(*startups, return_exceptions=False)

    def _self_boostrap(self) -> None:
        # 注册所有的状态模型.
        self._state_store.register(*self._builder.state_models)
        # 自己的 container 自己才可以启动.
        self.container.register(*self._builder.providers)
        if len(self._builder.container_instances) > 0:
            for contract, instance in self._builder.container_instances.items():
                self.container.set(contract, instance)
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
        await self.clear()
        await ctx.run(self._run_on_stop)
        self._stop_event.set()
        # 自己的 container 自己才可以关闭.
        await asyncio.to_thread(self.container.shutdown)

    async def _run_on_stop(self) -> None:
        on_stop_calls = []
        # 准备 start up 的运行.
        if len(self._builder.on_start_up_funcs) > 0:
            for on_stop_func, is_coroutine in self._builder.on_stop_funcs:
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
