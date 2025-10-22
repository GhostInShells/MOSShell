import inspect
from typing import Type, Optional, List, Callable, Dict, Tuple, Iterable, Any, Coroutine, Awaitable
from typing_extensions import Self

from ghoshell_moss import CommandTask
from ghoshell_moss.concepts.channel import (
    ChannelClient, Builder, Channel, LifecycleFunction, StringType, CommandFunction, ChannelMeta, R,
)
from ghoshell_moss.concepts.command import Command, PyCommand
from ghoshell_moss.concepts.errors import CommandError, FatalError
from ghoshell_moss.helpers.func import unwrap_callable_or_value
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, ensure_tasks_done_or_cancel
from ghoshell_container import (
    Container, IoCContainer, INSTANCE, BINDING, Provider, provide, set_container
)
from ghoshell_common.helpers import uuid
from contextvars import copy_context
import asyncio
import logging
import threading

__all__ = ['PyChannel', 'PyChannelBuilder', 'PyChannelClient']


class PyChannelBuilder(Builder):

    def __init__(self, *, name: str, description: str, block: bool):
        self.name = name
        self.block = block
        self.description = description
        self.description_fn: Optional[StringType] = None
        self.available_fn: Optional[Callable[[], bool]] = None
        self.policy_run_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.policy_pause_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.on_clear_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.on_start_up_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.on_stop_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.on_clear_funcs: List[Tuple[LifecycleFunction, bool]] = []
        self.providers: List[Provider] = []
        self.commands: Dict[str, Command] = {}
        self.contracts: List = []

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

    def command(
            self,
            *,
            name: str = "",
            chan: str | None = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            interface: Optional[StringType] = None,
            available: Optional[Callable[[], bool]] = None,
            block: Optional[bool] = None,
            call_soon: bool = False,
            return_command: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction]:
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

    def with_contracts(self, *contracts: Type) -> Self:
        self.contracts.extend(contracts)
        return self

    def with_binding(self, contract: Type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        provider = provide(contract, singleton=True)(binding)
        self.providers.append(provider)
        return self


class PyChannel(Channel):

    def __init__(
            self,
            *,
            name: str,
            description: str = "",
            block: bool = True,
    ):
        self._name = name
        self._description = description
        self._client: Optional[ChannelClient] = None
        self._children: Dict[str, Channel] = {}
        self._block = block
        # decorators
        self._builder = PyChannelBuilder(
            name=name,
            description=description,
            block=block,
        )

    def name(self) -> str:
        return self._name

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError("Server not start")
        elif self._client.is_running():
            return self._client
        else:
            raise RuntimeError("Server not running")

    def include_channels(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        if parent is not None:
            descendant = self.descendants().get(parent)
            if descendant is None:
                raise LookupError(f"the children parent name of {parent} does not exist")
            descendant.include_channels(*children)
            return

        for child in children:
            self._children[child.name()] = child
        return self

    def new_child(self, name: str) -> Self:
        child = PyChannel(name=name)
        self._children[name] = child
        return child

    def children(self) -> Dict[str, "Channel"]:
        return self._children

    def descendants(self) -> Dict[str, "Channel"]:
        channels = {}
        for child in self._children.values():
            channels[child.name()] = child
            for descendant in child.descendants().values():
                channels[descendant.name()] = descendant
        return channels

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError("Server already running")
        self._client = PyChannelClient(
            children=self._children,
            container=container,
            builder=self._builder,
        )
        return self._client

    def is_running(self) -> bool:
        return self._client is not None and self._client.is_running()

    @property
    def build(self) -> Builder:
        return self._builder

    def __del__(self):
        self._children.clear()


class PyChannelClient(ChannelClient):

    def __init__(
            self,
            *,
            children: Dict[str, Channel],
            builder: PyChannelBuilder,
            container: Optional[IoCContainer] = None,
            uid: Optional[str] = None,
    ):
        if container is not None:
            container = Container(parent=container, name=f"moss/chan_client/{builder.name}")
        else:
            container = Container(name=f"moss/chan_client/{builder.name}")
        self.container = container
        self.id = uid or uuid()
        self._children = children
        self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self._builder = builder
        self._meta_cache: Optional[ChannelMeta] = None
        self._stop_event = ThreadSafeEvent()
        self._failed_exception: Optional[Exception] = None
        self._policy_is_running = ThreadSafeEvent()
        self._policy_tasks: List[asyncio.Task] = []
        self._policy_lock = threading.Lock()
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = threading.Event()

    def __del__(self):
        self.container.shutdown()

    def is_none_block(self) -> bool:
        return self._builder.block

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        if no_cache or self._meta_cache is None:
            self._meta_cache = self._refresh_meta()
        return self._meta_cache

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

    def _refresh_meta(self) -> ChannelMeta:
        command_metas = []
        for command in self.commands(available_only=False).values():
            command_metas.append(command.meta())
        name = self._builder.name
        meta = ChannelMeta(
            name=name,
            channel_id=self.id,
            available=self.is_available(),
            description=self.description(),
            children=list(self._children.keys()),
        )
        meta.commands = command_metas
        return meta

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
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

    def update(self) -> ChannelMeta:
        self._check_running()
        self._meta_cache = self._refresh_meta()
        return self._meta_cache

    async def policy_run(self) -> None:
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
            self._logger.info(f"Policy tasks cancelled")
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
        self._logger.exception(error)
        self._starting = False
        self._stop_event.set()

    async def clear(self) -> None:
        clear_tasks = []
        for clear_func, is_coroutine in self._builder.policy_pause_funcs:
            if is_coroutine:
                task = asyncio.create_task(clear_func())
            else:
                task = asyncio.to_thread(clear_func)
            clear_tasks.append(task)
        try:
            await asyncio.gather(*clear_tasks, return_exceptions=False)
        except asyncio.CancelledError as e:
            self._logger.error(f"Cancelled due to {e}")
        except FatalError as e:
            self._logger.exception(e)
            raise
        except Exception as e:
            self._logger.exception(e)

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        # 启动所有容器.
        await asyncio.to_thread(self._self_boostrap)
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

        # 运行所有的子 channel, 传递相同的服务.
        start_all_children = []
        for child in self._children.values():
            if not child.is_running():
                client = child.bootstrap(self.container)
                start_all_children.append(client.start())
        if len(start_all_children) > 0:
            await asyncio.gather(*start_all_children, return_exceptions=False)
        self._started = True

    def _self_boostrap(self) -> None:
        self.container.register(*self._builder.providers)
        self.container.set(ChannelClient, self)
        self.container.bootstrap()

    async def execute(self, task: CommandTask[R]) -> R:
        return await self._execute(task.meta.name, *task.args, **task.kwargs)

    async def _execute(self, name: str, *args, **kwargs) -> Any:
        """
        直接在本地运行.
        """
        func = self._get_execute_func(name)
        # 方便使用 get_contract 可以拿到上下文.
        ctx = copy_context()
        set_container(self.container)
        # 必须返回的是一个 Awaitable 的函数.
        result = await ctx.run(func, *args, **kwargs)
        return result

    def _get_execute_func(self, name: str) -> Callable[..., Coroutine | Awaitable]:
        """重写这个函数可以重写调用逻辑实现. """
        command = self.get_command(name)
        if command is None:
            raise NotImplementedError(f"Command '{name}' is not implemented.")
        if not command.is_available():
            raise CommandError(
                CommandError.NOT_AVAILABLE,
                f"Command '{name}' is not available.",
            )
        return command.__call__

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_event.set()
        await self.policy_pause()
        await self.clear()
        await asyncio.to_thread(self.container.shutdown)
        self.container.shutdown()
