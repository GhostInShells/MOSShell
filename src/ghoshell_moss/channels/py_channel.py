import inspect
from typing import Type, Optional, List, Callable, Dict, Tuple, Iterable
from typing_extensions import Self

from ghoshell_moss.concepts.channel import (
    Controller, Builder, Channel, LifecycleFunction, StringType, FunctionCommand, ChannelMeta,
)
from ghoshell_moss.concepts.command import Command, PyCommand
from ghoshell_moss.helpers.func import run_coroutine_with_cancel, unwrap_callable_or_value
from ghoshell_container import Container, IoCContainer, INSTANCE, BINDING, Provider, provide
from ghoshell_moss.helpers.event import ThreadSafeEvent
import asyncio
import logging
import threading


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
            block: bool = True,
            call_soon: bool = False,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        def wrapper(func: FunctionCommand) -> FunctionCommand:
            command = PyCommand(
                func,
                name=name,
                chan=chan if chan is not None else self.name,
                doc=doc,
                comments=comments,
                tags=tags,
                interface=interface,
                available=available,
                block=block,
                call_soon=call_soon,
            )
            self.commands[command.name()] = command
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
        self._controller: Optional[Controller] = None
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
    def controller(self) -> Controller:
        if self._controller is None:
            raise RuntimeError("Controller not start")
        elif self._controller.is_running():
            return self._controller
        else:
            raise RuntimeError("Controller not running")

    def with_children(self, *children: "Channel") -> Self:
        for child in children:
            self._children[child.name()] = child
        return self

    def children(self) -> Dict[str, "Channel"]:
        return self._children

    def descendants(self) -> Dict[str, "Channel"]:
        channels = {}
        for child in self._children.values():
            channels[child.name()] = child
            for descendant in child.descendants().values():
                channels[descendant.name()] = descendant
        return channels

    def get_channel(self, name: str) -> Optional[Self]:
        if name == self._name:
            return self
        descendants = self.descendants()
        return descendants.get(name, None)

    def run(self, container: Optional[IoCContainer] = None) -> "Controller":
        if self._controller is not None and self._controller.is_running():
            raise RuntimeError("Controller already running")
        self._controller = PyChannelController(
            children=self.children(),
            container=container,
            builder=self._builder,
        )
        return self._controller

    @property
    def build(self) -> Builder:
        return self._builder


class PyChannelController(Controller):

    def __init__(
            self,
            *,
            children: Dict[str, Channel],
            builder: PyChannelBuilder,
            container: Optional[IoCContainer] = None,
    ):
        if container is not None:
            container = Container(parent=container, name=f"chan/container/{builder.name}")
        else:
            container = Container(name=f"chan/container/{builder.name}")
        self.container = container
        self._children = children
        self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self._builder = builder
        self._meta_cache: Optional[ChannelMeta] = None
        self._stopped_event = ThreadSafeEvent()
        self._failed_exception: Optional[Exception] = None
        self._policy_is_running = ThreadSafeEvent()
        self._policy_tasks: List[asyncio.Task] = []
        self._policy_lock = threading.Lock()
        self._started = False
        self._stopped = False

    def is_blocking(self) -> bool:
        return self._builder.block

    def is_running(self) -> bool:
        return self._started and not self._stopped_event.is_set()

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
        for command in self.commands(available_only=False):
            command_metas.append(command.meta())
        meta = ChannelMeta(
            name=self._builder.name,
            available=self.is_available(),
            description=self.description(),
            children=list(self._children.keys()),
        )
        return meta

    def commands(self, available_only: bool = True) -> Iterable[Command]:
        if not self.is_available():
            yield from []
            return
        for command in self._builder.commands.values():
            if not available_only or command.is_available():
                yield command

    def get_command(
            self,
            name: str,
            *,
            is_fullname: bool = False,
    ) -> Optional[Command]:
        if not is_fullname:
            name = PyCommand.make_fullname(self._builder.name, name)
        return self._builder.commands.get(name, None)

    def update(self) -> ChannelMeta:
        self._check_running()
        self._meta_cache = self._refresh_meta()
        return self._meta_cache

    async def policy_run(self) -> None:
        try:
            self._check_running()
            with self._policy_lock:
                self._check_running()
                if self._policy_is_running.is_set():
                    return
                policy_tasks = []
                for policy_run_func, is_coroutine in self._builder.policy_run_funcs:
                    if is_coroutine:
                        task = asyncio.create_task(policy_run_func())
                    else:
                        task = asyncio.to_thread(policy_run_func)
                    cancel_scope_task = run_coroutine_with_cancel(task, self._stopped_event.wait)
                    policy_tasks.append(cancel_scope_task)
                self._policy_is_running.set()
                self._policy_tasks = policy_tasks
        except RuntimeError as e:
            self._fail(e)

    async def _clear_policies(self) -> None:
        self._policy_is_running.clear()
        if len(self._policy_tasks) > 0:
            tasks = self._policy_tasks
            self._policy_tasks.clear()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=False)
            except asyncio.CancelledError as e:
                self._logger.error(f"Cancelled due to {e}")
            except Exception as e:
                self._fail(e)

    async def policy_pause(self) -> None:
        try:
            with self._policy_lock:
                await self._clear_policies()

                pause_tasks = []
                for policy_pause_func, is_coroutine in self._builder.policy_pause_funcs:
                    if is_coroutine:
                        task = asyncio.create_task(policy_pause_func())
                    else:
                        task = asyncio.to_thread(policy_pause_func)
                    cancel_scope_task = run_coroutine_with_cancel(task, self._stopped_event.wait)
                    pause_tasks.append(cancel_scope_task)
                try:
                    await asyncio.gather(*pause_tasks, return_exceptions=False)
                except asyncio.CancelledError as e:
                    self._logger.error(f"Cancelled due to {e}")
                except Exception as e:
                    self._fail(e)
        except Exception as e:
            self._fail(e)

    def _fail(self, error: Exception) -> None:
        self._logger.exception(error)
        self._started = False
        self._stopped_event.set()

    async def clear(self) -> None:
        clear_tasks = []
        for clear_func, is_coroutine in self._builder.policy_pause_funcs:
            if is_coroutine:
                task = asyncio.create_task(clear_func())
            else:
                task = asyncio.to_thread(clear_func)
            cancel_scope_task = run_coroutine_with_cancel(task, self._stopped_event.wait)
            clear_tasks.append(cancel_scope_task)
        try:
            await asyncio.gather(*clear_tasks, return_exceptions=False)
        except asyncio.CancelledError as e:
            self._logger.error(f"Cancelled due to {e}")
        except Exception as e:
            self._fail(e)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await asyncio.to_thread(self.container.bootstrap)

    async def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stopped_event.set()
        await asyncio.to_thread(self.container.shutdown)
