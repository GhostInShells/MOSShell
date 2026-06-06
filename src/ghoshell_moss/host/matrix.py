import asyncio
from typing import Coroutine, Iterable, Type, Literal

from typing_extensions import Self

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Container, Provider

from ghoshell_moss.contracts import (
    Workspace, ConfigStore, WorkspaceYamlConfigStoreProvider,
    SystemPrompter, BaseSystemPrompter,
    ResourceStorageFactoryBootstrapper,
)
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.manifests import Manifests
from ghoshell_moss.core.blueprint.matrix import Matrix, Cell, MatrixLifecycleObject
from ghoshell_moss.core.blueprint.app import AppStore, AppInfo
from ghoshell_moss.core.blueprint.host import Mode
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.host.providers import (
    WorkspaceZenohProvider, HostLoggerProvider, ZenohTopicServiceProvider,
    HostSessionProvider,
)
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.message import unique_id
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.host.cell_discovery import CellDiscovery

depend_zenoh()
import zenoh
import contextlib
import logging
import threading
import psutil

__all__ = ['AppCell', 'HostCell', 'MatrixImpl']


class AppCell(Cell):

    def __init__(self, app: AppInfo, event: threading.Event):
        self.name = app.fullname
        self.description = app.description
        self.type = "app"
        self.where = app.work_directory
        self.workspace = self.where
        self._alive_event = event
        self._address = app.address

    @property
    def address(self) -> str:
        return self._address

    def is_alive(self) -> bool:
        return self._alive_event.is_set()


class HostCell(Cell):

    def __init__(self, mode: Mode, event: threading.Event, workspace: str):
        self.name = mode.name
        self.type = 'host'
        self.description = mode.description
        self.where = mode.file
        self.workspace = workspace
        self._alive_event = event

    def is_alive(self) -> bool:
        return self._alive_event.is_set()


class UnknownCell(Cell):
    """
    unknown cell
    """

    def __init__(self):
        self.name = 'unknown'
        self.type = 'unknown'
        self.description = ''
        self.where = ''
        self._address = 'unknown/' + unique_id()

    @property
    def address(self) -> str:
        return self._address

    def is_alive(self) -> bool:
        return False


class MatrixImpl(Matrix):

    def __init__(
            self,
            *,
            mode: Mode,
            env: Environment,
            app_store: AppStore,
            manifest: Manifests,
            workspace: Workspace,
            logger: LoggerItf | logging.Logger | None = None,
    ):
        env.bootstrap()
        self._env = env
        self.apps = app_store
        self._ctml_version_cache: dict[str, str] = {}
        self._current_mode: Mode = mode
        self._this_cell_address = env.cell_address
        self._manifests = manifest
        self._workspace = workspace
        self._session_scope = env.session_scope
        self._cell_discovery = CellDiscovery(session_scope=self._session_scope)

        # prepare cell and events
        # app cells 都是根据约定发现的, 由 host 进程管理的. 不会自动注册.
        cells: dict[str, Cell] = {}
        cell_alive_events: dict[str, threading.Event] = {}
        for app in self.apps.list_apps():
            is_alive = threading.Event()
            cell = AppCell(app, is_alive)
            cell_alive_events[cell.address] = is_alive
            cells[cell.address] = cell

        # prepare main cell
        event = threading.Event()
        main_cell = HostCell(self._current_mode, event, str(self.workspace.root().abspath()))
        cell_alive_events[main_cell.address] = event
        cells[main_cell.address] = main_cell
        self._main_cell = main_cell
        if self._this_cell_address == main_cell.address:
            self._this_cell = main_cell
        else:
            self._this_cell = cells.get(
                self._this_cell_address,
                UnknownCell(),
            )

        self._cells = cells
        self._cell_alive_events = cell_alive_events
        self._is_main = isinstance(self._this_cell, HostCell)
        self._logger: LoggerItf | logging.Logger | None = logger
        self._started = False
        self._channel_provider_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._exit_stack = contextlib.ExitStack()
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._log_prefix = f"<HostMatrix address={self._this_cell_address} session_scope={self.env.session_scope}>"
        self._task_group: set[asyncio.Task] = set()
        locker_name = '-'.join(['moss', 'cell', self._this_cell.type, self._this_cell.name])
        locker_name = locker_name.replace('.', '_')
        locker_name = locker_name.replace('/', '_')
        self._process_locker = self._workspace.lock(locker_name)
        self._process_locker_name = locker_name
        self._system_prompter = self._prepare_system_prompter()
        self._container = self._prepare_container()
        self._lifecycle_bound_objects_or_types: list[MatrixLifecycleObject | Type[MatrixLifecycleObject]] = []

        if isinstance(self._this_cell, UnknownCell):
            log = self._logger or self.env.logger
            log.warning(
                "%s cell address %r not found in known cells: %s, fallback to UnknownCell",
                self._log_prefix, self._this_cell_address, list(cells.keys()),
            )

    def _prepare_system_prompter(self) -> SystemPrompter:
        from ghoshell_moss.host.system_prompter import MossSystemPrompterImpl
        prompter = MossSystemPrompterImpl(
            description="MOSS system instruction — assembled from ctml, project, mode, static layers.",
        )
        prompter.with_prompter(
            MossSystemPrompter.CTML_SLOT,
            BaseSystemPrompter(
                own_instruction=self.ctml_instruction(),
                description="CTML grammar prompt for the current version.",
            ),
        )
        prompter.with_prompter(
            MossSystemPrompter.PROJECT_SLOT,
            BaseSystemPrompter(
                own_instruction=self.env.meta_config.system_prompt,
                description="Workspace root MOSS.md project instruction.",
            ),
        )
        prompter.with_prompter(
            MossSystemPrompter.MODE_SLOT,
            BaseSystemPrompter(
                own_instruction=self._current_mode.instruction,
                description=f"Mode '{self._current_mode.name}' instruction.",
            ),
        )
        return prompter

    def ctml_version(self) -> str:
        """返回当前环境中定义的 ctml version """
        return self._current_mode.ctml_version or self.env.meta_config.ctml_version

    def get_ctml_prompt(self, ctml_version: str | None = None) -> str | None:
        """在当前环境约定的 workspace 下寻找 ctml 指定版本. """
        ctml_version = ctml_version or self.ctml_version()
        if ctml_version not in self._ctml_version_cache:
            versions = self.manifests.ctml_versions()
            version_info = versions.get(ctml_version)
            if version_info is None:
                raise KeyError(f"ctml version {ctml_version} not found in manifests")
            self._ctml_version_cache[ctml_version] = version_info.file.read_text(encoding="utf-8")
        return self._ctml_version_cache[ctml_version]

    def ctml_instruction(self) -> str:
        ctml_version = self.ctml_version()
        return self.get_ctml_prompt(ctml_version)

    def _prepare_container(self) -> Container:
        container = Container(name=self._this_cell_address)
        container.set(Matrix, self)
        container.set(MatrixImpl, self)
        container.set(Environment, self.env)
        container.set(Mode, self._current_mode)
        container.set(Workspace, self._workspace)
        container.set(Manifests, self._manifests)
        # system prompter — 同时注册两个 key, 指向同一实例
        container.set(SystemPrompter, self._system_prompter)
        container.set(MossSystemPrompter, self._system_prompter)

        # 注册 manifest providers. 包含环境与模式的双重配置.
        for contract in self._manifests.providers():
            # register provider from manifest.contracts.
            # 可能会覆盖系统自身约定的 contract.
            container.register(contract.provider)

        # 按需注册 default provider. 由于这里没有显示声明, 所以肯定没有声明的方式好.
        for provider in self._default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        # 注册环境发现的所有资源.
        # todo, 未来可以简单实现一个 host manifests resource storage registry, 自己在 bootstrap 时从 manifests 拿东西.
        for resource_storage_manifest in self.manifests.resource_storage_manifests():
            storage_factory = resource_storage_manifest.get_sync()
            bootstrapper = ResourceStorageFactoryBootstrapper(storage_factory)
            container.add_bootstrapper(bootstrapper)

        return container

    def _default_providers(self) -> list[Provider]:
        # 注册 workspace zenoh provider.
        # 可以被环境覆盖.
        default_providers = []
        if self._is_main:
            default_providers.append(WorkspaceZenohProvider("zenoh_config_main.json5"))
        else:
            # All non-host cells (app, script, future) share the connector config.
            default_providers.append(WorkspaceZenohProvider("zenoh_config_cell.json5"))

        # 注册 configs
        default_providers.append(WorkspaceYamlConfigStoreProvider(
            *[info.config for info in self.manifests.configs().values()]
        ))
        # 注册 session.
        default_providers.append(HostSessionProvider())
        # 否则注册约定的日志模块, 但仍然可能被 contracts 覆盖.
        default_providers.append(HostLoggerProvider())

        # 注册 Topic Service.
        default_providers.append(ZenohTopicServiceProvider(
            session_scope=self.env.session_scope,
            cell_address=self._this_cell.address,
        ))
        return default_providers

    def moss_system_prompter(self) -> SystemPrompter:
        return self._system_prompter

    @property
    def this(self) -> Cell:
        return self._this_cell

    @property
    def env(self) -> Environment:
        return self._env

    def cell_env(self) -> dict[str, str]:
        """
        Cell 自身相关的环境变量.
        """
        # 做显式的声明, 方便了解底层逻辑.
        return self.env.dump_moss_env(
            with_os_env=False,
            for_child_process=False,
        )

    @property
    def mode(self) -> Mode:
        return self._current_mode

    @property
    def ghost_name(self) -> str | Literal['None']:
        return self.env.ghost_name or 'None'

    def list_cells(self) -> dict[str, Cell]:
        return self._cells

    @property
    def session(self) -> Session:
        return self._container.force_fetch(Session)

    @property
    def manifests(self) -> Manifests:
        return self._manifests

    @property
    def container(self) -> IoCContainer:
        return self._container

    def provide_channel(
            self,
            channel: Channel,
            *,
            address: str | None = None,
    ) -> asyncio.Future[None]:
        self._check_running()
        # cancel providing channel
        cancelling = None
        if self._channel_provider_task is not None and not self._channel_provider_task.done():
            self._channel_provider_task.cancel()
            cancelling = self._channel_provider_task
            self._channel_provider_task = None

        provider_address = address or self._this_cell.address

        async def _providing():
            nonlocal cancelling, channel
            if cancelling is not None:
                try:
                    await cancelling
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error("%s close channel provider exception: %s", self._log_prefix, e)
            provider = ZenohChannelProvider(
                address=provider_address,
                session_scope=self.session.session_scope,
                container=self._container,
                zenoh_session=self._container.force_fetch(zenoh.Session)
            )
            await provider.arun_until_closed(channel)

        self._channel_provider_task = self._event_loop.create_task(_providing())
        return self._channel_provider_task

    def channel_proxy(
            self,
            address: str,
            name: str,
            description: str = '',
            id: str | None = None,
            only_allowed_in_host_cell: bool = True,
    ) -> ZenohProxyChannel:
        self._check_running()
        if only_allowed_in_host_cell and not self._is_main:
            raise RuntimeError(f"Only allowed in main cell type: {self.this.type}")
        return ZenohProxyChannel(
            address=address,
            session_scope=self.session.session_scope,
            name=name,
            description=description,
            zenoh_session=self._container.force_fetch(zenoh.Session),
            uid=id,
        )

    @property
    def logger(self) -> LoggerItf:
        if self._logger is not None:
            return self._logger
        return self.env.logger

    @property
    def configs(self) -> ConfigStore:
        return self.container.force_fetch(ConfigStore)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def is_running(self) -> bool:
        return self._started and not (self._closing_event.is_set() or self._closed_event.is_set())

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Matrix is not running")

    def is_host_running(self) -> bool:
        if self._is_main:
            return self.is_running()
        else:
            return self._main_cell.is_alive()

    def close(self) -> None:
        self._closing_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        return self._closed_event.wait_sync(timeout)

    def create_task(
            self,
            cor: Coroutine,
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        self._check_running()

        async def _wait_done():
            nonlocal stop_matrix_on_error, cor
            try:
                await cor
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error("%s receive exception on inner task %s: %r", self._log_prefix, name, e)
                if stop_matrix_on_error:
                    self.close()
            finally:
                self.logger.info("%s inner task %s done", self._log_prefix, name)

        task = self._event_loop.create_task(_wait_done())
        self._add_task(task)
        return task

    def register_lifecycle_objects(self, obj: MatrixLifecycleObject) -> None:
        if self.is_running():
            raise RuntimeError(f"Matrix is already running")
        self._lifecycle_bound_objects_or_types.append(obj)

    def _add_task(self, task: asyncio.Task) -> None:
        self._task_group.add(task)
        task.add_done_callback(self._remove_task)

    def _remove_task(self, task: asyncio.Task) -> None:
        self._task_group.discard(task)

    @contextlib.contextmanager
    def _ensure_container_lifecycle_ctx_manager(self):
        # 启动 container.
        self._container.bootstrap()
        try:
            for config_info in self.manifests.configs().values():
                self.configs.set_config(config_info.config)
                self.configs.get_or_create(config_info.config)
            yield
        finally:
            self._container.shutdown()

    @contextlib.contextmanager
    def _ensure_process_locker_ctx_manager(self):
        if not self._process_locker.acquire(3.0):
            raise RuntimeError(f"Matrix failed to lock {self._process_locker_name}")
        try:
            yield
        finally:
            self._process_locker.release()

    @contextlib.asynccontextmanager
    async def _ensure_channel_provider_task_cancelled_ctx_manager(self):
        try:
            yield
        finally:
            if self._channel_provider_task is not None:
                task = self._channel_provider_task
                self._channel_provider_task = None
                if not task.done():
                    try:
                        task.cancel()
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.exception(
                            "%s failed to cancel channel provider: %s",
                            self._log_prefix, e,
                        )

    @contextlib.asynccontextmanager
    async def _ensure_task_group_canceled_ctx_manager(self):
        try:
            yield
        finally:
            tasks = self._task_group.copy()
            self._task_group.clear()
            wait_done = []
            for t in tasks:
                if not t.done():
                    t.cancel()
                wait_done.append(t)
            await asyncio.gather(*wait_done, return_exceptions=True)

    async def _ensure_parent_process_exists(self) -> None:
        if self.env.parent_pid == 0:
            return
        try:
            parent = psutil.Process(int(self.env.parent_pid))
        except (ValueError, TypeError, psutil.NoSuchProcess):
            return

        while not self._closing_event.is_set():
            if not parent.is_running():
                self.close()
                break
            await asyncio.sleep(2)

    @contextlib.asynccontextmanager
    async def _ensure_parent_process_exists_ctx_manager(self):
        task = asyncio.create_task(self._ensure_parent_process_exists())
        try:
            yield
        finally:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def _lifecycle_level_contracts(self) -> Iterable[Type[MatrixLifecycleObject]]:
        """
        注册抽象里定义好的, 基于约定发现的特殊抽象类型.
        """
        # 暂时不做隐式绑定.
        yield from []

    def _session_communication_bus_ctx_manager(self):
        """管理 session 事件总线的生命周期. """
        # 未来考虑把 zenoh 完全屏蔽到 session 内侧. 暂时无法做到.
        zenoh_session = self._container.force_fetch(zenoh.Session)
        self._exit_stack.enter_context(zenoh_session)
        self._exit_stack.enter_context(
            self._cell_discovery.discover_cells(
                zenoh_session, self._cells, self._cell_alive_events,
                this_address=self._this_cell.address,
            )
        )
        self._exit_stack.enter_context(
            self._cell_discovery.declare_this_cell(zenoh_session, self._this_cell.address)
        )

    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        self._check_running()
        for registered in self._lifecycle_bound_objects_or_types:
            if obj is registered:
                return
        self._lifecycle_bound_objects_or_types.append(obj)
        await self._async_exit_stack.enter_async_context(obj)
        self._logger.info("%s add lifecycle object to exit stack %s", self._log_prefix, obj)

    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        if self._closing_event.is_set():
            raise RuntimeError(f"Matrix already closing")
        if self.is_running():
            self._event_loop.create_task(self.add_lifecycle_object(obj))
            self._logger.info("%s try to create task bind lifecycle object %s", self._log_prefix, obj)
        else:
            self._lifecycle_bound_objects_or_types.append(obj)
            self._logger.info("%s register lifecycle object %s", self._log_prefix, obj)

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("Matrix already started")
        self._started = True
        # 显式启动 ioc 容器. 同步生命周期启动. 因为 matrix 本身是进程级实例, 所以可以阻塞.
        self._event_loop = asyncio.get_running_loop()
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(self._ensure_process_locker_ctx_manager())
        self._exit_stack.enter_context(self._ensure_container_lifecycle_ctx_manager())
        # 实现 Matrix.session 的通讯总线同步启动部分.
        self._session_communication_bus_ctx_manager()

        # IoC 容器已启动，探查是否注册了 LoggerItf，有则覆写 _logger。
        logger = self._container.get(LoggerItf)
        if logger is not None:
            self._logger = logger

        # 启动 stack.
        try:
            await self._async_exit_stack.__aenter__()
            # 确认最后的 channel provider 一定会被 cancel.
            await self._async_exit_stack.enter_async_context(self._ensure_channel_provider_task_cancelled_ctx_manager())
            topic_service = self._container.force_fetch(TopicService)
            # ensure topic service lifecycle
            await self._async_exit_stack.enter_async_context(topic_service)
            # 完成 session 的异步启动逻辑.
            session = self._container.force_fetch(Session)
            await self._async_exit_stack.enter_async_context(session)
            # 完成启动后, 进入到关联依赖启动. 启动成功才进入到核心生命周期启动.
            lifecycle_objects = []
            if len(self._lifecycle_bound_objects_or_types) > 0:
                for lifecycle in self._lifecycle_bound_objects_or_types:
                    if isinstance(lifecycle, type):
                        self.logger.info("%s try to find lifecycle type: %s", self._log_prefix, lifecycle)
                        lifecycle_obj = self._container.get(lifecycle)
                    else:
                        # todo: 暂时不做类型检查, 交给 AI 在合适的时候做. 或者保留 todo, 报错时可以看到这里源码.
                        lifecycle_obj = lifecycle
                    lifecycle_objects.append(lifecycle_obj)
                    if lifecycle_obj is not None:
                        self.logger.info(
                            "%s bootstrap bound lifecycle object: %s",
                            self._log_prefix, lifecycle,
                        )
                        await self._async_exit_stack.enter_async_context(lifecycle_obj)
            self._lifecycle_bound_objects_or_types = lifecycle_objects
            # 进入到根据约定可以做绑定的生命周期对象.
            for lifecycle_contract in self._lifecycle_level_contracts():
                if bound := self._container.get(lifecycle_contract):
                    self.logger.info(
                        "%s bootstrap bound lifecycle contract: %s",
                        self._log_prefix, lifecycle_contract,
                    )
                    await self._async_exit_stack.enter_async_context(bound)

            await self._async_exit_stack.enter_async_context(self._ensure_task_group_canceled_ctx_manager())
            await self._async_exit_stack.enter_async_context(self._ensure_parent_process_exists_ctx_manager())
            if event := self._cell_alive_events.get(self._this_cell_address):
                event.set()
            self.logger.info("%s initialized with env: %s", self._log_prefix, self.env.dump_moss_env(
                with_os_env=False,
            ))
            return self
        except Exception as e:
            self.logger.exception("%s failed to start on exception: %s", self._log_prefix, e)
            raise e
        finally:
            self.logger.info("%s initialized", self._log_prefix)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_val is not None:
                if isinstance(exc_val, KeyboardInterrupt):
                    self.logger.info("%s stop on keyboard interrupt", self._log_prefix)
                elif isinstance(exc_val, asyncio.CancelledError):
                    self.logger.info("%s stop on cancelled", self._log_prefix)
                elif isinstance(exc_val, FatalError):
                    self.logger.exception("%s stop on fatal error: %s", self._log_prefix, exc_val)
                else:
                    self.logger.exception("%s stop on unknown error: %s", self._log_prefix, exc_val)

            if event := self._cell_alive_events.get(self._this_cell_address):
                event.clear()

            # exit all the stack
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self.logger.exception("%s failed to aexit on exception: %s", self._log_prefix, e)
        finally:
            self._closing_event.set()
            self._closed_event.set()
            # 结束同步运行逻辑.
            self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
