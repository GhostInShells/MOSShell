import asyncio
from typing import Coroutine

from typing_extensions import Self

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Container, Provider

from ghoshell_moss import TopicService
from ghoshell_moss.contracts import Workspace, ConfigStore, WorkspaceYamlConfigStoreProvider
from ghoshell_moss.core.concepts.session import Session
from ghoshell_moss.host.abcd.manifests import Manifests
from ghoshell_moss.host.abcd.matrix import Matrix, Cell
from ghoshell_moss.host.abcd.app import AppStore, AppInfo
from ghoshell_moss.host.abcd.host_design import Mode
from ghoshell_moss.host.abcd.environment import Environment, DEFAULT_CELL_ADDRESS
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.host.providers import (
    WorkspaceZenohProvider, WorkspaceLoggerProvider, ZenohTopicServiceProvider,
    WorkspaceSessionProvider,
)
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh
import contextlib
import logging
import threading

__all__ = ['AppCell', 'MossModeCell', 'MatrixImpl']


class AppCell(Cell):

    def __init__(self, app: AppInfo, event: threading.Event):
        self.name = app.fullname
        self.description = app.description
        self.docstring = app.docstring
        self.type = "app"
        self.where = app.work_directory
        self._alive_event = event
        self._address = app.address

    @property
    def address(self) -> str:
        return self._address

    def is_alive(self) -> bool:
        return self._alive_event.is_set()


class MossModeCell(Cell):

    def __init__(self, mode: Mode, event: threading.Event):
        self.name = mode.name
        self.type = 'main'
        self.description = mode.description
        self.docstring = mode.description
        self.where = mode.file
        self._alive_event = event

    @property
    def address(self) -> str:
        return DEFAULT_CELL_ADDRESS

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
        self.docstring = ''
        self.where = ''
        self._address = 'unknown/' + uuid()

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
        self.env = env
        self.apps = app_store
        self._current_mode = mode
        self._cell_address = env.cell_address
        self._manifest = manifest
        self._workspace = workspace
        self._current_mode = mode
        self._session_id = env.session_scope

        # prepare cell and events
        cells: dict[str, Cell] = {}
        cell_events: dict[str, threading.Event] = {}
        for app in self.apps.list_apps():
            is_alive = threading.Event()
            cell = AppCell(app, is_alive)
            cell_events[cell.address] = is_alive
            cells[cell.address] = cell

        event = threading.Event()
        main_cell = MossModeCell(self._current_mode, event)
        self._main_cell = main_cell
        cell_events[main_cell.address] = event
        cells[main_cell.address] = main_cell

        self._cells = cells
        self._cell_events = cell_events
        # 其实不会有 unknown, 不过开发测试阶段, 做一个兜底.
        self._this_cell = cells.get(
            self._cell_address,
            UnknownCell(),
        )
        self._is_main = self._this_cell.type == 'main'
        self._logger: LoggerItf | logging.Logger | None = logger
        self._container = self._prepare_container()
        self._started = False
        self._channel_provider_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._exit_stack = contextlib.ExitStack()
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._log_prefix = f"<HostMatrix address={self._cell_address} session_id={self.env.session_scope}>"
        self._task_group: set[asyncio.Task] = set()
        locker_name = '-'.join(['moss', 'cell', self._this_cell.type, self._this_cell.name])
        locker_name = locker_name.replace('.', '_')
        locker_name = locker_name.replace('/', '_')
        self._process_locker = self._workspace.lock(locker_name)
        self._process_locker_name = locker_name

    def _prepare_container(self) -> Container:
        container = Container(name=self._cell_address)
        container.set(Matrix, self)
        container.set(MatrixImpl, self)
        container.set(Environment, self.env)
        container.set(Workspace, self._workspace)
        container.set(Manifests, self._manifest)
        container.set(Cell, self._this_cell)

        # 注册 manifest providers. 包含环境与模式的双重配置.
        for contract in self._manifest.providers():
            # register provider from manifest.contracts.
            # 可能会覆盖系统自身约定的 contract.
            container.register(contract.provider)

        # 按需注册 default provider. 由于这里没有显示声明, 所以肯定没有声明的方式好.
        for provider in self._default_providers():
            if container.bound(provider.contract()):
                continue
            container.register(provider)

        if self._logger is not None:
            # 替换掉注册的.
            container.set(LoggerItf, self._logger)
        return container

    def _default_providers(self) -> list[Provider]:
        # 注册 workspace zenoh provider.
        # 可以被环境覆盖.
        default_providers = []
        if self._is_main:
            default_providers.append(WorkspaceZenohProvider("zenoh_config_main.json5"))
        elif self._this_cell.type == 'app':
            default_providers.append(WorkspaceZenohProvider("zenoh_config_app.json5"))
        else:
            raise RuntimeError(f"Unknown cell type: {self._this_cell.type}")

        # 注册 configs
        default_providers.append(WorkspaceYamlConfigStoreProvider(
            *[info.config for info in self.manifests.configs().values()]
        ))
        # 注册 session.
        default_providers.append(WorkspaceSessionProvider(session_scope=self.env.session_scope))
        # 否则注册约定的日志模块, 但仍然可能被 contracts 覆盖.
        default_providers.append(WorkspaceLoggerProvider(self._this_cell.log_name))

        # 注册 Topic Service.
        default_providers.append(ZenohTopicServiceProvider(
            session_scope=self.env.session_scope,
            cell_address=self._this_cell.address,
        ))
        return default_providers

    @property
    def this(self) -> Cell:
        return self._this_cell

    def cell_env(self) -> dict[str, str]:
        return self.env.dump_moss_env(with_os_env=False, for_child_process=False)

    @property
    def mode(self) -> str:
        return self._current_mode.name

    def list_cells(self) -> dict[str, Cell]:
        return self._cells

    @property
    def session(self) -> Session:
        return self._container.force_fetch(Session)

    @property
    def manifests(self) -> Manifests:
        return self._manifest

    @property
    def container(self) -> IoCContainer:
        return self._container

    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        self._check_running()
        # cancel providing channel
        cancelling = None
        if self._channel_provider_task is not None and not self._channel_provider_task.done():
            self._channel_provider_task.cancel()
            cancelling = self._channel_provider_task
            self._channel_provider_task = None

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
                node_name=self._this_cell.address,
                session_id=self.env.session_scope,
                container=self._container,
            )
            await provider.arun_until_closed(channel)

        self._channel_provider_task = self._event_loop.create_task(_providing())
        return self._channel_provider_task

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self._container.get(LoggerItf)
            if self._logger is None:
                self._logger = logging.getLogger(self._this_cell.log_name)
        return self._logger

    @property
    def configs(self) -> ConfigStore:
        return self.container.force_fetch(ConfigStore)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def topics(self) -> TopicService:
        self._check_running()
        topics = self.container.force_fetch(TopicService)
        return topics

    def is_running(self) -> bool:
        return self._started and not (self._closing_event.is_set() or self._closed_event.is_set())

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Matrix is not running")

    def is_moss_running(self) -> bool:
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

    def create_task(self, cor: Coroutine) -> asyncio.Task:
        self._check_running()
        task = self._event_loop.create_task(cor)
        self._add_task(task)
        return task

    def _add_task(self, task: asyncio.Task) -> None:
        self._task_group.add(task)
        task.add_done_callback(self._remove_task)

    def _remove_task(self, task: asyncio.Task) -> None:
        self._task_group.discard(task)

    @contextlib.contextmanager
    def _ensure_container_lifecycle_ctx_manager(self):
        self._container.bootstrap()
        try:
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

    @contextlib.contextmanager
    def _this_liveness_ctx_managers(self, session: zenoh.Session):
        # 实际上是同步调用逻辑.
        key_expr = self._moss_node_liveness_key_expr(self._this_cell.address)
        self_liveness = session.liveliness().declare_token(key_expr)
        try:
            yield
        finally:
            self_liveness.undeclare()

    def _moss_node_liveness_key_expr(self, address: str) -> str:
        return f"MOSS/{self._session_id}/cell/{address}"

    @contextlib.contextmanager
    def _all_cell_liveness_check_ctx_manager(self, session: zenoh.Session):
        if session.is_closed():
            raise RuntimeError(f"Matrix is not running, zenoh session is closed")
        subscribers = []
        for address, cell in self._cells.items():
            if address == self._this_cell.address:
                # 不监听自己.
                continue
            event = self._cell_events[address]
            sub = self._register_cell_liveness_listener(session, address, event)
            subscribers.append(sub)
        try:
            yield
        finally:
            for sub in subscribers:
                if not session.is_closed():
                    sub.undeclare()

    def _register_cell_liveness_listener(
            self,
            session: zenoh.Session,
            address: str,
            event: threading.Event,
    ) -> zenoh.Subscriber:
        key_expr = self._moss_node_liveness_key_expr(address)

        def _on_liveness_sample(sample: zenoh.Sample) -> None:
            nonlocal key_expr, event
            if sample.kind == zenoh.SampleKind.PUT:
                event.set()
            else:
                event.clear()

        return session.liveliness().declare_subscriber(key_expr, _on_liveness_sample)

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

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("Matrix already started")
        self._started = True
        # 显式启动 ioc 容器. 同步生命周期启动. 因为 matrix 本身是进程级实例, 所以可以阻塞.
        self._event_loop = asyncio.get_running_loop()
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(self._ensure_process_locker_ctx_manager())
        self._exit_stack.enter_context(self._ensure_container_lifecycle_ctx_manager())
        # 显式声明 zenoh session 生命周期, 不在 container 里 bootstrap 了.
        zenoh_session = self._container.force_fetch(zenoh.Session)
        self._exit_stack.enter_context(zenoh_session)
        self._exit_stack.enter_context(self._all_cell_liveness_check_ctx_manager(zenoh_session))
        self._exit_stack.enter_context(self._this_liveness_ctx_managers(zenoh_session))
        # 启动 stack.
        try:
            await self._async_exit_stack.__aenter__()
            # 确认最后的 channel provider 一定会被 cancel.
            await self._async_exit_stack.enter_async_context(self._ensure_channel_provider_task_cancelled_ctx_manager())
            topic_service = self._container.force_fetch(TopicService)
            # ensure topic service lifecycle
            await self._async_exit_stack.enter_async_context(topic_service)
            await self._async_exit_stack.enter_async_context(self._ensure_task_group_canceled_ctx_manager())
            if event := self._cell_events.get(self._cell_address):
                event.set()
            return self
        except Exception as e:
            self.logger.exception("%s failed to start on exception: %s", self._log_prefix, e)
            raise e

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

            if event := self._cell_events.get(self._cell_address):
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
