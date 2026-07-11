import asyncio
import os
import signal
from pathlib import Path
from typing import Coroutine, Iterable, Type, Literal, Callable
from typing_extensions import Self
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Container, Provider

from ghoshell_moss.contracts import (
    Workspace, ConfigStore, WorkspaceYamlConfigStoreProvider,
    SystemPrompter, BaseSystemPrompter,
    ResourceStorageFactoryBootstrapper, LocalWorkspace,
)
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.manifests import Manifests
from ghoshell_moss.core.blueprint.matrix import Matrix, MatrixLifecycleObject
# from ghoshell_moss.core.blueprint.host import Mode
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.blueprint.cell import Cell as MossCell, CellNetwork
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.host.providers import (
    WorkspaceZenohProvider, HostLoggerProvider, ZenohTopicServiceProvider,
    HostSessionProvider,
)
# from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.core.helpers import ThreadSafeEvent

import concurrent.futures
import contextlib
import logging
import psutil

__all__ = ['MatrixImpl']


class MossSystemPrompterImpl(BaseSystemPrompter, MossSystemPrompter):
    """MOSS 约定的 SystemPrompter 默认实现.

    BaseSystemPrompter 提供 tree 存储 + instruction 组装.
    MossSystemPrompter 提供四个命名访问器 (ctml/project/mode/static).
    二者通过钻石继承组合, 注册为 SystemPrompter 和 MossSystemPrompter 两个 IoC key.
    """
    pass


class MatrixImpl(Matrix):

    def __init__(
            self,
            *,
            # 当前启动的 cell.
            cell: MossCell,
            # mode: Mode,
            env: Environment,
            manifest: Manifests,
            logger: LoggerItf | logging.Logger | None = None,
    ):
        env.seal()
        self._env = env
        self._ctml_version_cache: dict[str, str] = {}
        # self._current_mode: Mode = mode
        self._manifests = manifest
        self._workspace = env.workspace
        self._session_scope = env.network_scope
        # 准备改进 cell, 使之具备完整的路径.
        # self._curr_cell = prepare_cell_with_env(cell, env)
        # 准备一个 cell 的 workspace.
        self._cell_workspace = LocalWorkspace(Path(self._curr_cell.launcher.cwd))

        self._cell_address = cell.address
        self._is_host = cell.meta.type == CellType.host.value

        self._logger: LoggerItf | logging.Logger | None = logger or env.logger
        self._started = False
        self._config_change_callbacks: dict[str, list[Callable[[], None]]] = {}
        self._channel_provider_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._exit_stack = contextlib.ExitStack()
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._log_prefix = f"<HostMatrix address={self._cell_address} session_scope={self.env.network_scope}>"
        self._task_group: set[asyncio.Task] = set()
        self._container = self._prepare_container()

        normalize_address = cell.normalized_address()
        locker_name = f"moss_{normalize_address}"
        self._process_locker_name = locker_name
        self._process_locker = self._workspace.lock(locker_name)

        self._system_prompter = self._prepare_system_prompter()
        self._lifecycle_bound_objects_or_types: list[MatrixLifecycleObject | Type[MatrixLifecycleObject]] = []
        self._refresh_future: concurrent.futures.Future | None = None

        # --- process re tasks --- #
        self._process_reclaim_tasks: set[asyncio.Task] = set()

    def _prepare_system_prompter(self) -> SystemPrompter:
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
                own_instruction=self.env.moss_meta.system_prompt,
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
        return self._current_mode.ctml_version or self.env.moss_meta.ctml_version

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

    @property
    def cells(self) -> CellNetwork:
        pass

    @property
    def cell_workspace(self) -> Workspace:
        return self._cell_workspace

    def ctml_instruction(self) -> str:
        ctml_version = self.ctml_version()
        return self.get_ctml_prompt(ctml_version)

    def _prepare_container(self) -> Container:
        # 准备容器.
        container = Container(name=self._curr_cell.address)
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
            # 只有没绑定, 才会绑定默认的 provider.
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
        if self._is_host:
            default_providers.append(WorkspaceZenohProvider("zenoh_config_main.json5"))
        else:
            # All non-host cells (app, script, future) share the connector config.
            default_providers.append(WorkspaceZenohProvider("zenoh_config_cell.json5"))

        # 注册 configs — 仅类型注册（is_override=False），文件持久化
        # 实例覆盖（is_override=True）在 lifecycle 中通过 set_config 内存写入
        default_providers.append(WorkspaceYamlConfigStoreProvider(
            *[info.config for info in self.manifests.configs().values() if not info.is_override],
            on_save=self._on_config_saved,
        ))
        # 注册 session.
        default_providers.append(HostSessionProvider())
        # 否则注册约定的日志模块, 但仍然可能被 contracts 覆盖.
        default_providers.append(HostLoggerProvider())

        # 注册 Topic Service.
        default_providers.append(ZenohTopicServiceProvider(
            session_scope=self.env.network_scope,
            cell_address=self._curr_cell.address,
        ))
        return default_providers

    def moss_system_prompter(self) -> SystemPrompter:
        return self._system_prompter

    @property
    def this(self) -> MossCell:
        return self._curr_cell

    @property
    def env(self) -> Environment:
        return self._env

    def cell_env(self) -> dict[str, str]:
        """
        Cell 自身相关的环境变量.
        可以用于 debug.
        """
        # 做显式的声明, 方便了解底层逻辑.
        return self.env.runtime_scope.dump_runtime_scope()

    # @property
    # def mode(self) -> Mode:
    #     return self._current_mode

    # def list_cells(self) -> dict[str, Cell]:
    #     return self._cells
    #
    # async def alist_cells(self) -> dict[str, Cell]:
    #     """异步从网络查询全量 cell 状态。
    #
    #     通过 Zenoh wildcard get 查询所有 per-cell queryable，
    #     能响应者即为在线。当前仅触发查询、返回本地缓存——
    #     Cell 不再携带运行时状态字段，存活判定由响应本身表达。
    #     """
    #     session = self._container.force_fetch(zenoh.Session)
    #     await asyncio.to_thread(self._cell_discovery.query_cells, session)
    #     return self._cells

    @property
    def session(self) -> Session:
        return self._container.force_fetch(Session)

    @property
    def manifests(self) -> Manifests:
        return self._manifests

    @property
    def container(self) -> IoCContainer:
        return self._container

    @property
    def logger(self) -> logging.Logger:
        if self._logger is not None:
            return self._logger
        # 使用 env logger 兜底.
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
        """判断 host (主 cell) 是否在运行中。

        read_scope_meta() 默认 alive_only=True，内部完成 PID 验活。
        文件不存在或 PID 已死均视为 host 不在运行。
        """
        if self._is_host:
            return self.is_running()
        return psutil.pid_exists(self._env.runtime_scope.host_pid)

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

    async def spawn(
            self,
            *args: str,
            cell_address: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
    ) -> asyncio.subprocess.Process:
        self._check_running()
        env = self.env.dump_cell_env(
            parent_cell_address=self.this.address,
            cell_address=cell_address or '',
            with_os_env=True,
        )
        if extra_env is not None:
            env.update(extra_env)
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )

        # 治理与收尸守护进程
        async def _async_reclaim():
            try:
                # 【核心改变】安全高效地阻塞在这里，静听死讯，零CPU消耗
                # 除非子进程挂了，或者这个 Task 被容器关闭逻辑外部 Cancel 掉，否则绝不醒来
                await proc.wait()

            except asyncio.CancelledError:
                # 进入这里，说明子进程还没死，但是整个容器（Cell）要关机了（Task被外部取消）
                # 这时触发你设计的“先礼后兵”主动治理机制
                if proc.returncode is None:
                    try:
                        # 礼：发送 SIGINT
                        proc.send_signal(signal.SIGINT)
                        try:
                            # 给它 2 秒延时善后
                            await asyncio.wait_for(proc.wait(), timeout=2.0)
                            return
                        except asyncio.TimeoutError:
                            pass

                        # 兵：强行抹杀
                        if proc.returncode is None:
                            proc.kill()
                            await proc.wait()
                    except ProcessLookupError:
                        pass
            finally:
                # 无论是自己死掉触发了 return，还是被强行杀掉
                # 最终一定会走到这里，彻底在内存中抹除它的痕迹
                # 如果外界没有强引用 proc，它在这里直接被 GC 回收
                pass

        task = self._event_loop.create_task(_async_reclaim())
        self._process_reclaim_tasks.add(task)
        task.add_done_callback(lambda _: self._process_reclaim_tasks.discard(task))
        return proc

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
                if config_info.is_override:
                    self.configs.set_config(config_info.config)
                else:
                    self.configs.get_or_create(config_info.config)
            yield
        finally:
            self._container.shutdown()

    @contextlib.contextmanager
    def _ensure_runtime_scope_files_lifecycle(self):
        try:
            if self._is_host:
                # 写入 runtime scope.
                self.env.runtime_scope.host_pid = os.getpid()
                # todo
                self.env.runtime_scope.write_to_workspace(self.env.workspace)
                # 删除所有运行时 scope 文件.
                self._clear_runtime_cell_and_files()
            # 记录运行状态.
            self._curr_cell.write_runtime_file(self.env.runtime_registry_dir)
            self.logger.info("%s write runtime cell file", self._log_prefix)
            yield
        finally:
            file = self._curr_cell.runtime_filepath(self.env.runtime_registry_dir)
            file.unlink()
            if self._is_host:
                # 删除 scope file.
                self.env.runtime_scope.delete_from_workspace(self.env.workspace)
            self._clear_runtime_cell_and_files()
            self.logger.info("%s clear runtime cell file", self._log_prefix)

    def _clear_runtime_cell_and_files(self):
        for cell in MossCell.find_runtime_cells(self.env.runtime_registry_dir):
            try:
                pid = int(cell.status.pid)
                if pid > 0 and psutil.pid_exists(pid):
                    os.kill(int(cell.status.pid), signal.SIGTERM)
            except AttributeError:
                continue
            finally:
                # 删除所有 scope 文件.
                file = cell.runtime_filepath(self.env.runtime_registry_dir)
                file.unlink()

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

    @contextlib.asynccontextmanager
    async def _nursery_spawned_process_ctx_manager(self):
        """确保所有创建的子进程不要变成僵尸. """
        try:
            yield
        finally:
            # 1. 浅拷贝一份当前还在运行的收尸 Task，不要直接 clear()
            # 留着它们，让 spawn 里的 done_callback 自然地去 discard 它们
            tasks = list(self._process_reclaim_tasks)

            if tasks:
                self.logger.info(f"Container closing. Reclaiming {len(tasks)} active process monitors...")

                # 2. 批量发出 Cancel 信号，让所有 Task 弹入 CancelledError 分支，并发触发“先礼后兵”
                for t in tasks:
                    if not t.done():
                        t.cancel()

                # 3. 【核心改变】必须使用 await！真正等待所有收尸 Task 执行完它们的善后逻辑
                # return_exceptions=True 可以确保某个子进程卡死或报错时，不影响其他子进程的收割
                _ = await asyncio.gather(*tasks, return_exceptions=True)

                self.logger.debug("All spawned processes have been safely reclaimed.")

    def _lifecycle_level_contracts(self) -> Iterable[Type[MatrixLifecycleObject]]:
        """
        注册抽象里定义好的, 基于约定发现的特殊抽象类型.
        """
        # 暂时不做隐式绑定.
        yield from []

    def _cell_info(self) -> dict:
        """构建当前 cell 的 info dict，用于 queryable announce。"""
        return {
            "address": self._this_cell.address,
            "name": self._this_cell.name,
            "type": self._this_cell.type,
            "where": self._this_cell.where,
            "workspace": self._this_cell.interpreter,
            "description": self._this_cell.description,
        }

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

            # ── session metadata 写 — 仅 main ──
            if self._is_host:
                # 写入 runtime scope.
                self._env.runtime_scope.host_pid = os.getpid()
                # todo
                self._env.runtime_scope.write_to_workspace(self._env.workspace)

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
            # 管理最后的子进程退出.
            await self._async_exit_stack.enter_async_context(self._nursery_spawned_process_ctx_manager())

            self.logger.info("%s initialized with env: %s", self._log_prefix, self.env.dump_cell_env(
                with_os_env=False,
            ))
            self._curr_cell.status.state = 'alive'

            # 最终才开始整理文件.
            self._exit_stack.enter_context(self._ensure_runtime_scope_files_lifecycle())
            return self
        except Exception as e:
            self.logger.exception("%s failed to start on exception: %s", self._log_prefix, e)
            # 记录异常.
            self._curr_cell.status.state = 'stopped'
            self._curr_cell.status.error = str(e)
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
                    self._curr_cell.status.error = str(exc_val)
                    self.logger.exception("%s stop on fatal error: %s", self._log_prefix, exc_val)
                else:
                    self._curr_cell.status.error = str(exc_val)
                    self.logger.exception("%s stop on unknown error: %s", self._log_prefix, exc_val)
            # exit all the stack
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self.logger.exception("%s failed to aexit on exception: %s", self._log_prefix, e)
        finally:
            self._closing_event.set()
            self._closed_event.set()
            # 标记运行结束.
            self._curr_cell.status.state = 'stopped'
        # 结束同步运行逻辑.
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def _write_session_metadata(self, session: Session) -> None:
        """写入 session metadata + 追加 SessionRecord — 仅 _is_main 调用。"""
        from datetime import datetime, timezone
        from ghoshell_moss.core.blueprint.session import SessionMetadata, SessionRecord

        now = datetime.now(timezone.utc).isoformat()
        meta = SessionMetadata(
            session_id=self.session_id,
            session_scope=self.network_scope,
            mode_name=self.mode_name,
            ghost_name=self.ghost_name,
            host_cell_address=self._this_cell_address,
            host_pid=os.getpid(),
            created_at=now,
        )
        session.storage.write_yaml("meta", meta)
        record = SessionRecord(
            session_id=self.session_id,
            created_at=now,
        )
        session.scope_storage.append_model("sessions", record)
