import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.concepts.channel import (
    Channel, ChannelFullPath, ChannelMeta, ChannelCtx, ChannelPaths,
)
from ghoshell_moss.core.concepts.runtime import AbsChannelRuntime, AbsChannelTreeRuntime
from ghoshell_moss.core.concepts.command import (
    BaseCommandTask, Command, CommandMeta, CommandTask, CommandWrapper,
    CommandUniqueName,
)
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

from .connection import Connection, ConnectionClosedError, ConnectionNotAvailable
from .protocol import (
    ChannelEvent,
    ChannelMetaUpdateEvent,
    ClearEvent,
    CommandCallEvent,
    CommandDoneEvent,
    CreateSessionEvent,
    ReconnectSessionEvent,
    SessionCreatedEvent,
    SyncChannelMetasEvent,
)

__all__ = ["DuplexChannelRuntime", "DuplexChannelProxy", ]

from ghoshell_moss.core.concepts.states import BaseStateStore, StateStore, State

"""
DuplexChannel Proxy 一侧的实现, 
todo: 全部改名为 Proxy 
"""


class DuplexChannelContext:
    """
    创建一个 Context 对象, 是所有 Duplex Channel Runtimes 共同依赖的.
    """

    def __init__(
            self,
            *,
            name: str,
            connection: Connection,
            container: Optional[IoCContainer] = None,
    ):
        self.root_name = name
        """根节点的名字. 这个名字可能和远端的 channel 根节点不一样. """
        self.remote_root_name = ""
        """远端的 root channel 名字"""

        self._wait_reconnect_interval = 0.2

        self.container = Container(parent=container, name="duplex channel context:" + self.root_name)
        self.connection = connection
        """双工连接本身."""

        self.session_id: str = ""
        self.provider_meta_map: dict[ChannelFullPath, ChannelMeta] = {}
        """所有远端上传的 metas. """

        self._starting = False
        self._started = asyncio.Event()

        self.stop_event = ThreadSafeEvent()
        """全局的 stop event, 会中断所有的子节点"""

        # runtime
        self._connected_event = asyncio.Event()
        """标记是否完成了和 provider 的正确连接. """

        self._sync_meta_started_event = asyncio.Event()
        self._sync_meta_done_event = ThreadSafeEvent()
        """记录一次更新 meta 的任务已经完成, 用于做更新的阻塞. """

        self._pending_provider_command_calls: dict[str, CommandTask] = {}
        self._main_task: Optional[asyncio.Task] = None

        self._logger: logging.Logger = self.container.get(LoggerItf) or logging.getLogger(__name__)
        """logger 的缓存."""

        self._states = None

    def get_meta(self, provider_chan_path: str) -> Optional[ChannelMeta]:
        """
        获取一个 meta 参数.
        """
        # 发送更新 meta 的指令.
        channel_path_meta_map = self.provider_meta_map
        return channel_path_meta_map.get(provider_chan_path, None)

    @property
    def states(self) -> StateStore:
        # todo: 实现 duplex state 通讯.
        if self._states is None:
            _states = self.container.get(StateStore)
            if _states is None:
                _states = BaseStateStore(self.root_name)
                self.container.set(StateStore, _states)
            self._states = _states
        return self._states

    async def refresh_meta(self) -> None:
        if not self.connection.is_connected():
            # 如果通讯不成立, 则无法更新.
            await self._clear_connection_status()
            return
        # 尝试发送更新 meta 的命令, 但是同一时间只发送一次.
        await self._send_sync_meta_event()
        # 阻塞等待到刷新成功, 或者连接失败.
        if self.connection.is_connected():
            # 只有在连接成功后, 才阻塞等待到连接成功.
            await self._sync_meta_done_event.wait()
            self._logger.info("refresh duplex channel %s context meta done", self.root_name)

    def is_idle(self) -> bool:
        tasks = self._pending_provider_command_calls.copy()
        for task in tasks.values():
            if not task.done() and task.meta.blocking:
                return False
        return True

    async def wait_idle(self) -> None:
        while True:
            tasks = self._pending_provider_command_calls.copy()
            waiting = []
            for task in tasks.values():
                if not task.done() and task.meta.blocking:
                    waiting.append(task.wait(throw=False))
            if len(waiting) > 0:
                _ = await asyncio.gather(*waiting)
            if self.is_idle():
                break

    async def send_event_to_provider(self, event: ChannelEvent, throw: bool = True) -> None:
        if self.stop_event.is_set():
            self.logger.warning("Channel %s connection is stopped or not available", self.root_name)
            if throw:
                raise ConnectionClosedError(f"Channel {self.root_name} Connection is stopped")
            return
        elif not self.connection.is_connected():
            if throw:
                raise ConnectionNotAvailable(f"Channel {self.root_name} Connection not available")
            return

        try:
            if not event["session_id"]:
                event["session_id"] = self.session_id
            await self.connection.send(event)
            self.logger.debug("channel %s sent event to channel %s", self.root_name, event)
        except (ConnectionClosedError, ConnectionNotAvailable):
            # 发送时连接异常, 标记 disconnected.
            await self._clear_connection_status()
            if throw:
                raise
        except asyncio.CancelledError:
            pass

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self.container.get(LoggerItf) or logging.getLogger("moss")
        return self._logger

    async def start(self) -> None:
        if self._starting:
            self.logger.info("DuplexChannelContext[name=%s] already started", self.root_name)
            await self._started.wait()
            return
        self.logger.info("DuplexChannelContext[name=%s] starting", self.root_name)
        self._starting = True
        # 完成初始化.
        await self._bootstrap()
        # 创建主循环.
        self._main_task = asyncio.create_task(self._main())
        self._started.set()
        self.logger.info("DuplexChannelContext[name=%s] started", self.root_name)

    async def wait_connected(self) -> None:
        await self._connected_event.wait()

    async def close(self) -> None:
        if self.stop_event.is_set():
            return
        # 通知关闭.
        self.stop_event.set()
        # 等待主任务结束.
        try:
            if self._main_task:
                await self._main_task
        except asyncio.CancelledError:
            pass
        await asyncio.to_thread(self.container.shutdown)

    def is_connected(self) -> bool:
        # 判断连接的关键, 是通信存在并且完成了同步.
        return self._connected_event.is_set()

    def is_channel_available(self, provider_chan_path: str) -> bool:
        connection_is_available = self.is_running() and self.connection.is_connected()
        if not connection_is_available:
            return False
        if not self._connected_event.is_set():
            # 标记了连接未生效.
            return False
        # 再判断 meta 也是 available 的.
        meta = self.get_meta(provider_chan_path)
        return meta and meta.available

    def is_channel_connected(self, provider_chan_path: str) -> bool:
        """判断一个 channel 是否可以运行."""
        connection_is_available = self.is_running() and self.connection.is_connected()
        if not connection_is_available:
            return False
        if not self._connected_event.is_set():
            # 标记了连接未生效.
            return False
        # 再判断 meta 也是 available 的.
        meta = self.get_meta(provider_chan_path)
        return meta is not None

    def is_running(self) -> bool:
        """判断 ctx 是否在运行."""
        return self._started.is_set() and not self.stop_event.is_set() and not self.connection.is_closed()

    async def _bootstrap(self):
        # 只启动一次 container, 也只有 context 启动它.
        await asyncio.to_thread(self.container.bootstrap)
        # context 的更新从主动改成被动, 依赖端侧进行握手协议.
        # connection 自身应该有重连机制.
        await self.connection.start()

    async def _main(self):
        try:
            # 异常管理放在外侧, 方便阅读代码.
            # 接受消息的 loop.
            receiving_task = asyncio.create_task(self._main_receiving_loop())
            is_stopped = asyncio.create_task(self.stop_event.wait())
            done, pending = await asyncio.wait(
                [receiving_task, is_stopped],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            # await会将任务产出的异常抛出.
            await receiving_task
        except asyncio.CancelledError as e:
            reason = "proxy proxy cancelled"
            self.logger.info(
                "Channel %s connection cancelled, error=%s, reason=%s",
                self.remote_root_name,
                e,
                reason,
            )
        except ConnectionClosedError as e:
            reason = "proxy proxy connection closed"
            self.logger.info(
                "Channel %s connection closed, error=%s, reason=%s",
                self.remote_root_name,
                e,
                reason,
            )
        except Exception:
            self.logger.exception("proxy proxy error")
            raise
        finally:
            self.stop_event.set()
            await self._clear_connection_status()

    async def _clear_connection_status(self):
        """
        清空连接状态.
        """
        if self._connected_event.is_set():
            self._connected_event.clear()
            self._sync_meta_done_event.clear()
            self._sync_meta_started_event.clear()
            self.session_id = ""
            self.provider_meta_map.clear()
            await self._clear_pending_provider_command_calls()

    async def _wait_task_done_or_stopped(self, task: asyncio.Task) -> bool:
        """
        语法糖, 等待一个任务完成, 但是如果全局 stopped 了, 或者断连了, 就会返回 False.
        """
        wait_stopped = asyncio.create_task(self.stop_event.wait())
        done, pending = await asyncio.wait(
            [task, wait_stopped],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        return task in done

    async def _clear_pending_provider_command_calls(self, reason: str = "") -> None:
        """
        清空所有未完成的任务.
        """
        tasks = self._pending_provider_command_calls.copy()
        self._pending_provider_command_calls.clear()
        for task in tasks.values():
            if not task.done():
                reason = reason or f"Channel proxy `{self.root_name}` not available"
                task.fail(CommandErrorCode.NOT_AVAILABLE.error(reason))

    async def _main_receiving_loop(self) -> None:
        # 等待到全部启动成功.
        try:
            is_reconnected = False
            # 进入到主循环.
            while not self.stop_event.is_set():
                # 如果通讯失效了, 就清空连接状态, 等待重连.
                if not self.connection.is_connected():
                    # 如果在连接状态, 则要清空.
                    if self._connected_event.is_set():
                        # 取消连接状态.
                        await self._clear_connection_status()
                        # 稍微等待下一轮.
                        await asyncio.sleep(0.1)
                        self.logger.info("Channel proxy %s connection status cleared", self.root_name)
                        continue
                    else:
                        await asyncio.sleep(self._wait_reconnect_interval)
                        continue
                else:
                    if not is_reconnected:
                        # 发送初始化连接.  proxy 一定要发送至少第一次, 因为 provider
                        is_reconnected = True
                        await self.send_event_to_provider(ReconnectSessionEvent().to_channel_event())
                        continue

                # 等待一个事件.
                try:
                    event = await self.connection.recv(0.5)
                except asyncio.TimeoutError:
                    continue
                except ConnectionNotAvailable:
                    # 连接失败会继续等待重连.
                    self.logger.debug("Proxy channel %s not available", self.root_name)
                    continue

                self.logger.debug("Proxy channel %s Received event: %s", self.root_name, event)
                # 默认的毒丸逻辑. 防止死锁.
                if event is None:
                    # 退出主循环.
                    # todo: 日志
                    break

                # sync metas 事件的标准处理.
                if create_session := CreateSessionEvent.from_channel_event(event):
                    # 如果是 provider 发送了握手的要求, 则立刻要求更新状态.
                    if create_session.session_id == self.session_id:
                        continue
                    await self._clear_connection_status()
                    self.session_id = create_session.session_id
                    # 标记创建连接成功.
                    event = SessionCreatedEvent(session_id=self.session_id)
                    await self.send_event_to_provider(event.to_channel_event())
                    continue

                elif update_meta := ChannelMetaUpdateEvent.from_channel_event(event):
                    # 如果是 provider 发送了更新状态的结果, 则更新连接状态.
                    await self._handle_update_channel_meta(update_meta)
                    continue
                elif not self._connected_event.is_set() or event["session_id"] != self.session_id:
                    # 如果没有完成 update meta, 所有的事件都会被拒绝, 要求重新开始运行.
                    self.logger.info(
                        "DuplexChannelContext[name=%s] drop event %s and ask reconnect",
                        self.root_name,
                        event,
                    )
                    invalid = ReconnectSessionEvent(session_id=self.session_id).to_channel_event()
                    # 要求 provider 必须完成重连.
                    await self.connection.send(invalid)
                    continue
                else:
                    # 拿到了其它正常的指令. 继续往下走.
                    pass

                if command_done := CommandDoneEvent.from_channel_event(event):
                    # 顺序执行, 避免并行逻辑导致混乱. 虽然可以加锁吧.
                    _ = asyncio.create_task(self._handle_command_done_event(command_done))
                    continue

                else:
                    self.logger.warning(
                        "Channel %s receive event error: unknown event %s",
                        self.root_name,
                        event,
                    )

        except asyncio.CancelledError:
            pass

    async def _send_sync_meta_event(self) -> None:
        """
        发送更新 meta 的请求. 但一个时间段只发送一次.
        """
        if not self._sync_meta_started_event.is_set():
            self._sync_meta_started_event.set()
            self._sync_meta_done_event.clear()
            sync_event = SyncChannelMetasEvent(session_id=self.session_id).to_channel_event()
            await self.send_event_to_provider(sync_event, throw=False)

    async def _handle_update_channel_meta(self, event: ChannelMetaUpdateEvent) -> None:
        """更新 metas 信息."""
        self.remote_root_name = event.root_chan
        # 更新 meta map.
        new_provider_meta_map = {}
        for provider_channel_path, meta in event.metas.items():
            meta = meta.model_copy()
            if provider_channel_path == "":
                meta.name = self.root_name
            new_provider_meta_map[provider_channel_path] = meta

        if not event.all:
            # 不是全量更新时, 也把旧的 meta 加回来.
            for channel_path, meta in self.provider_meta_map.items():
                if channel_path not in new_provider_meta_map:
                    new_provider_meta_map[channel_path] = meta

        # 直接变更当前的 meta map. 则一些原本存在的 channel, 也可能临时不存在了.
        self.provider_meta_map = new_provider_meta_map
        # 更新 sync 的标记.
        if not self._sync_meta_done_event.is_set():
            self._sync_meta_done_event.set()
        if self._sync_meta_started_event.is_set():
            self._sync_meta_started_event.clear()
        # 更新失联状态.
        self._connected_event.set()

    async def send_command_task(self, task: CommandTask) -> CommandCallEvent:
        try:
            cid = task.cid
            # 清空已经存在的 cid 错误?
            if cid in self._pending_provider_command_calls:
                t = self._pending_provider_command_calls.pop(cid)
                t.cancel()
                self.logger.error("Command Task %s duplicated call", cid)
            event = CommandCallEvent(
                session_id=self.session_id,
                name=task.meta.name,
                # channel 名称使用 provider 侧的名称, 用来对 channel 寻址.
                chan=task.chan,
                command_id=task.cid,
                args=list(task.args),
                kwargs=dict(task.kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
                call_id=task.call_id if task else "",
            )
            # 添加新的 task.
            await self.send_event_to_provider(event.to_channel_event(), throw=True)
            self._pending_provider_command_calls[cid] = task
            return event
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            self.logger.exception(exc)
            task.fail(exc)
            raise

    async def expect_task_done(self, event: CommandCallEvent, task: CommandTask) -> None:
        try:
            if task.done():
                return
            await task.wait(throw=False)
            # 来自服务端的异常.
            if task.cid in self._pending_provider_command_calls and self.is_channel_available(task.chan):
                if exp := task.exception():
                    await self.send_event_to_provider(event.cancel().to_channel_event(), throw=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception(exc)
        finally:
            if not task.done():
                task.cancel()
            if task.cid in self._pending_provider_command_calls:
                del self._pending_provider_command_calls[task.cid]

    async def _handle_command_done_event(self, event: CommandDoneEvent) -> None:
        try:
            command_id = event.command_id
            task = self._pending_provider_command_calls.pop(command_id)
            if task is not None:
                if task.done():
                    pass
                elif event.errcode == 0:
                    task.resolve(event.result)
                else:
                    error = CommandError(event.errcode, event.errmsg)
                    task.fail(error)
            else:
                self.logger.info("receive command done event %s match no command", event)
        except Exception as e:
            self.logger.exception("Handle command done event failed %s", e)
            raise


class DuplexChannelRuntime(AbsChannelRuntime):
    """
    实现一个极简的 Duplex Channel, 它核心是可以通过 ChannelMeta 被动态构建出来.
    """

    def __init__(
            self,
            *,
            channel: Channel,
            provider_chan_path: str,
            ctx: DuplexChannelContext,
    ) -> None:
        self._ctx = ctx
        self._provider_chan_path = provider_chan_path
        super().__init__(
            channel=channel,
            container=ctx.container,
            logger=ctx.logger,
        )
        self.log_prefix = f"[DuplexChannelRuntime name={self._name} id={self.id} cls={self.__class__}]"

    def is_running(self) -> bool:
        return super().is_running() and self._ctx.is_running()

    def prepare_container(self, container: IoCContainer | None) -> IoCContainer:
        container.set(LoggerItf, self._ctx.logger)
        container.set(StateStore, self._ctx.states)
        container = super().prepare_container(container)
        return container

    def imported(self) -> dict[str, Channel]:
        # 不需要展开节点.
        return {}

    async def on_running(self) -> None:
        return

    async def _generate_metas(self, force: bool) -> dict[ChannelFullPath, ChannelMeta]:
        await self._ctx.refresh_meta()
        metas = self._ctx.provider_meta_map
        self_meta = metas.get('')
        if self_meta:
            self_meta = self_meta.model_copy(update={"name": self._name})
            metas[''] = self_meta
        return metas

    def _is_available(self) -> bool:
        return self._ctx.is_channel_available(self._provider_chan_path)

    async def _push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        event = await self._ctx.send_command_task(task)
        _ = asyncio.create_task(self._ctx.expect_task_done(event, task))

    async def _main_loop(self) -> None:
        pass

    def is_idle(self) -> bool:
        return self._ctx.is_idle()

    async def wait_idle(self) -> None:
        await self._ctx.wait_idle()

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Channel proxy {self._name} is not running")

    def is_connected(self) -> bool:
        return self.is_running() and self._ctx.is_channel_connected(self._provider_chan_path)

    async def wait_connected(self) -> None:
        if not self.is_running():
            return
        await self._ctx.wait_connected()
        self._cached_metas = self._ctx.provider_meta_map

    def self_commands(self, available_only: bool = True) -> dict[str, Command]:
        # 先获取本地的命令.
        result = {}
        # 拿出原始的 meta.
        meta = self._ctx.get_meta(self._provider_chan_path)
        if meta is None:
            return result
        # 再封装远端的命令.
        for command_meta in meta.commands:
            if command_meta.name not in result and not available_only or command_meta.available:
                func = self._get_provider_command_func(self._provider_chan_path, command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def commands(self, available_only: bool = True) -> dict[CommandUniqueName, Command]:
        result = {}
        if not self.is_running():
            return {}
        for channel_path, meta in self.metas().items():
            for command_meta in meta.commands:
                unique_name = Command.make_uniquename(channel_path, command_meta.name)
                func = self._get_provider_command_func(channel_path, command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[unique_name] = command
        return result

    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        不需要递归获取了.
        """
        if not self.is_running():
            return None
        channel_path, command_name = Command.split_uniquename(name)
        channel_meta = self._ctx.get_meta(channel_path)
        if channel_meta is None:
            return None
        for command_meta in channel_meta.commands:
            if command_meta.name == command_name:
                func = self._get_provider_command_func(channel_path, command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                return command
        return None

    def _get_provider_command_func(
            self,
            chan: ChannelFullPath,
            meta: CommandMeta,
    ) -> Callable[[...], Coroutine[None, None, Any]]:

        # 回调服务端的函数.
        async def _call_provider_as_func(*args, **kwargs):
            if not self.is_available():
                # 告知上游运行失败.
                raise CommandError(CommandErrorCode.NOT_AVAILABLE, f"Channel {self._name} not available")
            if chan not in self._ctx.provider_meta_map:
                raise CommandErrorCode.NOT_FOUND.error(f"channel {chan} is not found")
            _chan_meta = self._ctx.provider_meta_map.get(chan)
            if not _chan_meta.available:
                raise CommandErrorCode.NOT_AVAILABLE.error(f"channel {chan} is not available")

            # 尝试透传上游赋予的参数.
            task: CommandTask | None = None
            try:
                task = ChannelCtx.task()
            except LookupError:
                pass
            cid = task.cid if task else uuid()

            # 生成对下游的调用.
            if task is None:
                task = BaseCommandTask(
                    chan=chan,
                    meta=meta,
                    tokens="",
                    func=None,
                    args=list(args),
                    kwargs=dict(kwargs),
                    cid=cid,
                )

            event = await self._ctx.send_command_task(task)
            await self._ctx.expect_task_done(event, task)
            return task.result()

        return _call_provider_as_func

    def get_self_command(self, name: str) -> Optional[Command]:
        meta = self.self_meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_provider_command_func(self._provider_chan_path, command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    async def _clear(self) -> None:
        if not self._ctx.is_running():
            return
        try:
            event = ClearEvent(
                session_id=self._ctx.session_id,
                chan=self._provider_chan_path,
            )
            await self._ctx.send_event_to_provider(event.to_channel_event(), throw=True)
        except Exception as e:
            self.logger.exception(e)

    async def on_start_up(self) -> None:
        # 启动 ctx.
        await self._ctx.start()

    async def on_close(self) -> None:
        await self._ctx.close()

    def default_states(self) -> list[State]:
        return []


class DuplexChannelProxy(Channel):
    def __init__(
            self,
            *,
            name: str,
            description: str = "",
            to_provider_connection: Connection,
    ):
        self._name = name
        self._description = description
        self._uid = uuid()
        self._provider_connection = to_provider_connection
        self._provider_channel_path = ""
        self._runtime: Optional[DuplexChannelRuntime] = None
        self._ctx: DuplexChannelContext | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def id(self) -> str:
        return self._uid

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "DuplexChannelRuntime":
        if self._runtime is not None and self._runtime.is_running():
            raise RuntimeError(f"Channel {self} has already been started.")

        self._ctx = DuplexChannelContext(
            name=self._name,
            container=container,
            connection=self._provider_connection,
        )

        runtime = DuplexChannelRuntime(
            channel=self,
            provider_chan_path="",
            ctx=self._ctx,
        )
        self._runtime = runtime
        return runtime
