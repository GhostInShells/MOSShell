import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer
from typing_extensions import Self

from ghoshell_moss.core.concepts.channel import Builder, Channel, ChannelBroker, ChannelFullPath, ChannelMeta, R
from ghoshell_moss.core.concepts.command import BaseCommandTask, Command, CommandMeta, CommandTask, CommandWrapper
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

from .connection import Connection, ConnectionClosedError, ConnectionNotAvailable
from .protocol import (
    ChannelEvent,
    ChannelMetaUpdateEvent,
    ClearCallEvent,
    ClearDoneEvent,
    CommandCallEvent,
    CommandDoneEvent,
    CommandPeekEvent,
    CreateSessionEvent,
    PausePolicyDoneEvent,
    PausePolicyEvent,
    ReconnectSessionEvent,
    RunPolicyDoneEvent,
    RunPolicyEvent,
    SessionCreatedEvent,
    SyncChannelMetasEvent,
)

__all__ = ["DuplexChannelBroker", "DuplexChannelProxy", "DuplexChannelStub"]

from ghoshell_moss.core.concepts.states import MemoryStateStore, StateStore

"""
DuplexChannel Proxy 一侧的实现, 
todo: 全部改名为 Proxy 
"""


class DuplexChannelContext:
    """
    创建一个 Context 对象, 是所有 Duplex Channel Brokers 共同依赖的.
    """

    def __init__(
        self,
        *,
        name: str,
        connection: Connection,
        container: Optional[IoCContainer] = None,
        command_peek_interval: float = 2.0,
    ):
        self.root_name = name
        """根节点的名字. 这个名字可能和远端的 channel 根节点不一样. """
        self.remote_root_name = ""
        """远端的 root channel 名字"""
        self._command_peek_interval = command_peek_interval if command_peek_interval > 0 else 2.0

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
        self._disconnected_event = asyncio.Event()
        self._disconnected_event.set()
        """标记是否完成了和 provider 的正确连接. """

        self._sync_meta_started_event = asyncio.Event()
        self._sync_meta_done_event = ThreadSafeEvent()
        """记录一次更新 meta 的任务已经完成, 用于做更新的阻塞. """

        self._pending_server_command_calls: dict[str, CommandTask] = {}

        self.provider_to_broker_event_queue_map: dict[str, asyncio.Queue[ChannelEvent | None]] = {}
        """按 channel 名称进行分发的队列."""

        self._main_task: Optional[asyncio.Task] = None

        self._logger: logging.Logger = self.container.get(LoggerItf) or logging.getLogger(__name__)
        """logger 的缓存."""

        self._states = None

    def get_meta(self, provider_chan_path: str) -> Optional[ChannelMeta]:
        """
        获取一个 meta 参数.
        """
        # 发送更新 meta 的指令.
        return self.provider_meta_map.get(provider_chan_path, None)

    @property
    def states(self) -> StateStore:
        # todo: 实现 duplex state 通讯.
        if self._states is None:
            _states = self.container.get(StateStore)
            if _states is None:
                _states = MemoryStateStore(self.root_name)
                self.container.set(StateStore, _states)
            self._states = _states
        return self._states

    async def refresh_meta(self) -> None:
        if not self.connection.is_available():
            # 如果通讯不成立, 则无法更新.
            await self._clear_connection_status()
            return
        # 尝试发送更新 meta 的命令, 但是同一时间只发送一次.
        await self._send_sync_meta_event()
        # 阻塞等待到刷新成功, 或者连接失败.
        if self.connection.is_available():
            # 只有在连接成功后, 才阻塞等待到连接成功.
            await self._sync_meta_done_event.wait()
            self._logger.info("refresh duplex channel %s context meta done", self.root_name)

    async def send_event_to_provider(self, event: ChannelEvent, throw: bool = True) -> None:
        if self.stop_event.is_set():
            self.logger.warning("Channel %s connection is stopped or not available", self.root_name)
            if throw:
                raise ConnectionClosedError(f"Channel {self.root_name} Connection is stopped")
            return
        elif not self.connection.is_available():
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

    def get_server_event_queue(self, name: str) -> asyncio.Queue[ChannelEvent | None]:
        """
        :param name: 这里的 name 是 channel 在远端的原名称.
        """
        if name == self.remote_root_name:
            # 用 "" 表示根节点.
            name = ""
        if name not in self.provider_to_broker_event_queue_map:
            self.provider_to_broker_event_queue_map[name] = asyncio.Queue()
        return self.provider_to_broker_event_queue_map[name]

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
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

    def connect_broker(self, channel_name: str) -> None:
        if channel_name in self.provider_meta_map:
            self.provider_to_broker_event_queue_map[channel_name] = asyncio.Queue()

    def disconnect_broker(self, channel_name: str) -> None:
        if channel_name in self.provider_to_broker_event_queue_map:
            del self.provider_to_broker_event_queue_map[channel_name]

    async def wait_connected(self) -> None:
        while self._disconnected_event.is_set():
            # 以 0.1 秒为周期等待 provider 和 proxy 连接成功.
            await asyncio.sleep(0.1)

    async def close(self) -> None:
        if self.stop_event.is_set():
            return
        # 通知关闭.
        self.stop_event.set()
        # 尝试通知所有的子节点关闭.
        for queue in self.provider_to_broker_event_queue_map.values():
            queue.put_nowait(None)
        # 等待主任务结束.
        try:
            if self._main_task:
                await self._main_task
        except asyncio.CancelledError:
            pass
        await asyncio.to_thread(self.container.shutdown)

    def is_connected(self) -> bool:
        # 判断连接的关键, 是通信存在并且完成了同步.
        return not self._disconnected_event.is_set()

    def is_channel_available(self, provider_chan_path: str) -> bool:
        connection_is_available = self.is_running() and self.connection.is_available()
        if not connection_is_available:
            return False
        if self._disconnected_event.is_set():
            # 标记了连接未生效.
            return False
        # 再判断 meta 也是 available 的.
        meta = self.get_meta(provider_chan_path)
        return meta and meta.available

    def is_channel_connected(self, provider_chan_path: str) -> bool:
        """判断一个 channel 是否可以运行."""
        connection_is_available = self.is_running() and self.connection.is_available()
        if not connection_is_available:
            return False
        if self._disconnected_event.is_set():
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
            reason = "client proxy cancelled"
            self.logger.info(
                "Channel %s connection cancelled, error=%s, reason=%s",
                self.remote_root_name,
                e,
                reason,
            )
        except ConnectionClosedError as e:
            reason = "client proxy connection closed"
            self.logger.info(
                "Channel %s connection closed, error=%s, reason=%s",
                self.remote_root_name,
                e,
                reason,
            )
        except Exception:
            self.logger.exception("Client proxy error")
            raise
        finally:
            self.stop_event.set()
            for queue in self.provider_to_broker_event_queue_map.values():
                queue.put_nowait(None)
            await self._clear_connection_status()

    async def _clear_connection_status(self):
        """
        清空连接状态.
        """
        if not self._disconnected_event.is_set():
            self._sync_meta_done_event.clear()
            self._sync_meta_started_event.clear()
            self.session_id = ""
            self._disconnected_event.set()
            self.provider_meta_map.clear()
            await self._clear_pending_server_command_calls()

    async def _wait_task_done_or_stopped(self, task: asyncio.Task) -> bool:
        """
        语法糖, 等待一个任务完成, 但是如果全局 stopped 了, 或者断连了, 就会返回 False.
        """
        wait_stopped = asyncio.create_task(self.stop_event.wait())
        wait_disconnected = asyncio.create_task(self._disconnected_event.wait())
        done, pending = await asyncio.wait(
            [task, wait_stopped, wait_disconnected],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        return task in done

    async def _clear_pending_server_command_calls(self, reason: str = "") -> None:
        """
        清空所有未完成的任务.
        """
        tasks = self._pending_server_command_calls.copy()
        self._pending_server_command_calls.clear()
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
                if not self.connection.is_available() and not self._disconnected_event.is_set():
                    # 取消连接状态.
                    await self._clear_connection_status()
                    # 稍微等待下一轮.
                    await asyncio.sleep(0.1)
                    self.logger.info("Channel proxy %s connection status cleared", self.root_name)
                    continue
                elif not is_reconnected:
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
                elif self._disconnected_event.is_set() or event["session_id"] != self.session_id:
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
                    await self._handle_command_done_event(command_done)
                    continue

                # 判断回调分发给哪个具体的 channel.
                if "chan" in event["data"]:
                    chan = event["data"]["chan"]
                    # 检查是否是已经注册的 channel.
                    if chan not in self.provider_meta_map:
                        self.logger.warning(
                            "Channel %s receive event error: channel %s queue not found, drop event %s",
                            self.root_name,
                            chan,
                            event,
                        )
                        continue

                    queue = self.get_server_event_queue(chan)
                    # 分发给指定 channel.
                    await queue.put(event)
                else:
                    # 拿到的 channel 不可理解.
                    self.logger.error("Channel %s receive unknown event: %s", self.root_name, event)
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
            new_provider_meta_map[provider_channel_path] = meta.model_copy()

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
        self._disconnected_event.clear()

    async def _peek_command_task_loop(self, task: CommandTask, call: CommandCallEvent) -> None:
        """
        周期性检查一个命令是否更新.
        todo: 考虑移除掉.
        """
        try:
            await asyncio.sleep(self._command_peek_interval)
            while not task.done():
                peek_event = CommandPeekEvent(
                    chan=call.chan,
                    command_id=call.command_id,
                )
                await self.send_event_to_provider(peek_event.to_channel_event())
                await asyncio.sleep(self._command_peek_interval)
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError as e:
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"Channel `{self.root_name}` connection closed: {e}"))
        except ConnectionNotAvailable as e:
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"Channel `{self.root_name}` connection not available: {e}"))
        except Exception:
            self.logger.exception("Peek command task loop failed")

    async def execute_command_call(self, meta: CommandMeta, event: CommandCallEvent) -> CommandTask:
        """与远程 server 进行通讯, 发送一个 command call, 并且保障有回调."""
        cid = event.command_id
        command_call_task_stub = BaseCommandTask(
            meta=meta,
            func=None,
            cid=event.command_id,
            tokens=event.tokens,
            args=event.args,
            kwargs=event.kwargs,
            context=event.context,
        )
        try:
            # 清空已经存在的 cid 错误?
            if cid in self._pending_server_command_calls:
                t = self._pending_server_command_calls.pop(cid)
                t.cancel()
                self.logger.error("Command Task %s duplicated call", cid)
            # 添加新的 task.
            self._pending_server_command_calls[cid] = command_call_task_stub

            # 等待异步返回结果.
            await self.send_event_to_provider(event.to_channel_event(), throw=True)
            task_done = asyncio.create_task(command_call_task_stub.wait(throw=False))
            await self._wait_task_done_or_stopped(task_done)
            return command_call_task_stub

        except (ConnectionClosedError, ConnectionNotAvailable):
            # 连接失败后.
            command_call_task_stub.fail(CommandErrorCode.NOT_AVAILABLE.error("channel connection not available"))
            return command_call_task_stub

        except asyncio.CancelledError:
            # 取消也会正常返回.
            if not command_call_task_stub.done():
                command_call_task_stub.cancel("cancelled by server")
                # 发送取消事件, 通知给下游.
                if self.is_channel_available(event.chan):
                    await self.send_event_to_provider(event.cancel().to_channel_event(), throw=False)
            return command_call_task_stub
        except Exception as e:
            self.logger.exception("Execute command call failed")
            # 拿到了不知名的异常后.
            if not command_call_task_stub.done():
                command_call_task_stub.fail(e)
                if self.is_channel_available(event.chan):
                    await self.send_event_to_provider(event.cancel().to_channel_event(), throw=False)
            raise
        finally:
            # 必须移除自身在列表的存在.
            if cid in self._pending_server_command_calls:
                del self._pending_server_command_calls[cid]

    async def _handle_command_done_event(self, event: CommandDoneEvent) -> None:
        try:
            command_id = event.command_id
            if command_id in self._pending_server_command_calls:
                task = self._pending_server_command_calls[command_id]
                if task.done():
                    pass
                elif event.errcode == 0:
                    task.resolve(event.result)
                else:
                    error = CommandError(event.errcode, event.errmsg)
                    task.fail(error)
            else:
                self.logger.info("receive command done event %s match no command", event)
        except Exception:
            self.logger.exception("Handle command done event failed")


class DuplexChannelStub(Channel):
    """被 channel meta 动态生成的子 channel."""

    def __init__(
        self,
        *,
        name: str,  # 本地的名称.
        ctx: DuplexChannelContext,
        server_chan_name: str = "",  # 远端真实的名称.
    ) -> None:
        self._name = name
        self._server_chan_name = server_chan_name or name
        self._ctx = ctx
        # 运行时缓存.
        self._broker: ChannelBroker | None = None
        self._children_stubs: dict[str, DuplexChannelStub] = {}

    def name(self) -> str:
        return self._name

    def _get_server_channel_meta(self) -> Optional[ChannelMeta]:
        # 获取自己在 server 端的 channel meta.
        return self._ctx.provider_meta_map.get(self._server_chan_name)

    @property
    def broker(self) -> ChannelBroker:
        if self._broker is None:
            raise RuntimeError(f"Channel {self} has not been started yet.")
        return self._broker

    def import_channels(self, *children: "Channel") -> Self:
        raise NotImplementedError(f"Duplex Channel {self._name} not allowed to import channels")

    def new_child(self, name: str) -> Self:
        raise NotImplementedError(f"Duplex Channel {self._name} not allowed to create child")

    def children(self) -> dict[str, "Channel"]:
        server_chan_meta = self._get_server_channel_meta()
        if server_chan_meta is None:
            # 没有远端的 channel meta.
            return {}

        # 遍历自己的 meta children.
        children_stubs = {}
        for child_channel_name in server_chan_meta.children:
            if child_channel_name in self._children_stubs:
                # 这个 stub 已经被创建过了. 复制到新字典.
                children_stubs[child_channel_name] = self._children_stubs[child_channel_name]
                continue
            # 获取这个子节点的远程 channel 路径.
            child_server_chan_path = Channel.join_channel_path(self._server_chan_name, child_channel_name)
            stub = DuplexChannelStub(
                name=child_channel_name,
                ctx=self._ctx,
                server_chan_name=child_server_chan_path,
            )
            children_stubs[child_channel_name] = stub
        # 每次都更新当前的 children stubs.
        self._children_stubs.clear()
        self._children_stubs = children_stubs
        result: dict[str, Channel] = children_stubs.copy()
        return result

    def is_running(self) -> bool:
        return self._broker is not None and self._ctx.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelBroker":
        if self._broker is not None and self._broker.is_running():
            raise RuntimeError(f"Channel {self._name} has already been started.")
        if not self._ctx.is_running():
            raise RuntimeError(f"Duplex Channel {self._name} Context is not running")

        broker = DuplexChannelBroker(
            name=self._name,
            provider_chan_path=self._server_chan_name,
            ctx=self._ctx,
            is_root=False,
        )
        self._broker = broker
        return broker

    @property
    def build(self) -> Builder:
        raise NotImplementedError(f"Duplex Channel {self._name} not allowed to build channel")


class DuplexChannelBroker(ChannelBroker):
    """
    实现一个极简的 Duplex Channel, 它核心是可以通过 ChannelMeta 被动态构建出来.
    """

    def __init__(
        self,
        *,
        name: str,
        provider_chan_path: str,
        ctx: DuplexChannelContext,
        is_root: bool = False,
    ) -> None:
        """
        :param name: channel local name
        :param provider_chan_path: the origin channel name from the remote server
        :param ctx: shared ctx of all the channels.
        """
        self._name = name
        self._provider_chan_path = provider_chan_path
        self._ctx = ctx
        self._is_root = is_root
        # 重新定义 id.
        meta = ctx.get_meta(self._provider_chan_path)

        self._id = meta.channel_id if meta else uuid()

        # 运行时参数
        self._starting = False
        self._started_at: Optional[float] = None
        self._logger: logging.Logger | None = self.container.get(LoggerItf) or logging.getLogger(__name__)

        self._self_close_event = ThreadSafeEvent()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._main_loop_done_event = ThreadSafeEvent()

    def name(self) -> str:
        return self._name

    @property
    def container(self) -> IoCContainer:
        return self._ctx.container

    @property
    def id(self) -> str:
        return self._id

    def is_running(self) -> bool:
        return self._starting and self._ctx.is_running() and not self._self_close_event.is_set()

    @property
    def logger(self) -> logging.Logger:
        return self._ctx.logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Channel client {self._name} is not running")

    def meta(self) -> ChannelMeta:
        self._check_running()
        return self._build_meta_from_ctx()

    async def refresh_meta(self) -> None:
        self._check_running()
        # 永远不同步获取 meta.
        refresh = self._is_root
        if refresh:
            await self._ctx.refresh_meta()

    def _build_meta_from_ctx(self) -> ChannelMeta:
        meta = self._ctx.get_meta(self._provider_chan_path)
        if meta is None:
            return ChannelMeta(
                name=self._name,
                channel_id=self.id,
                available=False,
                dynamic=True,
            )
        # 避免污染.
        meta = meta.model_copy()
        # 从 server meta 中准备 commands 的原型.
        if meta.name != self._name:
            commands = {}
            for command_meta in meta.commands:
                # 命令替换名称为自身的名称. 给调用方看.
                command_meta = command_meta.model_copy(update={"chan": self._name})
                commands[command_meta.name] = command_meta
            meta.commands = list(commands.values())
            # 修改别名.
            meta.name = self._name
        return meta

    def is_available(self) -> bool:
        return self.is_running() and self._ctx.is_channel_available(self._provider_chan_path)

    def is_connected(self) -> bool:
        return self.is_running() and self._ctx.is_channel_connected(self._provider_chan_path)

    async def wait_connected(self) -> None:
        while not self.is_connected():
            await asyncio.sleep(0.1)

    def commands(self, available_only: bool = True) -> dict[str, Command]:
        # 先获取本地的命令.
        result = {}
        # 拿出原始的 meta.
        meta = self._ctx.get_meta(self._provider_chan_path)
        if meta is None:
            return result
        # 再封装远端的命令.
        for command_meta in meta.commands:
            if command_meta.name not in result and not available_only or command_meta.available:
                func = self._get_server_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_server_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        session_id = self._ctx.session_id

        # 回调服务端的函数.
        async def _call_server_as_func(*args, **kwargs):
            if not self.is_available():
                # 告知上游运行失败.
                raise CommandError(CommandErrorCode.NOT_AVAILABLE, f"Channel {self._name} not available")

            # 尝试透传上游赋予的参数.
            task: CommandTask | None = None
            try:
                task = CommandTask.get_from_context()
            except LookupError:
                pass
            cid = task.cid if task else uuid()

            # 生成对下游的调用.
            event = CommandCallEvent(
                session_id=session_id,
                name=name,
                # channel 名称使用 server 侧的名称, 用来对 channel 寻址.
                chan=self._provider_chan_path,
                command_id=cid,
                args=list(args),
                kwargs=dict(kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
            )

            task = await self._ctx.execute_command_call(meta, event)
            if exp := task.exception():
                raise exp
            return task.result()

        return _call_server_as_func

    def get_command(self, name: str) -> Optional[Command]:
        meta = self.meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_server_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        func = self._get_server_command_func(task.meta)
        if func is None:
            raise LookupError(f"Channel {self._name} can find command {task.meta.name}")
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            event = RunPolicyEvent(
                session_id=self._ctx.session_id,
                chan=self._provider_chan_path,
            )
            await self._ctx.send_event_to_provider(event.to_channel_event(), throw=False)
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Send run policy event failed")

    async def policy_pause(self) -> None:
        self._check_running()
        try:
            event = PausePolicyEvent(
                session_id=self._ctx.session_id,
                chan=self._provider_chan_path,
            )
            await self._ctx.send_event_to_provider(event.to_channel_event(), throw=True)
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Send pause policy event failed")

    async def clear(self) -> None:
        self._check_running()
        try:
            event = ClearCallEvent(
                session_id=self._ctx.session_id,
                chan=self._provider_chan_path,
            )
            await self._ctx.send_event_to_provider(event.to_channel_event(), throw=True)
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Send clear event failed")

    async def _consume_server_event_loop(self):
        try:
            while self.is_running():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Consume server event loop failed")
            self._self_close_event.set()
        finally:
            self.logger.info("channel %s consume_server_event_loop stopped", self._name)

    async def _main_loop(self):
        try:
            consume_loop_task = asyncio.create_task(self._consume_server_event_loop())
            await consume_loop_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("DuplexChannelBroker main loop failed")
            raise
        finally:
            # 内层不允许shutdown外层传递的container.
            # await asyncio.to_thread(self.container.shutdown)
            self._main_loop_done_event.set()

    async def _consume_server_event(self):
        try:
            if self._ctx.connection.is_closed():
                self._self_close_event.set()
                return

            queue = self._ctx.get_server_event_queue(self._provider_chan_path)

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                return
            if item is None:
                self._self_close_event.set()
                return
            if item.get("timestamp") < self._started_at:
                self.logger.warning("receive overdue events %s", item)
                return
            if model := RunPolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s run policy is done from event %s", self._name, model)
            elif model := PausePolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s pause policy is done from event %s", self._name, model)
            elif model := ClearDoneEvent.from_channel_event(item):
                self.logger.info("channel %s clear is done from event %s", self._name, model)
            else:
                self.logger.info("unknown server event %s", item)
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("Consume server event failed")

    async def start(self) -> None:
        if self._starting:
            self.logger.info("DuplexChannelBroker[name=%s] already started", self._name)
            return
        self.logger.info("DuplexChannelBroker[name=%s] starting", self._name)
        self._starting = True
        self._started_at = time.time()
        if not self._ctx.is_running():
            # 启动 ctx.
            await self._ctx.start()
        # 建立拉取数据的联系.
        self._ctx.connect_broker(self._provider_chan_path)
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self.logger.info("DuplexChannelBroker[name=%s] started", self._name)

    def is_root(self) -> bool:
        return self._is_root

    @property
    def states(self) -> StateStore:
        return self._ctx.states

    async def close(self) -> None:
        if self._self_close_event.is_set():
            return
        self._self_close_event.set()

        try:
            if self._main_loop_task:
                self._main_loop_task.cancel()
                await self._main_loop_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("DuplexChannelBroker close failed")
            raise
        finally:
            self._started_at = None
            self._starting = False
            if self.is_root():
                # root 节点可以关闭 ctx.
                await self._ctx.close()
            else:
                # 关闭结束 ctx.
                self._ctx.disconnect_broker(self._provider_chan_path)
                self._ctx = None


class DuplexChannelProxy(Channel):
    def __init__(
        self,
        *,
        name: str,
        to_server_connection: Connection,
    ):
        self._name = name
        self._server_connection = to_server_connection
        self._server_channel_path = ""
        self._broker: Optional[DuplexChannelBroker] = None
        self._ctx: DuplexChannelContext | None = None
        """运行的时候才会生成 Channel Context"""
        self._children_stubs: dict[str, DuplexChannelStub] = {}

    def name(self) -> str:
        return self._name

    @property
    def broker(self) -> ChannelBroker:
        if self._broker is None:
            raise RuntimeError(f"Channel {self} has not been started yet.")
        return self._broker

    def import_channels(self, *children: "Channel") -> Self:
        raise NotImplementedError(f"Duplex Channel {self._name} cannot import channels")

    def new_child(self, name: str) -> Self:
        raise NotImplementedError(f"Duplex Channel {self._name} cannot create child")

    def children(self) -> dict[str, "Channel"]:
        # todo: 目前没有加锁, 可能需要有锁实现?

        children_stubs = {}
        # 服务端的已经不存在了. 则自己也不一定存在了.
        if self._server_channel_path not in self._ctx.provider_meta_map:
            return {}

        # 从 server meta 里判断自己的孩子们.
        server_meta = self._ctx.provider_meta_map[self._server_channel_path]
        for child_name in server_meta.children:
            child_provider_channel_path = Channel.join_channel_path(self._server_channel_path, child_name)
            # 儿子节点不存在.
            if child_provider_channel_path not in self._ctx.provider_meta_map:
                # 跳过. 这种情况肯定是有 bug.
                # todo: log
                continue

            if child_name in self._children_stubs:
                # 这个说明, 相同命名和路径的 stub 已经创建过了.
                children_stubs[child_name] = self._children_stubs[child_name]
            else:
                # 准备一个 local channel.
                stub = DuplexChannelStub(
                    name=child_name,
                    ctx=self._ctx,
                    server_chan_name=child_provider_channel_path,
                )
                # 增加之前不存在的 child.
                children_stubs[child_name] = stub
        self._children_stubs = children_stubs
        # 生成一个新的组合.
        result: dict[str, Channel] = self._children_stubs.copy()
        return result

    def is_running(self) -> bool:
        return self._broker is not None and self._broker.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "DuplexChannelBroker":
        if self._broker is not None and self._broker.is_running():
            raise RuntimeError(f"Channel {self} has already been started.")

        self._ctx = DuplexChannelContext(
            name=self._name,
            container=container,
            connection=self._server_connection,
        )

        client = DuplexChannelBroker(
            name=self._name,
            provider_chan_path="",
            ctx=self._ctx,
            # 标记是根节点.
            is_root=True,
        )
        self._broker = client
        return client

    @property
    def build(self) -> Builder:
        raise NotImplementedError(f"Duplex Channel {self._name} cannot build channel")
