from typing import TypedDict, Dict, Any, ClassVar, Optional, List, Callable, Coroutine
from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss import ChannelClient
from ghoshell_moss.concepts.channel import Channel, ChannelServer, ChannelMeta, Builder, R
from ghoshell_moss.concepts.errors import FatalError, CommandError, CommandErrorCode
from ghoshell_moss.concepts.command import Command, CommandTask, BaseCommandTask, CommandMeta, CommandWrapper
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer
from pydantic import BaseModel, Field, ValidationError
import logging
import asyncio


class ChannelEvent(TypedDict):
    event_id: str
    event_type: str
    session_id: str
    data: Dict[str, Any]


class ChannelEventModel(BaseModel, ABC):
    event_type: ClassVar[str] = ""

    event_id: str = Field(default_factory=uuid, description="event id for transport")
    session_id: str = Field(description="channel client id")

    def to_channel_event(self) -> ChannelEvent:
        data = self.model_dump(exclude_none=True, exclude={'event_type', 'channel_id', 'channel_name', 'event_id'})
        return ChannelEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            session_id=self.session_id,
            data=data,
        )

    @classmethod
    def from_channel_event(cls, channel_event: ChannelEvent) -> Optional[Self]:
        if cls.event_type != channel_event['event_type']:
            return None
        data = channel_event['data']
        data['event_id'] = channel_event['event_id']
        data['session_id'] = channel_event['session_id']
        return cls(**data)


# --- client event --- #


class RunPolicyEvent(ChannelEventModel):
    """开始运行 channel 的 policy"""
    event_type: ClassVar[str] = "moss.channel.client.policy.run"
    chan: str = Field(description="channel name")


class PausePolicyEvent(ChannelEventModel):
    """暂停某个 channel 的 policy 运行状态"""
    event_type: ClassVar[str] = "moss.channel.client.policy.pause"
    chan: str = Field(description="channel name")


class ClearCallEvent(ChannelEventModel):
    """发出讯号给某个 channel, 执行状态清空的逻辑"""
    event_type: ClassVar[str] = "moss.channel.client.clear.call"
    chan: str = Field(description="channel name")


class CommandCallEvent(ChannelEventModel):
    """发起一个 command 的调用. """

    # todo: 未来要加一个用 command_id 轮询 server 状态的事件. 用来避免通讯丢失.

    event_type: ClassVar[str] = "moss.channel.client.command.call"
    name: str = Field(description="command name")
    chan: str = Field(description="channel name")
    command_id: str = Field(default_factory=uuid, description="command id")
    args: List[Any] = Field(default_factory=list, description="command args")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="kwargs of the command")
    tokens: str = Field("", description="command tokens")
    context: Dict[str, Any] = Field(default_factory=dict, description="context of the command")

    def not_available(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_AVAILABLE.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not available",
            data=None,
            chan=self.chan,
        )

    def done(self, result: Any, errcode: int, errmsg: str) -> "CommandDoneEvent":
        return CommandDoneEvent(
            command_id=self.command_id,
            errcode=errcode,
            errmsg=errmsg,
            data=result,
            chan=self.chan,
        )

    def not_found(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_FOUND.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not found",
            chan=self.chan,
            data=None,
        )


class CommandPeekEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.client.command.peek"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")


class CommandCancelEvent(ChannelEventModel):
    """通知 channel 指定的 command 被取消. """
    event_type: ClassVar[str] = "moss.channel.client.command.cancel"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")


class SyncChannelMetasEvent(ChannelEventModel):
    """要求同步 channel 的 meta 信息. """
    event_type: ClassVar[str] = "moss.channel.meta.sync"
    channels: List[str] = Field(default_factory=list, description="channel names to sync")


# --- server event --- #

class CommandDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.command.done"
    chan: str = Field(description="channel name")
    command_id: str = Field(description="command id")
    errcode: int = Field(default=0, description="command errcode")
    errmsg: str = Field(default="", description="command errmsg")
    data: Any = Field(description="result of the command")


class ClearDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.clear.done"
    chan: str = Field(description="channel name")


class RunPolicyDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.policy.run_done"


class PausePolicyDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.policy.pause_done"
    chan: str = Field(description="channel name")


class ChannelMetaUpdateEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.meta.update"
    metas: List[ChannelMeta] = Field(default_factory=list, description="channel meta")
    root_chan: str = Field(description="channel name")


class ServerErrorEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.error"
    errcode: int = Field(description="error code")
    errmsg: str = Field(description="error message")


# --- errors --- #

class ConnectionClosedError(Exception):
    """表示 connection 已经连接失败. """
    pass


class Connection(ABC):
    """
    Server 与 client 之间的通讯连接, 用来接受和发布事件.
    Server 持有的应该是 ClientConnection
    而 Client 持有的应该是 ServerConnection.
    但两者的接口目前看起来应该是相似的.
    """

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        """从通讯事件循环中获取一个事件. client 获取的是 server event, server 获取的是 client event"""
        pass

    @abstractmethod
    async def send(self, event: ChannelEvent) -> None:
        """发送一个事件给远端, client 发送的是 client event, server 发送的是 server event."""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """判断 connection 是否已经结束了. """
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭这个 connection """
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动这个 connection. """
        pass


# --- event handlers --- #

ChannelEventHandler = Callable[[Channel, ChannelEvent], Coroutine[None, None, bool]]
""" 自定义的 Event Handler, 用于 override 或者扩展 Channel Client/Server 原有的事件处理逻辑."""


class AbsDuplexChannelServer(ChannelServer, ABC):
    """
    实现一个基础的 Duplex Channel Server, 是为了展示 Channel Client/Server 通讯的基本方式.
    注意:
    1. 有的 channel server, 可以同时有多个 client session 连接它. 有的 server 只能有一个 client session 连接.
    2. 有的 channel 是有状态的, 比如每个 session 的状态都相互隔离. 但有的 channel, 所有的函数应该是可以随便调用的.
    """

    def __init__(
            self,
            container: Container,
            client_connection: Connection,
            server_event_handlers: Dict[str, ChannelEventHandler] | None = None,
    ):
        self.container = container
        """提供的 ioc 容器"""

        self.connection = client_connection
        """从外面传入的 Connection, Channel Server 不关心参数, 只关心交互逻辑. """

        self._server_event_handlers: Dict[str, ChannelEventHandler] = server_event_handlers or {}
        """注册的事件管理."""

        # --- runtime status ---#

        self._closing_event: ThreadSafeEvent = ThreadSafeEvent()
        self._closed_event: ThreadSafeEvent = ThreadSafeEvent()
        self._started: bool = False

        # --- runtime properties ---#

        self.channel: Channel | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._logger: logging.Logger | None = None

        self._running_command_tasks: Dict[str, CommandTask] = {}
        """正在运行, 没有结果的 command tasks"""

        self._running_command_tasks_lock = asyncio.Lock()
        """加个 lock 避免竞态, 不确定是否是必要的."""

        self._channel_lifecycle_tasks: Dict[str, asyncio.Task] = {}
        self._channel_lifecycle_idle_events: Dict[str, asyncio.Event] = {}
        """channel 生命周期的控制任务. """

    @property
    def logger(self) -> logging.Logger:
        """实现一个运行时的 logger. """
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    async def arun_until_closed(self, channel: Channel) -> None:
        """生命周期管理."""
        if self.loop is not None:
            raise RuntimeError(f'{self} is already running')

        # 初始化, 获取 loop 方便后续有线程安全操作.
        self.loop = asyncio.get_running_loop()
        # 获取 channel 本身.
        self.channel = channel
        # 初始化容器.
        await asyncio.to_thread(self.container.bootstrap)
        # 初始化目标 channel, 还有所有的子 channel.
        await self._bootstrap_channels()
        # 启动 connection, 允许被连接.
        await self.connection.start()
        # 运行事件消费逻辑.
        task = await asyncio.create_task(self._main())
        # 标记已经启动.
        self._started = True
        await self.wait_closed()
        if not task.done():
            task.cancel()
        await task

    async def _bootstrap_channels(self) -> None:
        """递归启动所有的 client. """
        client = self.channel.bootstrap(self.container)
        starting = [client.start()]
        for channel in self.channel.descendants().values():
            client = channel.bootstrap(self.container)
            starting.append(client.start())
        await asyncio.gather(*starting)

    def _check_running(self):
        if not self._started:
            raise RuntimeError(f'{self} is not running')

    async def _main(self) -> None:
        try:
            consume_loop_task = asyncio.create_task(self._consume_client_event_loop())
            stop_task = asyncio.create_task(self._closing_event.wait())
            # 主要用来保证, 当 stop 发生的时候, consume loop 应该中断. 这样响应速度应该更快.
            done, pending = asyncio.wait([consume_loop_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            try:
                await consume_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                # 通知 connection 关闭.
                await self.connection.close()

        except asyncio.CancelledError:
            self.logger.info("channel server main loop is cancelled")
        except Exception as e:
            self.logger.exception(e)
        finally:
            if len(self._running_command_tasks) > 0:
                for task in self._running_command_tasks.values():
                    task.cancel()
            if len(self._channel_lifecycle_tasks) > 0:
                for task in self._channel_lifecycle_tasks.values():
                    task.cancel()

            if len(self._channel_lifecycle_idle_events) > 0:
                for event in self._channel_lifecycle_idle_events.values():
                    event.set()
            self._running_command_tasks.clear()
            self._channel_lifecycle_tasks.clear()
            self._channel_lifecycle_idle_events.clear()
            # 通知 session 已经彻底结束了.
            self._closed_event.set()

    async def wait_closed(self) -> None:
        if not self._started:
            return
        self._closing_event.set()
        await self._closed_event.wait()

    async def aclose(self) -> None:
        if self._closing_event.is_set():
            return
        self._closing_event.set()

    # --- consume client event --- #

    async def _consume_client_event_loop(self) -> None:
        while not self._closing_event.is_set():
            event = await self.connection.recv()
            # 所有的事件都异步运行.
            # 如果希望 Channel Server 完全按照阻塞逻辑来执行, 正确的架构设计应该是:
            # 1. 服务端下发 command tokens 流.
            # 2. 本地运行一个 Shell, 消费 command token 生成命令.
            # 3. 本地的 shell 走独立的调度逻辑.
            _ = asyncio.create_task(self._consume_single_event(event))

    async def _consume_single_event(self, event: ChannelEvent) -> None:
        """消费单一事件. 这一层解决 task 生命周期管理. """
        try:
            self.logger.info("Received event: %s", event)
            handle_task = asyncio.create_task(self._handle_single_event(event))
            wait_close = asyncio.create_task(self._closing_event.wait())
            done, pending = await asyncio.wait([handle_task, wait_close], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await handle_task
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _handle_single_event(self, event: ChannelEvent) -> None:
        """做单个事件的异常管理, 理论上不要抛出任何异常. """
        try:
            event_type = event['event_type']
            # 如果有自定义的 event, 先处理.
            if event_type in self._server_event_handlers:
                handler = self._server_event_handlers[event_type]
                # 运行这个 event, 判断是否继续.
                go_on = await handler(self.channel, event)
                if not go_on:
                    return
            # 运行系统默认的 event 处理.
            await self._handle_default_event(event)

        except asyncio.CancelledError:
            # todo: log
            pass
        except ConnectionClosedError:
            # todo: log
            pass
        except FatalError as e:
            self.logger.exception(e)
            self._closing_event.set()
        except Exception as e:
            self.logger.exception(e)

    async def _handle_default_event(self, event: ChannelEvent) -> None:
        # system event
        try:
            if model := CommandCallEvent.from_channel_event(event):
                await self._handle_command_call(model)
            elif model := CommandPeekEvent.from_channel_event(event):
                await self._handle_command_peek(model)
            elif model := CommandCancelEvent.from_channel_event(event):
                await self._handle_command_cancel(model)
            elif model := SyncChannelMetasEvent.from_channel_event(event):
                await self._handle_sync_channel_meta(model)
            elif model := RunPolicyEvent.from_channel_event(event):
                await self._handle_run_policy(model)
            elif model := PausePolicyEvent.from_channel_event(event):
                await self._handle_pause_policy(model)
            elif model := ClearCallEvent.from_channel_event(event):
                await self._handel_clear(model)
            else:
                self.logger.info("Unknown event: %s", event)
        except ValidationError as err:
            self.logger.error("Received invalid event: %s, err: %s", event, err)

    async def _handle_command_peek(self, model: CommandPeekEvent) -> None:
        command_id = model.command_id
        if command_id not in self._running_command_tasks:
            command_done = CommandDoneEvent(
                chan=model.chan,
                command_id=command_id,
                errcode=CommandErrorCode.CANCEL_CODE,
                errmsg="canceled",
                data=None,
            )
            await self._send_response_to_client(command_done.to_channel_event())
        else:
            cmd_task = self._running_command_tasks.pop(command_id)
            if cmd_task.done():
                command_done = CommandDoneEvent(
                    chan=model.chan,
                    command_id=command_id,
                    data=cmd_task.result(),
                    errcode=cmd_task.errcode,
                    errmsg=cmd_task.errmsg,
                )
                await self._send_response_to_client(command_done.to_channel_event())

    async def _handel_clear(self, event: ClearCallEvent):
        """执行 clear 逻辑. """
        channel_name = event.chan
        try:
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return
            await self._cancel_channel_lifecycle_task(channel_name)
            # 执行 clear 命令.
            task = asyncio.create_task(channel.client.clear())
            self._channel_lifecycle_tasks[channel_name] = task
            await task
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            server_error = ServerErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                error="failed to cancel channel %s: %s" % (channel_name, str(e)),
            )
            await self._send_response_to_client(server_error.to_channel_event())
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            # 成功还是失败都是上传.
            response = ClearDoneEvent(
                session_id=event.session_id,
                chan=channel_name,
            )
            await self._send_response_to_client(response.to_channel_event())

    async def _cancel_channel_lifecycle_task(self, chan_name: str) -> None:
        if chan_name not in self._channel_lifecycle_idle_events:
            # 确保注册一个事件.
            event = asyncio.Event()
            event.set()
            self._channel_lifecycle_idle_events[chan_name] = event

        if chan_name in self._channel_lifecycle_tasks:
            task = self._channel_lifecycle_tasks.pop(chan_name)
            task.cancel()
            event = self._channel_lifecycle_idle_events.get(chan_name)
            if event is not None:
                await event.wait()

    async def _clear_channel_lifecycle_task(self, chan_name: str) -> None:
        """清空运行中的 lifecycle task. """
        if chan_name in self._channel_lifecycle_tasks:
            _ = self._channel_lifecycle_tasks.pop(chan_name)
        if chan_name in self._channel_lifecycle_idle_events:
            event = self._channel_lifecycle_idle_events[chan_name]
            event.set()

    async def _handle_run_policy(self, event: RunPolicyEvent) -> None:
        """启动 policy 的运行. """
        channel_name = event.chan
        try:

            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return

            # 先取消生命周期函数.
            await self._cancel_channel_lifecycle_task(channel_name)

            run_policy_task = asyncio.create_task(channel.client.policy_run())
            self._channel_lifecycle_tasks[channel_name] = run_policy_task

            await run_policy_task

        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            server_error = ServerErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                error="failed to run policy of channel %s: %s" % (channel_name, str(e)),
            )
            await self.connection.send(server_error.to_channel_event())
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            response = PausePolicyDoneEvent(session_id=event.session_id, chan=channel_name)
            await self._send_response_to_client(response.to_channel_event())

    async def _send_response_to_client(self, event: ChannelEvent) -> None:
        """做好事件发送的异常管理. """
        try:
            await self.connection.send(event)
        except asyncio.CancelledError:
            raise
        except ConnectionClosedError as e:
            self.logger.exception(e)
            # 关闭整个 channel server.
            self._closing_event.set()
        except Exception as e:
            self.logger.exception(e)

    async def _handle_pause_policy(self, event: PausePolicyEvent) -> None:
        channel_name = event.chan
        try:
            await self._cancel_channel_lifecycle_task(channel_name)
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return

            task = asyncio.create_task(channel.client.policy_pause())
            self._channel_lifecycle_tasks[channel_name] = task
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
            server_error = ServerErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                error="failed to pause policy of channel %s: %s" % (channel_name, str(e)),
            )
            await self.connection.send(server_error.to_channel_event())
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            response = PausePolicyDoneEvent(session_id=event.session_id, chan=channel_name)
            await self._send_response_to_client(response.to_channel_event())

    async def _handle_sync_channel_meta(self, event: SyncChannelMetasEvent) -> None:
        metas = []
        names = set(event.channels)
        for channel in self.channel.all_channels():
            if not channel.is_running():
                continue
            if not names or channel.name in names:
                metas.append(channel.client.meta(no_cache=True))
        response = ChannelMetaUpdateEvent(
            session_id=event.session_id,
            metas=metas,
            root_chan=self.channel.name(),
        )
        await self.connection.send(response.to_channel_event())

    async def _handle_command_cancel(self, event: CommandCancelEvent) -> None:
        cid = event.command_id
        task = self._running_command_tasks.get(cid, None)
        if task is not None:
            self.logger.info("cancel task %s by event %s", task, event)
            # 设置 task 取消.
            task.cancel()

    async def _handle_command_call(self, event: CommandCallEvent) -> None:
        """执行一个命令运行的逻辑. """
        # 先取消 lifecycle 的命令.
        await self._cancel_channel_lifecycle_task(event.chan)
        channel = self.channel.get_channel(event.chan)
        if channel is None:
            response = event.not_available("channel %s not found" % event.chan)
            await self.connection.send(response.to_channel_event())
            return
        elif not self.channel.is_running():
            response = event.not_available("channel %s is not running" % event.chan)
            await self.connection.send(response.to_channel_event())
            return

        # 获取真实的 command 对象.
        command = channel.client.get_command(event.name)
        if command is None or not command.is_available():
            response = event.not_available()
            await self._send_response_to_client(response.to_channel_event())
            return

        task = BaseCommandTask(
            meta=command.meta(),
            func=command.__call__,
            tokens=event.tokens,
            args=event.args,
            kwargs=event.kwargs,
            cid=event.id,
            context=event.context,
        )
        # 真正执行这个 task.
        try:
            task.set_state("running")
            await self._add_running_task(task)
            await self._execute_task(task)
        finally:
            if not task.done():
                task.cancel()
            # todo: log
            await self._remove_running_task(task)
            # todo: 通讯如果存在问题, 会导致阻塞. 需要思考.
            result = task.result()
            response = event.done(result, task.errcode, task.errmsg)
            await self._send_response_to_client(response.to_channel_event())

    async def _execute_task(self, task: CommandTask) -> None:
        try:
            execution = asyncio.create_task(task.dry_run())
            # 如果 task 被提前 cancel 了, 执行命令也会被取消.
            wait_done = asyncio.create_task(task.wait())
            closing = asyncio.create_task(self._closing_event.wait())
            done, pending = await asyncio.wait([execution, wait_done, closing], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            result = await execution
            task.resolve(result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception(e)

    async def _add_running_task(self, task: CommandTask) -> None:
        await self._running_command_tasks_lock.acquire()
        try:
            self._running_command_tasks[task.cid] = task
        finally:
            self._running_command_tasks_lock.release()

    async def _remove_running_task(self, task: CommandTask) -> None:
        await self._running_command_tasks_lock.acquire()
        try:
            cid = task.cid
            if cid in self._running_command_tasks:
                del self._running_command_tasks[cid]
        finally:
            self._running_command_tasks_lock.release()


class DuplexChannelClient(ChannelClient):
    """
    实现一个极简的 Duplex Channel, 它核心是可以通过 ChannelMeta 被动态构建出来.
    """

    def __init__(
            self,
            *,
            container: IoCContainer,
            alias: str,
            channel_id: str,
            session_id: str,
            server_meta: ChannelMeta,
            server_event_queue: asyncio.Queue[ChannelEvent | None],
            client_event_queue: asyncio.Queue[ChannelEvent],
            stop_event: Optional[ThreadSafeEvent],
            local_channel: Optional[Channel],
    ) -> None:
        """
        :param alias: channel 别名.
        :param session_id: 唯一的 session id. 和 server 通讯都必须要携带.
        :param server_meta: 从 server 同步过来的 ChannelMeta.
        :param server_event_queue: 从 server 发送来的事件, 经过队列分发到不同的 channel client.
        :param client_event_queue: 向 server 发送事件的队列.
        :param stop_event: 从上一层传递过来的统一关闭事件.
        :param container: ioc 容器.
        :param local_channel: 是否有本地的 channel, 提供额外的本地方法.
        """
        self.alias = alias
        self.session_id = session_id
        self.id = channel_id
        self.py_channel = local_channel
        self.container = Container(parent=container, name=f"moss/duplex_channel/{self.alias}")
        self.server_event_queue = server_event_queue
        self.client_event_queue = client_event_queue

        # meta 的讯息.
        self._server_chan_meta: ChannelMeta = server_meta
        self._cached_meta: Optional[ChannelMeta] = None

        # 运行时参数
        self._started = False
        self._logger: logging.Logger | None = None
        self._stop_event = stop_event or ThreadSafeEvent()

        self._self_close_event = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._main_loop_done_event = ThreadSafeEvent()
        # runtime
        self._pending_server_command_calls: Dict[str, CommandTask] = {}

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel client {self.alias} is not running')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        if self._cached_meta is not None and not no_cache:
            return self._cached_meta.model_copy()
        self._cached_meta = self._build_meta()
        return self._cached_meta.model_copy()

    def _build_meta(self) -> ChannelMeta:
        self._check_running()
        meta = self._server_chan_meta.model_copy()
        # 从 server meta 中准备 commands 的原型.
        commands = {}
        for command_meta in self._server_chan_meta.commands:
            command_meta = command_meta.model_copy(update={"chan": self.alias})
            commands[command_meta.name] = command_meta
        # 如果有本地注册的函数, 用它们取代 server 同步过来的.
        if self.py_channel is not None:
            local_meta = self.py_channel.client.meta()
            for command_meta in local_meta.commands:
                commands[command_meta.name] = command_meta
        meta.commands = list(commands.values())
        # 修改别名.
        meta.name = self.alias
        return meta

    def is_available(self) -> bool:
        return self.is_running() and self._server_chan_meta.available

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        meta = self.meta(no_cache=False)
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        # 优先尝试从 local channel 中返回.
        if self.py_channel is not None:
            command = self.py_channel.client.get_command(name)
            if command is not None:
                return command.__call__

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            task: CommandTask | None = None
            try:
                task = CommandTask.get_from_context()
            except LookupError:
                pass
            cid = task.cid if task else uuid()
            event = CommandCallEvent(
                session_id=self.session_id,
                name=name,
                # channel 名称使用 server 侧的名称
                chan=self._server_chan_meta.name,
                command_id=cid,
                args=list(args),
                kwargs=dict(kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
            )
            try:
                await self.client_event_queue.put(event.to_channel_event())
            except Exception as e:
                self.logger.exception(e)
                raise e

            wait_result_task = BaseCommandTask(
                meta=meta,
                func=None,
                cid=event.command_id,
                tokens=event.tokens,
                args=event.args,
                kwargs=event.kwargs,
                context=event.context,
            )
            try:
                self._pending_server_command_calls[cid] = wait_result_task
                # 等待异步返回结果.
                await wait_result_task.wait()
                wait_result_task.raise_exception()
                return wait_result_task.result()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(e)
                raise
            finally:
                if cid in self._pending_server_command_calls:
                    t = self._pending_server_command_calls.pop(cid)
                    if not t.done():
                        t.cancel()
                        cancel_event = CommandCancelEvent(
                            session_id=self.session_id,
                            command_id=event.command_id,
                            chan=event.chan,
                        )
                        await self.client_event_queue.put(cancel_event.to_channel_event())

        return _server_caller_as_command

    def get_command(self, name: str) -> Optional[Command]:
        meta = self.meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        func = self._get_command_func(task.meta)
        if func is None:
            raise LookupError(f'Channel {self.alias} can find command {task.meta.name}')
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_run()

            event = RunPolicyEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def policy_pause(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = PausePolicyEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def clear(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = ClearCallEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _consume_server_event_loop(self):
        try:
            while not self._stop_event.is_set() and not self._self_close_event.is_set():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()

    async def _command_peek_loop(self):
        while not self._stop_event.is_set() and not self._self_close_event.is_set():
            if len(self._pending_server_command_calls) > 0:
                tasks = self._pending_server_command_calls.copy()
                for task in tasks.values():
                    peek_event = CommandPeekEvent(
                        chan=task.meta.chan,
                        command_id=task.cid,
                    )
                    await self.client_event_queue.put(peek_event.to_channel_event())
            await asyncio.sleep(1)

    async def _main_loop(self):
        try:
            consume_loop_task = asyncio.create_task(self._consume_server_event_loop())
            command_peek_task = asyncio.create_task(self._command_peek_loop())
            await asyncio.gather(consume_loop_task, command_peek_task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self._main_loop_done_event.set()

    async def _consume_server_event(self):
        try:
            item = await self.server_event_queue.get()
            if item is None:
                self._stop_event.set()
                return

            if model := CommandDoneEvent.from_channel_event(item):
                await self._handle_command_done(model)
            elif model := ChannelMetaUpdateEvent.from_channel_event(item):
                await self._handle_channel_meta_update_event(model)
            elif model := RunPolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s run policy is done from event %s", self.name, model)
            elif model := PausePolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s pause policy is done from event %s", self.name, model)
            elif model := ClearDoneEvent.from_channel_event(item):
                self.logger.info("channel %s clear is done from event %s", self.name, model)
            else:
                self.logger.info('unknown server event %s', item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _handle_channel_meta_update_event(self, event: ChannelMetaUpdateEvent) -> None:
        for meta in event.metas:
            if meta.name == self._server_chan_meta.name:
                self._server_chan_meta = meta.model_copy()
                break

    async def _handle_command_done(self, event: CommandDoneEvent) -> None:
        command_id = event.command_id
        if command_id in self._pending_server_command_calls:
            task = self._pending_server_command_calls[command_id]
            if event.errcode == 0:
                task.resolve(event.data)
            else:
                error = CommandError(event.errcode, event.errmsg)
                task.fail(error)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await asyncio.to_thread(self.container.bootstrap)
        self._main_loop_task = asyncio.create_task(self._main_loop())

    async def close(self) -> None:
        if self._self_close_event.is_set():
            await self._main_loop_done_event.wait()
        self._self_close_event.set()
        self._main_loop_task.cancel()
        await self._main_loop_task
        await asyncio.to_thread(self.container.shutdown)


class DuplexChannelStub(Channel):

    def __init__(
            self,
            *,
            alias: str,
            session_id: str,
            server_meta: ChannelMeta,
            server_event_queue: asyncio.Queue[ChannelEvent | None],
            client_event_queue: asyncio.Queue[ChannelEvent],
            stop_event: Optional[ThreadSafeEvent],
            local_channel: Optional[Channel],
    ) -> None:
        self._alias = alias
        self._session_id = session_id
        self._server_chan_meta = server_meta
        self._server_event_queue = server_event_queue
        self._client_event_queue = client_event_queue
        self._stop_event = stop_event
        self._running_client: Optional[DuplexChannelClient] = None
        self._children: Dict[str, Channel] = {}
        self.local_channel = local_channel or PyChannel(name=self._alias)

    def name(self) -> str:
        return self._alias

    @property
    def client(self) -> ChannelClient:
        if self._running_client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._running_client

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        return self.local_channel.with_children(*children, parent=parent)

    def new_child(self, name: str) -> Self:
        return self.local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        return self.local_channel.children()

    def is_running(self) -> bool:
        return self._running_client is not None and self._running_client.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._running_client is not None:
            raise RuntimeError(f'Channel {self} has already been started.')
        running_client = DuplexChannelClient(
            alias=self._alias,
            session_id=self._session_id,
            channel_id="%s_%s" % (self._session_id, self.name()),
            server_meta=self._server_chan_meta,
            server_event_queue=self._server_event_queue,
            client_event_queue=self._client_event_queue,
            stop_event=self._stop_event,
            local_channel=self.local_channel,
            container=container,
        )
        self._running_client = running_client
        return running_client

    @property
    def build(self) -> Builder:
        return self.local_channel.build


class MainDuplexChannelClient(ChannelClient):
    """双工通道的主 Client, 它的任务是基于通讯的结果, 生成出不同的 channel stub 和对应的 client. """

    def __init__(
            self,
            *,
            name: str,
            server_connection: Connection,
            local_channel: Channel,
            connect_timeout: float = 10.0,
    ):
        self._name = name
        self._server_connection = server_connection
        self._connect_timeout = connect_timeout
        self._local_channel = local_channel
        self._client: Optional[MainDuplexChannelClient] = None

        self._starting = False
        self._started_event = asyncio.Event()
        self._session_id: str = ""

        self._root_channel_stub: Optional[DuplexChannelStub] = None
        self._shared_client_queue = asyncio.Queue()
        self._server_event_dispatch_queues: Dict[str, asyncio.Queue[ChannelEvent | None]] = {}

        self._stop_event = ThreadSafeEvent()

    def is_running(self) -> bool:
        is_ran = self._starting and not self._stop_event.is_set()
        return is_ran and not self._server_connection.is_closed() and self._root_channel_stub is not None

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel {self._name} is not running.')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        return self._root_channel_stub.client.meta(no_cache=no_cache)

    def is_available(self) -> bool:
        if not self.is_running():
            return False
        return self._root_channel_stub.client.is_available()

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        self._check_running()
        return self._root_channel_stub.client.commands(available_only=available_only)

    def get_command(self, name: str) -> Optional[Command]:
        self._check_running()
        return self._root_channel_stub.client.get_command(name)

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        return await self._root_channel_stub.client.execute(task)

    async def policy_run(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.policy_run()

    async def policy_pause(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.policy_pause()

    async def clear(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.clear()

    async def start(self) -> None:
        if self._starting:
            await self._started_event.wait()
            return
        self._starting = True
        self._session_id = uuid()

        # 开始创建连接.
        await self._server_connection.start()
        # 解决授权问题.
        # 要求同步所有的 channel meta.
        await self._server_connection.send(SyncChannelMetasEvent().to_channel_event())

        # 等待第一个 event.
        first_event = await self._server_connection.recv(timeout=self._connect_timeout)

        # 如果第一个 event 不是 meta update client
        meta_update_event = ChannelMetaUpdateEvent.from_channel_event(first_event)
        if meta_update_event is None:
            raise RuntimeError(f'Channel {self._name} server can not be connected.')

        # 开始实例化 channel stub.
        meta_map = {meta.name: meta for meta in meta_update_event.metas}
        root_meta = meta_map.get(meta_update_event.root_chan)
        if root_meta is None:
            raise RuntimeError(f'Channel {self._name} server has no root meta.')

        root_queue = asyncio.Queue()
        self._server_event_dispatch_queues[root_meta.name] = root_queue
        self._root_channel_stub = DuplexChannelStub(
            alias=self._name,
            session_id=self._session_id,
            server_meta=root_meta,
            server_event_queue=root_queue,
            client_event_queue=self._shared_client_queue,
            stop_event=self._stop_event,
            local_channel=self._local_channel,
        )
        await self._root_channel_stub.bootstrap(self.container).start()

        # 递归启动子孙.
        start_all_children = []
        for child_name in root_meta.children:
            cor = self._recursive_start_channel_from_metas(self._root_channel_stub, child_name, meta_map)
            start_all_children.append(cor)
        await asyncio.gather(*start_all_children)

        # 标记启动完成.
        self._started_event.set()

    async def _recursive_start_channel_from_metas(
            self,
            parent: Channel,
            name: str,
            meta_maps: Dict[str, ChannelMeta],
    ) -> None:
        local_channel = self._local_channel.get_channel(name)
        current_meta = meta_maps.get(name)
        if current_meta is None:
            return
        queue = asyncio.Queue()
        self._server_event_dispatch_queues[current_meta.name] = queue
        channel_stub = DuplexChannelStub(
            alias=self._name,
            session_id=self._session_id,
            server_meta=current_meta,
            server_event_queue=queue,
            client_event_queue=self._shared_client_queue,
            stop_event=self._stop_event,
            local_channel=local_channel,
        )

        # 父节点挂载自身.
        parent.with_children(channel_stub)
        await channel_stub.bootstrap(self.container).start()

        recursive_start = []
        for child_name in current_meta.children:
            recursive_start.append(self._recursive_start_channel_from_metas(channel_stub, child_name, meta_maps))
        await asyncio.gather(*recursive_start)

    async def close(self) -> None:
        if self._stop_event.is_set():
            return
        # 同时也会通知所有子节点.
        self._stop_event.set()
        await self._root_channel_stub.client.close()


class MainDuplexChannel(Channel, ABC):

    def __init__(
            self,
            server_connection: Connection,
            block: bool,
            name: str,
            description: str = "",
    ):
        self._name = name
        self._server_connection = server_connection
        self._local_channel = PyChannel(name=name, description=description, block=block)
        self._client: Optional[MainDuplexChannelClient] = None
        self._children: Dict[str, Channel] = {}

    def name(self) -> str:
        return self._name

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._client

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        self._local_channel.with_children(*children, parent=parent)
        return self

    def new_child(self, name: str) -> Self:
        return self._local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        return self._children.copy()

    def is_running(self) -> bool:
        return self._client is not None and self._client.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self} has already been started.')
        client = MainDuplexChannelClient(
            name=self._name,
            server_connection=self._server_connection,
            local_channel=self._local_channel,
        )
        self._client = client
        return client

    @property
    def build(self) -> Builder:
        return self._local_channel.build
