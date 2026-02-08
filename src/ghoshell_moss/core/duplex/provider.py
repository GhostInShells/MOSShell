import asyncio
import logging
from collections.abc import Callable, Coroutine

from ghoshell_common.helpers import uuid
from ghoshell_container import Container
from pydantic import ValidationError

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from ghoshell_moss.core.concepts.command import BaseCommandTask, CommandTask
from ghoshell_moss.core.concepts.errors import CommandErrorCode, FatalError
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

from .connection import Connection, ConnectionClosedError, ConnectionNotAvailable
from .protocol import (
    ChannelEvent,
    ChannelMetaUpdateEvent,
    ClearCallEvent,
    ClearDoneEvent,
    CommandCallEvent,
    CommandCancelEvent,
    CommandDoneEvent,
    CommandPeekEvent,
    CreateSessionEvent,
    PausePolicyDoneEvent,
    PausePolicyEvent,
    ProviderErrorEvent,
    ReconnectSessionEvent,
    RunPolicyDoneEvent,
    RunPolicyEvent,
    SessionCreatedEvent,
    SyncChannelMetasEvent,
)

__all__ = ["ChannelEventHandler", "DuplexChannelProvider"]

# --- event handlers --- #

ChannelEventHandler = Callable[[Channel, ChannelEvent], Coroutine[None, None, bool]]
""" 自定义的 Event Handler, 用于 override 或者扩展 Channel Client/Server 原有的事件处理逻辑."""


class DuplexChannelProvider(ChannelProvider):
    """
    实现一个基础的 Duplex Channel Server, 是为了展示 Channel Client/Server 通讯的基本方式.
    注意:
    1. 有的 channel server, 可以同时有多个 broker session 连接它. 有的 server 只能有一个 broker session 连接.
    2. 有的 channel 是有状态的, 比如每个 session 的状态都相互隔离. 但有的 channel, 所有的函数应该是可以随便调用的.
    """

    def __init__(
        self,
        container: Container,
        provider_connection: Connection,
        proxy_event_handlers: dict[str, ChannelEventHandler] | None = None,
        receive_interval_seconds: float = 0.5,
    ):
        self.container = container
        """提供的 ioc 容器"""

        self.connection = provider_connection
        """从外面传入的 Connection, Channel Server 不关心参数, 只关心交互逻辑. """

        self._proxy_event_handlers: dict[str, ChannelEventHandler] = proxy_event_handlers or {}
        """注册的事件管理."""

        # --- runtime status ---#
        self._receive_interval_seconds = receive_interval_seconds
        self._closing_event: ThreadSafeEvent = ThreadSafeEvent()
        self._closed_event: ThreadSafeEvent = ThreadSafeEvent()

        # --- connect session --- #

        self._session_id: str | None = None
        """当前连接的 session id"""
        self._session_creating_event: asyncio.Event = asyncio.Event()

        self._starting: bool = False

        # --- runtime properties ---#

        self.channel: Channel | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._logger: logging.Logger | None = None

        self._running_command_tasks: dict[str, CommandTask] = {}
        """正在运行, 没有结果的 command tasks"""

        self._running_command_tasks_lock = asyncio.Lock()
        """加个 lock 避免竞态, 不确定是否是必要的."""

        self._channel_lifecycle_tasks: dict[str, asyncio.Task] = {}
        self._channel_lifecycle_idle_events: dict[str, asyncio.Event] = {}
        """channel 生命周期的控制任务. """

        self._main_task: asyncio.Task | None = None

    @property
    def logger(self) -> logging.Logger:
        """实现一个运行时的 logger."""
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    async def arun(self, channel: Channel) -> None:
        if self._starting:
            self.logger.info(
                "DuplexChannelProvider[cls=%s] already started, channel=%s", self.__class__.__name__, channel.name()
            )
            return
        self.logger.info("DuplexChannelProvider[cls=%s] starting, channel=%s", self.__class__.__name__, channel.name())
        self._starting = True
        self.loop = asyncio.get_running_loop()
        self.channel = channel
        try:
            # 初始化容器.
            await asyncio.to_thread(self.container.bootstrap)
            # 初始化目标 channel, 还有所有的子 channel.
            await self._bootstrap_channels()
            # 启动 connection, 允许被连接.
            await self.connection.start()
            # 运行事件消费逻辑.
            self._main_task = asyncio.create_task(self._main())
            self.logger.info(
                "DuplexChannelProvider[cls=%s] started, channel=%s", self.__class__.__name__, channel.name()
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("DuplexChannelProvider start failed")
            raise

    async def _bootstrap_channels(self) -> None:
        """递归启动所有的 broker."""
        broker = self.channel.bootstrap(self.container)
        starting = [broker.start()]
        for channel in self.channel.descendants().values():
            broker = channel.bootstrap(self.container)
            starting.append(broker.start())
        await asyncio.gather(*starting)

    def _check_running(self):
        if not self._starting:
            raise RuntimeError(f"{self} is not running")

    async def _main(self) -> None:
        try:
            consume_loop_task = asyncio.create_task(self._consume_proxy_event_loop())
            stop_task = asyncio.create_task(self._closing_event.wait())
            # 主要用来保证, 当 stop 发生的时候, consume loop 应该中断. 这样响应速度应该更快.
            done, pending = await asyncio.wait(
                [consume_loop_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            try:
                await consume_loop_task
            except asyncio.CancelledError:
                pass

        except asyncio.CancelledError:
            self.logger.info("channel server main loop is cancelled")
        except Exception:
            self.logger.exception("DuplexChannelProvider main loop failed")
            raise
        finally:
            await self._clear_running_status()
            await self.connection.close()
            close_all_channels = []
            for channel in self.channel.all_channels().values():
                if channel.is_running():
                    close_all_channels.append(channel.broker.close())
            await asyncio.gather(*close_all_channels)
            await asyncio.to_thread(self.container.shutdown)
            # 通知 session 已经彻底结束了.
            self._closed_event.set()

    async def _clear_running_status(self) -> None:
        """
        清空运行状态.
        """
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
        clearing = []
        for channel in self.channel.all_channels().values():
            if channel.is_running():
                clearing.append(channel.broker.clear())
        done = await asyncio.gather(*clearing, return_exceptions=True)
        for val in done:
            if isinstance(val, Exception):
                self.logger.exception("clear channel error")

    async def wait_closed(self) -> None:
        if not self._starting:
            return
        await self._closed_event.wait()

    def wait_closed_sync(self) -> None:
        self._closed_event.wait_sync()

    async def aclose(self) -> None:
        if self._closing_event.is_set():
            await self._closed_event.wait()
            return
        self._closing_event.set()
        try:
            if self._main_task is not None:
                await self._main_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("DuplexChannelProvider close failed")
            raise
        finally:
            await self._closing_event.wait()

    def is_running(self) -> bool:
        return self._starting and not (self._closing_event.is_set() or self._closed_event.is_set())

    # --- consume broker event --- #

    async def _clear_session_status(self) -> None:
        if self._session_id:
            self._session_id = None
            await self._clear_running_status()

    async def _sync_session(self, new: bool) -> None:
        if new or not self._session_id:
            self._session_id = uuid()
            self._session_creating_event.clear()
        try:
            event = CreateSessionEvent(session_id=self._session_id).to_channel_event()
            await self._send_event_to_proxy(event)
            self._session_creating_event.set()
        except asyncio.CancelledError:
            pass
        except (ConnectionNotAvailable, ConnectionClosedError):
            pass

    async def _consume_proxy_event_loop(self) -> None:
        try:
            while not self._closing_event.is_set():
                if not self.connection.is_available():
                    # 连接未成功, 则清空等待状态. 需要重新创建 session.
                    await self._clear_session_status()
                    # 进行下一轮检查.
                    await asyncio.sleep(self._receive_interval_seconds)
                    continue

                if not self._session_id:
                    # 没有创建过 session, 则尝试创建 session.
                    await self._sync_session(new=True)
                    continue

                try:
                    event = await self.connection.recv(timeout=self._receive_interval_seconds)
                except asyncio.TimeoutError:
                    continue
                except ConnectionNotAvailable:
                    # 保持重连.
                    continue

                if event is None:
                    break
                # todo: 添加 debug 日志.

                if created := SessionCreatedEvent.from_channel_event(event):
                    # proxy 声明创建 Session 成功.
                    if created.session_id == self._session_id:
                        self._session_creating_event.set()
                        # 开始同步 channel metas.
                        sync_meta = SyncChannelMetasEvent(
                            session_id=self._session_id,
                        )
                        await self._handle_sync_channel_meta(sync_meta)
                    else:
                        # 继续提醒云端重建 session.
                        await self._sync_session(new=False)
                    continue
                elif reconnected := ReconnectSessionEvent.from_channel_event(event):
                    # session id 不对齐, 重新建立 session.
                    if reconnected.session_id != self._session_id:
                        await self._clear_session_status()
                        await self._sync_session(new=len(reconnected.session_id) > 0)
                    continue

                if event["session_id"] != self._session_id:
                    # 丢弃不同 session 的事件.
                    self.logger.info("channel session %s mismatch, drop event %s", self._session_id, event)
                    # 频繁要求服务端同步 session.
                    await self._sync_session(new=False)
                    continue

                # 所有的事件都异步运行.
                # 如果希望 Channel Server 完全按照阻塞逻辑来执行, 正确的架构设计应该是:
                # 1. 服务端下发 command tokens 流.
                # 2. 本地运行一个 Shell, 消费 command token 生成命令.
                # 3. 本地的 shell 走独立的调度逻辑.
                _ = asyncio.create_task(self._consume_single_event(event))
        except asyncio.CancelledError:
            self.logger.warning("Consume broker event loop is cancelled")
        except ConnectionClosedError:
            self.logger.warning("Consume broker event loop is closed")
        except Exception:
            self.logger.exception("Consume broker event loop failed")
            raise

    async def _consume_single_event(self, event: ChannelEvent) -> None:
        """消费单一事件. 这一层解决 task 生命周期管理."""
        try:
            self.logger.info("Received event: %s", event)
            handle_task = asyncio.create_task(self._handle_single_event(event))
            wait_close = asyncio.create_task(self._closing_event.wait())
            done, pending = await asyncio.wait([handle_task, wait_close], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await handle_task
        except Exception:
            self.logger.exception("Handle event task failed")

    async def _handle_single_event(self, event: ChannelEvent) -> None:
        """做单个事件的异常管理, 理论上不要抛出任何异常."""
        try:
            event_type = event["event_type"]
            # 如果有自定义的 event, 先处理.
            if event_type in self._proxy_event_handlers:
                handler = self._proxy_event_handlers[event_type]
                # 运行这个 event, 判断是否继续.
                go_on = await handler(self.channel, event)
                if not go_on:
                    return
            # 运行系统默认的 event 处理.
            await self._handle_default_event(event)

        except asyncio.CancelledError:
            # todo: log
            pass
        except FatalError:
            self.logger.exception("Fatal error while handling event")
            self._closing_event.set()
        except Exception:
            self.logger.exception("Unhandled error while handling event")

    async def _handle_default_event(self, event: ChannelEvent) -> None:
        # system event
        try:
            if model := CommandCallEvent.from_channel_event(event):
                await self._handle_command_call(model)
            elif model := CommandPeekEvent.from_channel_event(event):
                pass
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
        except ValidationError:
            self.logger.exception("Received invalid event: %s", event)
        except Exception:
            self.logger.exception("Handle default event failed")
            raise
        finally:
            self.logger.info("handled event: %s", event)

    async def _handle_command_peek(self, model: CommandPeekEvent) -> None:
        command_id = model.command_id
        if command_id not in self._running_command_tasks:
            command_done = CommandDoneEvent(
                chan=model.chan,
                command_id=command_id,
                errcode=CommandErrorCode.NOT_FOUND.value,
                errmsg="canceled",
                result=None,
            )
            # todo: log
            await self._send_event_to_proxy(command_done.to_channel_event())
        else:
            cmd_task = self._running_command_tasks.get(command_id)
            # todo: log
            if cmd_task.done():
                command_done = CommandDoneEvent(
                    chan=model.chan,
                    command_id=command_id,
                    errcode=int(cmd_task.errcode),
                    errmsg=cmd_task.errmsg,
                    result=cmd_task.result(),
                )
                await self._send_event_to_proxy(command_done.to_channel_event())

    async def _handel_clear(self, event: ClearCallEvent):
        """执行 clear 逻辑."""
        channel_name = event.chan
        try:
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.broker.is_available():
                return
            await self._cancel_channel_lifecycle_task(channel_name)
            # 执行 clear 命令.
            task = asyncio.create_task(channel.broker.clear())
            self._channel_lifecycle_tasks[channel_name] = task
            await task
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Clear channel failed")
            server_error = ProviderErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                errmsg=f"failed to cancel channel {channel_name}",
            )
            await self._send_event_to_proxy(server_error.to_channel_event())
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            # 成功还是失败都是上传.
            response = ClearDoneEvent(
                session_id=event.session_id,
                chan=channel_name,
            )
            await self._send_event_to_proxy(response.to_channel_event())

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
        """清空运行中的 lifecycle task."""
        if chan_name in self._channel_lifecycle_tasks:
            _ = self._channel_lifecycle_tasks.pop(chan_name)
        if chan_name in self._channel_lifecycle_idle_events:
            event = self._channel_lifecycle_idle_events[chan_name]
            event.set()

    async def _handle_run_policy(self, event: RunPolicyEvent) -> None:
        """启动 policy 的运行."""
        channel_name = event.chan
        session_id = self._session_id
        try:
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.broker.is_available():
                return

            # 先取消生命周期函数.
            await self._cancel_channel_lifecycle_task(channel_name)

            run_policy_task = asyncio.create_task(channel.broker.policy_run())
            self._channel_lifecycle_tasks[channel_name] = run_policy_task

            await run_policy_task

        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception:
            self.logger.exception("Run policy failed")
            server_error = ProviderErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                errmsg=f"failed to run policy of channel {channel_name}",
            )
            await self._send_event_to_proxy(server_error.to_channel_event(), session_id=session_id)
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            response = RunPolicyDoneEvent(session_id=event.session_id)
            await self._send_event_to_proxy(response.to_channel_event(), session_id=session_id)

    async def _send_event_to_proxy(self, event: ChannelEvent, session_id: str = "") -> None:
        """做好事件发送的异常管理."""
        try:
            event["session_id"] = session_id or self._session_id or ""
            await self.connection.send(event)
        except asyncio.CancelledError:
            raise
        except ConnectionNotAvailable:
            await self._clear_session_status()

        except ConnectionClosedError:
            self.logger.exception("Connection closed while sending event")
            # 关闭整个 channel server.
            self._closing_event.set()
        except Exception:
            self.logger.exception("Send event failed")

    async def _handle_pause_policy(self, event: PausePolicyEvent) -> None:
        channel_name = event.chan
        try:
            await self._cancel_channel_lifecycle_task(channel_name)
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.broker.is_available():
                return

            task = asyncio.create_task(channel.broker.policy_pause())
            self._channel_lifecycle_tasks[channel_name] = task
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("Pause policy failed")
            server_error = ProviderErrorEvent(
                session_id=event.session_id,
                # todo
                errcode=-1,
                errmsg=f"failed to pause policy of channel {channel_name}",
            )
            await self._send_event_to_proxy(server_error.to_channel_event())
        finally:
            await self._clear_channel_lifecycle_task(channel_name)
            response = PausePolicyDoneEvent(session_id=event.session_id, chan=channel_name)
            await self._send_event_to_proxy(response.to_channel_event())

    async def _handle_sync_channel_meta(self, event: SyncChannelMetasEvent) -> None:
        metas = {}
        all_channels = self.channel.all_channels().values()
        refresh_tasks = []

        # 并发刷新所有的 channel metas.
        for channel in all_channels:
            if channel.is_running() and channel.broker.is_available():
                refresh_tasks.append(channel.broker.refresh_meta())
        await asyncio.gather(*refresh_tasks)

        for channel_path, channel in self.channel.all_channels().items():
            if not channel.is_running():
                continue
            metas[channel_path] = channel.broker.meta()
        response = ChannelMetaUpdateEvent(
            session_id=event.session_id,
            metas=metas,
            root_chan=self.channel.name(),
        )
        await self._send_event_to_proxy(response.to_channel_event())

    async def _handle_command_cancel(self, event: CommandCancelEvent) -> None:
        cid = event.command_id
        task = self._running_command_tasks.get(cid, None)
        if task is not None:
            self.logger.info("cancel task %s by event %s", task, event)
            # 设置 task 取消.
            task.cancel()

    async def _handle_command_call(self, call_event: CommandCallEvent) -> None:
        """执行一个命令运行的逻辑."""
        # 先取消 lifecycle 的命令.
        await self._cancel_channel_lifecycle_task(call_event.chan)
        channel = self.channel.get_channel(call_event.chan)
        if channel is None:
            response = call_event.not_available(f"channel `{call_event.chan}` not found")
            await self._send_event_to_proxy(response.to_channel_event())
            return
        elif not self.channel.is_running():
            response = call_event.not_available(f"channel `{call_event.chan}` is not running")
            await self._send_event_to_proxy(response.to_channel_event())
            return

        # 获取真实的 command 对象.
        command = channel.broker.get_command(call_event.name)
        if command is None or not command.is_available():
            response = call_event.not_available()
            await self._send_event_to_proxy(response.to_channel_event())
            return

        task = BaseCommandTask(
            meta=command.meta(),
            func=command.__call__,
            tokens=call_event.tokens,
            args=call_event.args,
            kwargs=call_event.kwargs,
            cid=call_event.command_id,
            context=call_event.context,
        )
        # 真正执行这个 task.
        try:
            # 多余的, 没什么用.
            task.set_state("running")
            await self._add_running_task(task)
            result = await channel.execute_task(task)
            task.resolve(result)
        except asyncio.CancelledError:
            task.cancel("cancelled")
            pass
        except Exception as e:
            self.logger.exception("Execute command failed")
            task.fail(e)
        finally:
            # todo: log
            await self._remove_running_task(task)
            if not task.done():
                task.cancel()
            # todo: 通讯如果存在问题, 会导致阻塞. 需要思考.
            result = task.result()
            response = call_event.done(result, task.errcode, task.errmsg)
            await self._send_event_to_proxy(response.to_channel_event())

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

    def close(self) -> None:
        if self._closing_event.is_set():
            return
        self._closing_event.set()
