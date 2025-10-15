from typing import Dict, Callable, Coroutine

from ghoshell_moss.concepts.channel import Channel, ChannelServer
from ghoshell_moss.concepts.errors import FatalError, CommandErrorCode
from ghoshell_moss.concepts.command import CommandTask, BaseCommandTask
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from .protocol import *
from .connection import *
from ghoshell_container import Container
from pydantic import ValidationError
import logging
import asyncio

__all__ = ['ChannelEventHandler', 'DuplexChannelServer']

# --- event handlers --- #

ChannelEventHandler = Callable[[Channel, ChannelEvent], Coroutine[None, None, bool]]
""" 自定义的 Event Handler, 用于 override 或者扩展 Channel Client/Server 原有的事件处理逻辑."""


class DuplexChannelServer(ChannelServer):
    """
    实现一个基础的 Duplex Channel Server, 是为了展示 Channel Client/Server 通讯的基本方式.
    注意:
    1. 有的 channel server, 可以同时有多个 client session 连接它. 有的 server 只能有一个 client session 连接.
    2. 有的 channel 是有状态的, 比如每个 session 的状态都相互隔离. 但有的 channel, 所有的函数应该是可以随便调用的.
    """

    def __init__(
            self,
            container: Container,
            to_client_connection: Connection,
            client_event_handlers: Dict[str, ChannelEventHandler] | None = None,
    ):
        self.container = container
        """提供的 ioc 容器"""

        self.connection = to_client_connection
        """从外面传入的 Connection, Channel Server 不关心参数, 只关心交互逻辑. """

        self._client_event_handlers: Dict[str, ChannelEventHandler] = client_event_handlers or {}
        """注册的事件管理."""

        # --- runtime status ---#

        self._closing_event: ThreadSafeEvent = ThreadSafeEvent()
        self._closed_event: ThreadSafeEvent = ThreadSafeEvent()
        self._starting: bool = False

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

        self._main_task: asyncio.Task | None = None

    @property
    def logger(self) -> logging.Logger:
        """实现一个运行时的 logger. """
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    async def arun(self, channel: Channel) -> None:
        if self._starting:
            return
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
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
            raise

    async def _bootstrap_channels(self) -> None:
        """递归启动所有的 client. """
        client = self.channel.bootstrap(self.container)
        starting = [client.start()]
        for channel in self.channel.descendants().values():
            client = channel.bootstrap(self.container)
            starting.append(client.start())
        await asyncio.gather(*starting)

    def _check_running(self):
        if not self._starting:
            raise RuntimeError(f'{self} is not running')

    async def _main(self) -> None:
        try:
            consume_loop_task = asyncio.create_task(self._consume_client_event_loop())
            stop_task = asyncio.create_task(self._closing_event.wait())
            # 主要用来保证, 当 stop 发生的时候, consume loop 应该中断. 这样响应速度应该更快.
            done, pending = await asyncio.wait([consume_loop_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            try:
                await consume_loop_task
            except asyncio.CancelledError:
                pass

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
            await self.connection.close()
            close_all_channels = []
            for channel in self.channel.all_channels():
                if channel.is_running():
                    close_all_channels.append(channel.client.close())
            await asyncio.gather(*close_all_channels)
            await asyncio.to_thread(self.container.shutdown)
            # 通知 session 已经彻底结束了.
            self._closed_event.set()

    async def wait_closed(self) -> None:
        if not self._starting:
            return
        await self._closed_event.wait()

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
        except Exception as e:
            self.logger.exception(e)
            raise


    def is_running(self) -> bool:
        return self._starting and not (self._closing_event.is_set() or self._closed_event.is_set())

    # --- consume client event --- #

    async def _consume_client_event_loop(self) -> None:
        try:
            while not self._closing_event.is_set():
                event = await self.connection.recv()
                # 所有的事件都异步运行.
                # 如果希望 Channel Server 完全按照阻塞逻辑来执行, 正确的架构设计应该是:
                # 1. 服务端下发 command tokens 流.
                # 2. 本地运行一个 Shell, 消费 command token 生成命令.
                # 3. 本地的 shell 走独立的调度逻辑.
                _ = asyncio.create_task(self._consume_single_event(event))
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

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
            if event_type in self._client_event_handlers:
                handler = self._client_event_handlers[event_type]
                # 运行这个 event, 判断是否继续.
                go_on = await handler(self.channel, event)
                if not go_on:
                    return
            # 运行系统默认的 event 处理.
            await self._handle_default_event(event)

        except asyncio.CancelledError:
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
        except Exception as e:
            self.logger.exception(e)
            raise
        finally:
            self.logger.info('handled event: %s', event)

    async def _handle_command_peek(self, model: CommandPeekEvent) -> None:
        command_id = model.command_id
        if command_id not in self._running_command_tasks:
            command_done = CommandDoneEvent(
                chan=model.chan,
                command_id=command_id,
                errcode=CommandErrorCode.CANCELLED,
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
            cid=event.command_id,
            context=event.context,
        )
        # 真正执行这个 task.
        try:
            task.set_state("running")
            await self._add_running_task(task)
            await self._execute_task(task)
        finally:
            # todo: log
            await self._remove_running_task(task)
            if not task.done():
                task.cancel()
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

    def close(self) -> None:
        if self._closing_event.is_set():
            return
        self._closing_event.set()
