import asyncio
import contextlib
import logging
import threading
from typing import Callable, Coroutine, Optional, AsyncIterator
from typing_extensions import Self
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer
from pydantic import ValidationError

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider, ChannelRuntime
from ghoshell_moss.core.concepts.command import BaseCommandTask, CommandTask, CommandToken, CommandTaskState
from ghoshell_moss.core.concepts.errors import FatalError, CommandErrorCode
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.topic import QueueBasedTopicService, TopicService, Topic
from ghoshell_moss.core.helpers.stream import (
    create_sender_and_receiver,
    ThreadSafeStreamReceiver,
    ThreadSafeStreamSender,
)

from .connection import Connection, ConnectionClosedError, ConnectionNotAvailable
from .protocol import (
    ChannelEvent,
    ChannelMetaUpdateEvent,
    ClearEvent,
    CommandCallEvent,
    CommandDeltaEvent,
    CommandCancelEvent,
    CreateSessionEvent,
    ProviderErrorEvent,
    ReconnectSessionEvent,
    SessionCreatedEvent,
    SyncChannelMetasEvent,
    ProviderPubTopicEvent,
    ProviderSubTopicEvent,
    ProxyPubTopicEvent,
)

__all__ = ["ChannelEventHandler", "DuplexChannelProvider"]

# --- event handlers --- #

ChannelEventHandler = Callable[[Channel, ChannelEvent], Coroutine[None, None, bool]]
""" 自定义的 Event Handler, 用于 override 或者扩展 Channel proxy/provider 原有的事件处理逻辑."""


class ProviderTopicService(QueueBasedTopicService):
    """专门为 provider 准备的 topic."""

    def __init__(
            self,
            get_connection_id: Callable[[], str],
            connection: Connection,
            sender: str = "",
            *,
            logger: LoggerItf | None = None,
    ):
        super().__init__(sender=sender, logger=logger)
        self._connection = connection
        self._get_connection_id_fn = get_connection_id

    async def _on_topic_published(self, topic: Topic) -> None:
        try:
            if self._connection.is_connected() and not self._connection.is_closed():
                # 不会跨网络传输.
                if topic.meta.local:
                    return
                event = ProviderPubTopicEvent(topic=topic, connection_id=self._get_connection_id_fn())
                await self._connection.send(event.to_channel_event())
        except (ConnectionClosedError, ConnectionNotAvailable):
            pass

    async def _on_topic_subscribed(self, topic_name: str) -> None:
        try:
            if self._connection.is_connected() and not self._connection.is_closed():
                event = ProviderSubTopicEvent(topic_name=topic_name, connection_id=self._get_connection_id_fn())
                await self._connection.send(event.to_channel_event())
        except (ConnectionClosedError, ConnectionNotAvailable):
            pass


class DuplexChannelProvider(ChannelProvider):
    """
    实现一个基础的 Duplex Channel provider, 实现 Channel proxy/provider 通讯的基本方式.
    """

    def __init__(
            self,
            provider_connection: Connection,
            proxy_event_handlers: dict[str, ChannelEventHandler] | None = None,
            reconnect_interval_seconds: float = 0.5,
            container: Container = None,
    ):
        self._uid = uuid()
        self._container = Container(
            name=f"moss.duplex_provider.{self.__class__.__name__}",
            parent=container,
        )
        """提供的 ioc 容器"""

        self._connection = provider_connection
        """从外面传入的 Connection, Channel provider 不关心参数, 只关心交互逻辑. """

        self._proxy_event_handlers: dict[str, ChannelEventHandler] = proxy_event_handlers or {}
        """注册的事件管理."""

        # --- runtime status ---#
        self._reconnect_interval_seconds = reconnect_interval_seconds
        self._stopping_event: ThreadSafeEvent = ThreadSafeEvent()
        self._closed_event: ThreadSafeEvent = ThreadSafeEvent()

        # --- connect connection --- #

        self._connection_id: str | None = None
        """当前连接的 connection id"""
        self._connection_creating_event: asyncio.Event = asyncio.Event()

        self._starting: bool = False

        # --- runtime properties ---#

        self._root_runtime: Optional[ChannelRuntime] = None
        self._channel: Channel | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._logger: logging.Logger | None = None
        self._log_prefix = "[DuplexChannelProvider %s %s]" % (self.__class__.__name__, self._uid)

        self._running_command_tasks: dict[str, CommandTask] = {}
        self._running_command_delta_stream: dict[str, tuple[ThreadSafeStreamSender, ThreadSafeStreamReceiver]] = {}
        """正在运行, 没有结果的 command tasks"""

        self._running_command_tasks_lock = asyncio.Lock()
        """加个 lock 避免竞态, 不确定是否是必要的."""
        self._provider_topic_service: Optional[TopicService] = None
        self._main_loop_task: asyncio.Task | None = None
        self._running_thread: threading.Thread | None = None
        self._running_task: asyncio.Task | None = None

    def _get_connection_id(self) -> str:
        return self._connection_id or ""

    @property
    def logger(self) -> LoggerItf:
        """实现一个运行时的 logger."""
        if self._logger is None:
            self._logger = self._container.get(LoggerItf) or logging.getLogger("moss")
        return self._logger

    @property
    def channel(self) -> Channel:
        if self._channel is None:
            raise RuntimeError("Channel provider has not been initialized.")
        return self._channel

    @property
    def runtime(self) -> ChannelRuntime:
        if self._root_runtime is None:
            raise RuntimeError("Channel provider has not been initialized.")
        return self._root_runtime

    @property
    def container(self) -> IoCContainer:
        return self._container

    @contextlib.asynccontextmanager
    async def _bootstrap_container_stack(self) -> AsyncIterator[None]:
        try:
            await asyncio.to_thread(self._container.bootstrap)
            yield
        finally:
            await asyncio.to_thread(self._container.shutdown)

    @contextlib.asynccontextmanager
    async def _bootstrap_runtime_stack(self) -> AsyncIterator[None]:
        try:
            await self._root_runtime.start()
            yield
        finally:
            await self._root_runtime.close()

    @contextlib.asynccontextmanager
    async def _bootstrap_connection_stack(self) -> AsyncIterator[None]:
        try:
            await self._connection.start()
            yield
        finally:
            try:
                await self._connection.close()
            except Exception as exc:
                self.logger.exception("%s close connection failed: %s", self._log_prefix, exc)

    @contextlib.asynccontextmanager
    async def _bootstrap_main_loop_stack(self) -> AsyncIterator[None]:
        try:
            # 运行事件消费逻辑.
            await self._clear_running_status()
            self._main_loop_task = asyncio.create_task(self._main_loop())
            yield
        finally:
            try:
                if not self._main_loop_task.done():
                    self._main_loop_task.cancel()
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.logger.exception("%s close main loop task failed: %s", self._log_prefix, exc)

    @contextlib.asynccontextmanager
    async def arun(self, channel: Channel) -> AsyncIterator[Self]:
        if self._starting:
            self.logger.info(f"%s already started, channel=%s", self._log_prefix, channel.name())
            raise RuntimeError(f"Channel {channel.name()} already started.")

        self._starting = True
        self.logger.info(f"%s start to run, channel=%s", self._log_prefix, channel.name())
        self._loop = asyncio.get_running_loop()
        self._channel = channel

        # 注册 topic service.
        if not self._container.bound(TopicService):
            # 只有在 container 没有提供 topic service 时, 才会自行创建一个基于双工通讯的 topic service.
            # 这时, 所有的 topics 会通过 channel 的通道去传递.
            # 这个实现准备移除. 预计所有的通讯组件不提供的话, 都保持本地化.
            # todo: 向前兼容只保留 Topic, 预计正式版本删除.
            self._provider_topic_service = ProviderTopicService(
                self._get_connection_id,
                self._connection,
                sender=f"DuplexChannelProvider/{self._uid}",
                logger=self.logger,
            )
            self._container.set(
                TopicService,
                self._provider_topic_service,
            )
        # 启动时, topic service 同样会注入到根节点的 importlib 中.
        self._root_runtime = channel.bootstrap(self._container)

        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self._bootstrap_container_stack())
            await stack.enter_async_context(self._bootstrap_runtime_stack())
            await stack.enter_async_context(self._bootstrap_connection_stack())
            await stack.enter_async_context(self._bootstrap_main_loop_stack())
            try:
                yield self
            except Exception as exc:
                self.logger.exception("%s close channel task failed: %s", self._log_prefix, exc)
            finally:
                self._closed_event.set()

    def _check_running(self):
        if not self._starting:
            raise RuntimeError(f"{self} is not running")

    async def _main_loop(self) -> None:
        """
        provider 生命周期中的主循环.
        """
        try:
            consume_loop_task = asyncio.create_task(self._consume_proxy_event_loop())
            stop_task = asyncio.create_task(self._stopping_event.wait())
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
            self.logger.info("%s provider main loop is cancelled", self._log_prefix)
        except Exception as e:
            self.logger.exception("%s main loop failed %s", self._log_prefix, e)
            raise
        finally:
            self.logger.info("%s provider main loop is finally done", self._log_prefix)

    async def _clear_running_status(self) -> None:
        """
        清空运行状态.
        """
        self._connection.clear()
        if len(self._running_command_tasks) > 0:
            for task in self._running_command_tasks.values():
                if not task.done():
                    task.cancel()
        self._running_command_tasks.clear()
        if len(self._running_command_delta_stream) > 0:
            for sender, receiver in self._running_command_delta_stream.values():
                sender.fail(CommandErrorCode.CLEARED.error("cleared"))
        self._running_command_delta_stream.clear()
        await self._root_runtime.clear()

    async def wait_closed(self) -> None:
        if not self._starting:
            return
        await self._closed_event.wait()

    async def wait_stop(self) -> None:
        if not self.is_running():
            return
        await self._stopping_event.wait()

    def wait_closed_sync(self) -> None:
        self._closed_event.wait_sync()
        if self._running_thread is not None:
            self._running_thread.join()

    async def aclose(self) -> None:
        self._stopping_event.set()

    def is_running(self) -> bool:
        return self._starting and not (self._stopping_event.is_set() or self._closed_event.is_set())

    # --- consume runtime event --- #

    async def _clear_connection_status(self) -> None:
        if self._connection_id:
            self._connection_id = None
            await self._clear_running_status()

    async def _sync_connection(self, new: bool) -> None:
        if new or not self._connection_id:
            self._connection_id = uuid()
            self._connection_creating_event.clear()
        try:
            listening_topics = []
            if self._provider_topic_service is not None:
                listening_topics = self._provider_topic_service.listening()
            event = CreateSessionEvent(
                connection_id=self._connection_id,
                # 提供当前正在监听的事件.
                listening_topics=listening_topics,
            ).to_channel_event()
            await self._send_event_to_proxy(event)
            self._connection_creating_event.set()
        except asyncio.CancelledError:
            pass
        except (ConnectionNotAvailable, ConnectionClosedError):
            pass

    async def _consume_proxy_event_loop(self) -> None:
        while not self._stopping_event.is_set():
            try:
                await asyncio.sleep(0.0)
                if not self._connection.is_connected():
                    # 连接未成功, 则清空等待状态. 需要重新创建 connection.
                    await self._clear_connection_status()
                    # 进行下一轮检查.
                    await asyncio.sleep(self._reconnect_interval_seconds)
                    continue

                if not self._connection_id:
                    # 没有创建过 connection, 则尝试创建 connection.
                    await self._sync_connection(new=True)
                    continue

                try:
                    event = await self._connection.recv(timeout=self._reconnect_interval_seconds)
                except asyncio.TimeoutError:
                    continue
                except ConnectionNotAvailable:
                    # 保持重连.
                    continue

                if event is None:
                    break

                if created := SessionCreatedEvent.from_channel_event(event):
                    # proxy 声明创建 Session 成功.
                    if created.connection_id == self._connection_id:
                        self._connection_creating_event.set()
                        # 开始同步 channel metas.
                        sync_meta = SyncChannelMetasEvent(
                            connection_id=self._connection_id,
                        )
                        await self._handle_sync_channel_meta(sync_meta)
                    else:
                        # 继续提醒云端重建 connection.
                        await self._sync_connection(new=False)
                    continue
                elif reconnected := ReconnectSessionEvent.from_channel_event(event):
                    # connection id 不对齐, 重新建立 connection.
                    if reconnected.connection_id != self._connection_id:
                        await self._clear_connection_status()
                        await self._sync_connection(new=len(reconnected.connection_id) > 0)
                    continue

                if event["connection_id"] != self._connection_id:
                    # 丢弃不同 connection 的事件.
                    self.logger.info(
                        "%s channel connection %s mismatch, drop event %s", self._log_prefix, self._connection_id, event
                    )
                    # 频繁要求服务端同步 connection.
                    await self._sync_connection(new=False)
                    continue

                # 所有的事件都异步运行.
                # 如果希望 Channel provider 完全按照阻塞逻辑来执行, 正确的架构设计应该是:
                # 1. 服务端下发 command tokens 流.
                # 2. 本地运行一个 Shell, 消费 command token 生成命令.
                # 3. 本地的 shell 走独立的调度逻辑.
                # 有的是阻塞的, 有的不是阻塞的.
                await self._consume_single_event(event)
            except asyncio.CancelledError:
                self.logger.warning("%s consume runtime event loop is cancelled", self._log_prefix)
                # 中断循环.
                break
            except ConnectionNotAvailable:
                # 继续运行.
                continue
            except ConnectionClosedError:
                self.logger.warning("%s consume runtime event loop is closed", self._log_prefix)
                # 中断循环.
                break
            except Exception as e:
                self.logger.exception("%s consume runtime event loop failed: %s", self._log_prefix, e)
                provider_error = ProviderErrorEvent(
                    connection_id=self._connection_id,
                    errcode=-1,
                    errmsg=f"provider error: {e}",
                )
                await self._send_event_to_proxy(provider_error.to_channel_event())

    async def _consume_single_event(self, event: ChannelEvent) -> None:
        """消费单一事件. 这一层解决 task 生命周期管理."""
        try:
            self.logger.info("%s Received event: %s", self._log_prefix, event)
            handle_task = asyncio.create_task(self._handle_single_event(event))
            wait_close = asyncio.create_task(self._stopping_event.wait())
            done, pending = await asyncio.wait([handle_task, wait_close], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await handle_task
        except Exception as e:
            self.logger.exception("%s Handle event %s task failed: %s", self._log_prefix, event, e)
            raise e

    async def _handle_single_event(self, event: ChannelEvent) -> None:
        """做单个事件的异常管理, 理论上不要抛出任何异常."""
        try:
            event_type = event["event_type"]
            # 如果有自定义的 event, 先处理.
            if event_type in self._proxy_event_handlers:
                handler = self._proxy_event_handlers[event_type]
                # 运行这个 event, 判断是否继续.
                go_on = await handler(self._channel, event)
                if not go_on:
                    return
            # 运行系统默认的 event 处理.
            await self._handle_default_event(event)

        except asyncio.CancelledError:
            pass
        except FatalError as e:
            self.logger.exception("%s fatal error while handling event: %s", self._log_prefix, e)
            self._stopping_event.set()
        except Exception as e:
            self.logger.exception("%s Unhandled error while handling event: %s", self._log_prefix, e)

    async def _handle_default_event(self, event: ChannelEvent) -> None:
        # system event
        try:
            if model := CommandCallEvent.from_channel_event(event):
                # 异步运行 command call.
                _ = self._loop.create_task(self._handle_command_call(model))

            elif model := CommandCancelEvent.from_channel_event(event):
                _ = self._loop.create_task(self._handle_command_cancel(model))

            elif model := SyncChannelMetasEvent.from_channel_event(event):
                await self._handle_sync_channel_meta(model)

            elif model := ClearEvent.from_channel_event(event):
                await self._handel_clear(model)
            elif model := CommandDeltaEvent.from_channel_event(event):
                await self._handle_command_delta_arg(model)
            elif model := ProxyPubTopicEvent.from_channel_event(event):
                await self._handle_proxy_topic(model)
            else:
                self.logger.info("%s unknown event: %s", self._log_prefix, event)
        except ValidationError:
            self.logger.exception("%s received invalid event: %s", self._log_prefix, event)
        except Exception as e:
            self.logger.exception("%s handle default event failed: %s", self._log_prefix, e)
            raise e
        finally:
            self.logger.info("%s handled event: %s", self._log_prefix, event)

    async def _handle_proxy_topic(self, event: ProxyPubTopicEvent) -> None:
        try:
            self._root_runtime.pub_topic(event.topic)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s receive proxy topic failed: %s", self._log_prefix, e)
            raise e

    async def _handel_clear(self, event: ClearEvent):
        """执行 clear 逻辑."""
        channel_name = event.chan
        try:
            node = self._root_runtime.fetch_sub_runtime(channel_name)
            if not node:
                return
            # 执行 clear 命令.
            await node.clear()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s Clear channel failed: %s", self._log_prefix, e)
            raise e

    async def _send_event_to_proxy(self, event: ChannelEvent, connection_id: str = "") -> None:
        """做好事件发送的异常管理."""
        try:
            if not self._connection.is_connected():
                return
            event["connection_id"] = connection_id or self._connection_id or ""
            await self._connection.send(event)
        except asyncio.CancelledError:
            raise
        except ConnectionNotAvailable:
            await self._clear_connection_status()

        except ConnectionClosedError:
            self.logger.exception("%s Connection closed while sending event %s", self._log_prefix, event)
            # 关闭整个 channel provider.
            self._stopping_event.set()
        except Exception as e:
            self.logger.exception("%s Send event %s failed %s", self._log_prefix, event, e)
            # 不抛出异常.

    async def _handle_sync_channel_meta(self, event: SyncChannelMetasEvent) -> None:
        try:
            try:
                await self._root_runtime.tree.refresh_all()
            except Exception as e:
                self.logger.exception("%s run meta event %s failed: %s", self._log_prefix, event, e)

            metas = self._root_runtime.tree.metas()
            response = ChannelMetaUpdateEvent(
                connection_id=event.connection_id,
                metas=metas.copy(),
                root_chan=self._channel.name(),
            )
            await self._send_event_to_proxy(response.to_channel_event())
        except asyncio.CancelledError:
            pass

    async def _handle_command_cancel(self, event: CommandCancelEvent) -> None:
        cid = event.command_id
        task = self._running_command_tasks.get(cid, None)
        if task is not None:
            self.logger.info("cancel task %s by event %s", task, event)
            # 设置 task 取消.
            task.cancel()

    async def _handle_command_delta_arg(self, event: CommandDeltaEvent) -> None:
        cid = event.command_id
        if cid not in self._running_command_delta_stream:
            return
        sender, receiver = self._running_command_delta_stream[cid]
        if event.command_token:
            command_token = CommandToken(**event.command_token)
            sender.append(command_token)
        elif event.chunk:
            sender.append(event.chunk)
        else:
            sender.commit()

    async def _handle_command_call(self, call_event: CommandCallEvent) -> None:
        """执行一个命令运行的逻辑."""
        # 先取消 lifecycle 的命令.
        node = self._root_runtime.fetch_sub_runtime(call_event.chan)
        if node is None:
            response = call_event.not_available()
            await self._send_event_to_proxy(response.to_channel_event())
            return

        # 获取真实的 command 对象.
        command = node.get_command(call_event.name)
        if command is None or not command.is_available():
            response = call_event.not_available()
            await self._send_event_to_proxy(response.to_channel_event())
            return

        task = BaseCommandTask(
            chan=call_event.chan,
            meta=command.meta(),
            func=command.__call__,
            tokens=call_event.tokens,
            args=call_event.args,
            kwargs=call_event.kwargs,
            cid=call_event.command_id,
            context=call_event.context,
            call_id=call_event.call_id,
        )
        # 真正执行这个 task.
        try:
            # 多余的, 没什么用.
            task.set_state(CommandTaskState.executing.value)
            task.add_done_callback(self._remove_running_task)
            await self._add_running_task(task)
            self._root_runtime.push_task(task)
            await task
        except asyncio.CancelledError:
            task.cancel("cancelled")
            pass
        except Exception as e:
            self.logger.exception("Execute command failed")
            task.fail(e)
        finally:
            if not task.done():
                task.cancel()
            result = task.task_result().serializable() if task.success() else None
            response = call_event.done(result, task.errcode, task.errmsg)
            await self._send_event_to_proxy(response.to_channel_event())

    async def _add_running_task(self, task: CommandTask) -> None:
        await self._running_command_tasks_lock.acquire()
        try:
            if task.meta.delta_arg is not None and task.meta.delta_arg not in task.kwargs:
                sender, receiver = create_sender_and_receiver()
                task.kwargs[task.meta.delta_arg] = receiver
                self._running_command_delta_stream[task.cid] = (sender, receiver)
            self._running_command_tasks[task.cid] = task
        finally:
            self._running_command_tasks_lock.release()

    def _remove_running_task(self, task: CommandTask) -> None:
        cid = task.cid
        if cid in self._running_command_tasks:
            del self._running_command_tasks[cid]

    def run_in_thread(self, channel: Channel) -> threading.Thread:
        if self._running_thread is not None:
            return self._running_thread
        self._running_thread = super().run_in_thread(channel)
        return self._running_thread

    async def arun_until_closed(self, channel: Channel) -> None:
        if self._running_task is not None:
            await self._running_task
            return
        self._running_task = asyncio.create_task(self._arun_until_closed(channel))
        await self._running_task

    async def _arun_until_closed(self, channel: Channel) -> None:
        async with self.arun(channel):
            await self.wait_stop()

    def close(self) -> None:
        self._stopping_event.set()
