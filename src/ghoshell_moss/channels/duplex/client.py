from typing import Dict, Any, Optional, Callable, Coroutine
from typing_extensions import Self

from ghoshell_moss import ChannelClient
from ghoshell_moss.concepts.channel import Channel, ChannelMeta, Builder, R
from ghoshell_moss.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.concepts.command import Command, CommandTask, BaseCommandTask, CommandMeta, CommandWrapper
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from .protocol import *
from .connection import *
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer
import logging
import asyncio

__all__ = ['DuplexChannelClient', 'DuplexChannelStub', 'DuplexChannelProxy']


class DuplexChannelContext:

    def __init__(
            self,
            *,
            session_id: str,
            container: IoCContainer,
            connection: Connection,
            root_local_channel: Channel,
            command_peek_interval: float = 1.0,
    ):
        self.root_name = root_local_channel.name()
        """根节点的名字. 这个名字可能和远端的 channel 根节点不一样. """
        self.remote_root_name = ""
        """远端的 root channel 名字"""
        self._command_peek_interval = command_peek_interval if command_peek_interval > 0 else 1.0

        self.session_id = session_id
        self.container = container or Container(name="duplex channel context container")
        self.connection = connection
        """双工连接本身."""

        self.root_local_channel = root_local_channel
        """根节点的本地 channel. """

        self.meta_map: Dict[str, ChannelMeta] = {}
        """所有远端上传的 metas. """

        self.started = False
        self.available = True
        self.stop_event = ThreadSafeEvent()
        """全局的 stop event, 会中断所有的子节点"""

        # runtime
        self._pending_server_command_calls: Dict[str, CommandTask] = {}

        self.server_event_queue_map: Dict[str, asyncio.Queue[ChannelEvent | None]] = {}
        """按 channel 名称进行分发的队列."""

        self._main_task: Optional[asyncio.Task] = None

        self._logger: logging.Logger | None = None
        """logger 的缓存."""

    def get_meta(self, name: str) -> Optional[ChannelMeta]:
        if not name:
            name = self.remote_root_name
        return self.meta_map.get(name, None)

    async def send_event_to_server(self, event: ChannelEvent) -> bool:
        if self.stop_event.is_set() or not self.connection.is_available():
            self.logger.warning("Channel %s Connection is stopped or not available" % self.root_name)
        try:
            await self.connection.send(event)
            return True
        except ConnectionClosedError:
            self.logger.warning("Channel %s Connection is closed" % self.root_name)
            self.available = False
            return False
        except Exception as e:
            self.logger.exception(e)
            return False

    def get_server_event_queue(self, name: str) -> asyncio.Queue[ChannelEvent | None]:
        """
        :param name: 这里的 name 是 channel 在远端的原名称.
        """
        if name == self.remote_root_name:
            # 用 "" 表示根节点.
            name = ""
        if name not in self.server_event_queue_map:
            self.server_event_queue_map[name] = asyncio.Queue()
        return self.server_event_queue_map[name]

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        # 完成初始化.
        await self._bootstrap()
        # 创建主循环.
        self._main_task = asyncio.create_task(self._main())

    async def close(self) -> None:
        if self.stop_event.is_set():
            return
        # 通知关闭.
        self.stop_event.set()
        # 尝试通知所有的子节点关闭.
        for queue in self.server_event_queue_map.values():
            queue.put_nowait(None)
        # 等待主任务结束.
        if self._main_task:
            await self._main_task

    def is_available(self, name: str) -> bool:
        """判断一个 channel 是否可以运行. """
        connection_is_available = self.is_running() and self.connection.is_available()
        if not connection_is_available:
            return False
        # 再判断 meta 也是 available 的.
        meta = self.get_meta(name)
        return meta and meta.available

    def is_running(self) -> bool:
        """判断 ctx 是否在运行. """
        return self.started and not self.stop_event.is_set() and not self.connection.is_closed()

    async def _bootstrap(self):
        await asyncio.to_thread(self.container.bootstrap)
        await self.connection.start()
        sync_event = SyncChannelMetasEvent(session_id=self.session_id).to_channel_event()
        await self.connection.send(sync_event)
        received = await self.connection.recv(timeout=10)
        update_metas = ChannelMetaUpdateEvent.from_channel_event(received)
        if update_metas is None:
            raise ConnectionClosedError(f'Channel {self.root_local_channel.name()} initialize failed: no meta update')
        await self.update_meta(update_metas)

    async def _main(self):
        try:
            # 异常管理放在外侧, 方便阅读代码.
            receiving_loop = asyncio.create_task(self._main_receiving_loop())
            peek_loop = asyncio.create_task(self._command_peek_loop())
            is_stopped = asyncio.create_task(self.stop_event.wait())
            done, pending = await asyncio.wait(
                [receiving_loop, is_stopped, peek_loop],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await receiving_loop
        except asyncio.CancelledError:
            raise
        except ConnectionClosedError:
            self.logger.info(f"Channel {self.root_name} Connection closed")
        except Exception as e:
            self.logger.exception(e)
            raise
        finally:
            self.stop_event.set()
            for queue in self.server_event_queue_map.values():
                queue.put_nowait(None)
            for task in self._pending_server_command_calls.values():
                task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"Channel {self.root_name} connection closed"))

    async def _main_receiving_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                # 等待一个事件.
                event = await self.connection.recv()
                # 默认的毒丸逻辑. 防止死锁.
                if event is None:
                    break

                # sync metas 事件的标准处理.
                if update_meta := ChannelMetaUpdateEvent.from_channel_event(event):
                    await self.update_meta(update_meta)
                    continue
                # server errors 的标准处理.
                elif server_error := ChannelMetaUpdateEvent.from_channel_event(event):
                    self.logger.error(f'Channel {self.root_name} error: {server_error}')
                    continue
                elif command_done := CommandDoneEvent.from_channel_event(event):
                    await self._handle_command_done(command_done)

                # 判断回调分发给哪个具体的 channel.
                if "chan" in event['data']:
                    chan = event['data']['chan']
                    # 检查是否是已经注册的 channel.
                    if chan not in self.meta_map:
                        self.logger.error(f'Channel {self.root_name} error: {chan} not found')
                        continue

                    queue = self.get_server_event_queue(chan)
                    # 分发给指定 channel.
                    await queue.put(event)
                else:
                    # 拿到的 channel 不可理解.
                    self.logger.error(f'Channel {self.root_name} receive unknown event : {event}')
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError:
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop_event.set()

    async def update_meta(self, event: ChannelMetaUpdateEvent) -> None:
        """更新 metas 信息. """
        self.remote_root_name = event.root_chan
        # 更新 meta map.
        meta_map = {}
        for meta in event.metas:
            meta_map[meta.name] = meta.model_copy()
            if meta.name not in self.server_event_queue_map:
                # 提前补充好 channel 分发用的 queue.
                self.server_event_queue_map[meta.name] = asyncio.Queue()

        for meta in self.meta_map.values():
            if meta.name not in meta_map:
                meta.available = False
        # 直接变更当前的 meta map. 则一些原本存在的 channel, 也可能临时不存在了.
        self.meta_map = meta_map

    async def _command_peek_loop(self):
        try:
            while self.is_running():
                if len(self._pending_server_command_calls) > 0:
                    tasks = self._pending_server_command_calls.copy()
                    # 不能联通的情况下, 主动清空所有任务.
                    if not self.connection.is_available():
                        for task in tasks.values():
                            task.fail(CommandErrorCode.NOT_AVAILABLE.error("Channel connection not available"))
                        continue
                    for task in tasks.values():
                        peek_event = CommandPeekEvent(
                            chan=task.meta.chan,
                            command_id=task.cid,
                        )
                        await self.send_event_to_server(peek_event.to_channel_event())
                await asyncio.sleep(self._command_peek_interval)
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError:
            self.stop_event.set()
        except Exception as e:
            self.logger.exception(e)

    async def execute_command_call(self, meta: CommandMeta, event: CommandCallEvent) -> CommandTask:
        """与远程 server 进行通讯, 发送一个 command call, 并且保障有回调. """
        cid = event.command_id
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
            # 清空已经存在的 cid 错误?
            if cid in self._pending_server_command_calls:
                t = self._pending_server_command_calls.pop(cid)
                t.cancel()
                self.logger.error(f"Command Task {cid} duplicated call")
            # 添加新的 task.
            self._pending_server_command_calls[cid] = wait_result_task

            # 等待异步返回结果.
            success = await self.send_event_to_server(event.to_channel_event())
            if not success:
                wait_result_task.fail(CommandErrorCode.FAILED.error("Failed to send command to server"))
                return wait_result_task

            task_done = asyncio.create_task(wait_result_task.wait(throw=False))
            is_stopped = asyncio.create_task(self.stop_event.wait())
            _, pending = await asyncio.wait(
                [task_done, is_stopped],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            await task_done
            return wait_result_task

        except ConnectionClosedError:
            wait_result_task.fail(CommandErrorCode.FAILED.error("remote connection closed"))
            return wait_result_task

        except asyncio.CancelledError:
            if not wait_result_task.done():
                wait_result_task.cancel()
                # 发送取消事件, 通知给下游.
                if self.is_available(event.chan):
                    await self.send_event_to_server(event.cancel().to_channel_event())
            return wait_result_task
        except Exception as e:
            if not wait_result_task.done():
                wait_result_task.fail(e)
                if self.is_running():
                    await self.send_event_to_server(event.cancel().to_channel_event())

            self.logger.exception(e)
            return wait_result_task

        finally:
            if cid in self._pending_server_command_calls:
                self._pending_server_command_calls.pop(cid)

    async def _handle_command_done(self, event: CommandDoneEvent) -> None:
        try:
            command_id = event.command_id
            if command_id in self._pending_server_command_calls:
                task = self._pending_server_command_calls[command_id]
                if event.errcode == 0:
                    task.resolve(event.data)
                else:
                    error = CommandError(event.errcode, event.errmsg)
                    task.fail(error)
        except Exception as e:
            self.logger.exception(e)


class DuplexChannelStub(Channel):
    """被 channel meta 动态生成的子 channel. """

    def __init__(
            self,
            *,
            name: str,  # 本地的名称.
            ctx: DuplexChannelContext,
            server_chan_name: str = "",  # 远端真实的名称.
            local_channel: Channel = None,
    ) -> None:
        self._name = name
        self._server_chan_name = server_chan_name or name
        self._ctx = ctx
        self._local_channel = local_channel or PyChannel(name=name)
        # 运行时缓存.
        self._client: ChannelClient | None = None
        self._started = False
        self._children_stubs: Dict[str, DuplexChannelStub] = {}

    def name(self) -> str:
        return self._name

    def _get_server_channel_meta(self) -> Optional[ChannelMeta]:
        # 获取自己在 server 端的 channel meta.
        return self._ctx.meta_map.get(self._server_chan_name)

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._client

    def include_channels(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        return self._local_channel.include_channels(*children, parent=parent)

    def new_child(self, name: str) -> Self:
        return self._local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        local_children = self._local_channel.children()
        meta = self._get_server_channel_meta()
        if meta is None:
            return local_children

        # 遍历自己的 meta children.
        children_stubs = {}
        for child_meta_name in meta.children:
            if child_meta_name in self._children_stubs:
                # 复制到新字典.
                children_stubs[child_meta_name] = self._children_stubs[child_meta_name]
                continue
            local_child_channel = self._ctx.root_local_channel.get_channel(child_meta_name)
            stub = DuplexChannelStub(
                name=child_meta_name,
                ctx=self._ctx,
                local_channel=local_child_channel,
                server_chan_name=child_meta_name,
            )
            children_stubs[child_meta_name] = stub
        # 每次都更新当前的 children stubs.
        self._children_stubs.clear()
        self._children_stubs = children_stubs
        result: Dict[str, Channel] = children_stubs.copy()
        # 补全本地的 channel.
        for local_child_name, local_child_channel in local_children.items():
            if local_child_name not in children_stubs:
                result[local_child_name] = local_child_channel
        return result

    def is_running(self) -> bool:
        return self._started and self._client is not None and self._ctx.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self._name} has already been started.')
        if not self._ctx.is_running():
            raise RuntimeError(f'Duplex Channel {self._name} Context is not running')

        running_client = DuplexChannelClient(
            name=self._name,
            server_chan_name=self._server_chan_name,
            ctx=self._ctx,
            local_channel=self._local_channel,
            container=container,
        )
        self._client = running_client
        return running_client

    @property
    def build(self) -> Builder:
        return self._local_channel.build


class DuplexChannelClient(ChannelClient):
    """
    实现一个极简的 Duplex Channel, 它核心是可以通过 ChannelMeta 被动态构建出来.
    """

    def __init__(
            self,
            *,
            name: str,
            server_chan_name: str,
            ctx: DuplexChannelContext,
            local_channel: Channel,
            container: Optional[IoCContainer] = None,
            channel_id: Optional[str] = None,
    ) -> None:
        """
        :param name: channel local name
        :param server_chan_name: the origin channel name from the remote server
        :param ctx: shared ctx of all the channels.
        :param local_channel: the local channel object, provide local commands and functions.
        :param container: the channel container object.
        :param channel_id: the channel id
        """
        self._name = name
        self._server_chan_name = server_chan_name
        self._ctx = ctx
        self.container = container or ctx.container
        self.id = channel_id or uuid()
        self._local_channel = local_channel
        # meta 的讯息.
        self._cached_meta: Optional[ChannelMeta] = None

        # 运行时参数
        self._started = False
        self._logger: logging.Logger | None = None

        self._self_close_event = ThreadSafeEvent()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._main_loop_done_event = ThreadSafeEvent()

    def is_running(self) -> bool:
        return self._started and self._ctx.is_running() and not self._self_close_event.is_set()

    @property
    def logger(self) -> logging.Logger:
        return self._ctx.logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel client {self._name} is not running')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        if self._cached_meta is not None and not no_cache:
            return self._cached_meta.model_copy()
        self._cached_meta = self._build_meta()
        return self._cached_meta.model_copy()

    def _build_meta(self) -> ChannelMeta:
        self._check_running()
        meta = self._ctx.get_meta(self._server_chan_name)
        if meta is None:
            return ChannelMeta(
                name=self._name,
                channel_id=self.id,
                available=False,
            )
        # 避免污染.
        meta = meta.model_copy()
        # 从 server meta 中准备 commands 的原型.
        commands = {}
        for command_meta in meta.commands:
            # 命令替换名称为自身的名称. 给调用方看.
            command_meta = command_meta.model_copy(update={"chan": self._name})
            commands[command_meta.name] = command_meta

        # 如果有本地注册的函数, 用它们取代 server 同步过来的.
        if self._local_channel is not None:
            local_meta = self._local_channel.client.meta()
            for command_meta in local_meta.commands:
                commands[command_meta.name] = command_meta
        meta.commands = list(commands.values())
        # 修改别名.
        meta.name = self._name
        return meta

    def is_available(self) -> bool:
        return self.is_running() and self._ctx.is_available(self._server_chan_name)

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        # 先获取本地的命令.
        result = self._local_channel.client.commands(available_only=available_only)
        meta = self._ctx.get_meta(self._server_chan_name)
        if meta is None:
            return result
        # 再封装远端的命令.
        for command_meta in meta.commands:
            if command_meta.name not in result and not available_only or command_meta.available:
                func = self._get_server_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_server_channel_name(self) -> str:
        return self._server_chan_name or self._ctx.remote_root_name

    def _get_server_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        session_id = self._ctx.session_id

        # 回调服务端的函数.
        async def _call_server_as_func(*args, **kwargs):
            if not self.is_running():
                # 告知上游运行失败.
                raise CommandError(CommandErrorCode.NOT_AVAILABLE, f'Channel {self._name} not available')

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
                # channel 名称使用 server 侧的名称
                chan=self._get_server_channel_name(),
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
            raise LookupError(f'Channel {self._name} can find command {task.meta.name}')
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            if self._local_channel is not None:
                await self._local_channel.client.policy_run()

            event = RunPolicyEvent(
                session_id=self._ctx.session_id,
                chan=self._server_chan_name,
            )
            await self._ctx.send_event_to_server(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def policy_pause(self) -> None:
        self._check_running()
        try:
            if self._local_channel is not None:
                await self._local_channel.client.policy_pause()

            event = PausePolicyEvent(
                session_id=self._ctx.session_id,
                chan=self._server_chan_name,
            )
            await self._ctx.send_event_to_server(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def clear(self) -> None:
        self._check_running()
        try:
            if self._local_channel is not None:
                await self._local_channel.client.policy_pause()

            event = ClearCallEvent(
                session_id=self._ctx.session_id,
                chan=self._server_chan_name,
            )
            await self._ctx.send_event_to_server(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _consume_server_event_loop(self):
        try:
            while self.is_running():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            self._self_close_event.set()
        finally:
            self.logger.info("channel %s consume_server_event_loop stopped", self._name)

    async def _main_loop(self):
        try:
            consume_loop_task = asyncio.create_task(self._consume_server_event_loop())
            await consume_loop_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            if self._local_channel is not None and self._local_channel.is_running():
                await self._local_channel.client.close()
            await asyncio.to_thread(self.container.shutdown)
            self._main_loop_done_event.set()

    async def _consume_server_event(self):
        try:
            if self._ctx.connection.is_closed():
                self._self_close_event.set()
                return

            queue = self._ctx.get_server_event_queue(self._server_chan_name)

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                return
            if item is None:
                self._self_close_event.set()
                return

            if model := RunPolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s run policy is done from event %s", self._name, model)
            elif model := PausePolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s pause policy is done from event %s", self._name, model)
            elif model := ClearDoneEvent.from_channel_event(item):
                self.logger.info("channel %s clear is done from event %s", self._name, model)
            else:
                self.logger.info('unknown server event %s', item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._ctx.root_name == self._name:
            # 启动 ctx.
            await self._ctx.start()

        await asyncio.to_thread(self.container.bootstrap)
        if self._local_channel is not None:
            await self._local_channel.bootstrap(self.container).start()
        self._main_loop_task = asyncio.create_task(self._main_loop())

    def is_root(self) -> bool:
        return self._name == self._ctx.root_name

    async def close(self) -> None:
        if self._self_close_event.is_set():
            return
        self._self_close_event.set()
        # 关闭结束 ctx.
        if self.is_root():
            await self._ctx.close()
        try:
            if self._main_loop_task:
                self._main_loop_task.cancel()
                await self._main_loop_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)


class DuplexChannelProxy(Channel):

    def __init__(
            self,
            *,
            name: str,
            description: str = "",
            block: bool = True,
            to_server_connection: Connection,
    ):
        self._name = name
        self._server_connection = to_server_connection
        self._local_channel = PyChannel(name=name, description=description, block=block)
        self._client: Optional[DuplexChannelClient] = None

        self._ctx: DuplexChannelContext | None = None
        """运行的时候才会生成 Channel Context"""

        self._children_stubs: Dict[str, DuplexChannelStub] = {}

    def name(self) -> str:
        return self._name

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._client

    def include_channels(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        # 新添加的 channel 都放到 local channel 里.
        self._local_channel.include_channels(*children, parent=parent)
        return self

    def new_child(self, name: str) -> Self:
        return self._local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        if self._ctx is None:
            return self._local_channel.children()
        # 每次动态检查生成 children channels.
        local_children = self._local_channel.children()
        children_stubs = {}
        for name, meta in self._ctx.meta_map.items():
            if name in self._children_stubs:
                # 已经生成过了.
                children_stubs[name] = self._children_stubs[name]
                continue
            local_child = local_children.get(name, None)
            stub = DuplexChannelStub(
                name=name,
                ctx=self._ctx,
                server_chan_name=name,
                local_channel=local_child,
            )
            children_stubs[name] = stub
        self._children_stubs = children_stubs
        # 生成一个新的组合.
        result: Dict[str, Channel] = self._children_stubs.copy()
        # 补齐有 local channel, 但没有 channel stub 的.
        for name, local_child_channel in local_children.items():
            if name not in result:
                result[name] = local_child_channel
        return result

    def is_running(self) -> bool:
        return self._client is not None and self._client.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self} has already been started.')

        self._ctx = DuplexChannelContext(
            session_id=uuid(),
            container=container,
            connection=self._server_connection,
            root_local_channel=self._local_channel,
        )

        client = DuplexChannelClient(
            name=self._name,
            server_chan_name="",
            ctx=self._ctx,
            local_channel=self._local_channel,
            container=container,
        )
        self._client = client
        return client

    @property
    def build(self) -> Builder:
        return self._local_channel.build
