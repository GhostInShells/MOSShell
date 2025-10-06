from typing import TypedDict, Dict, Any, ClassVar, Optional, List, Callable, Coroutine
from typing_extensions import Self
from abc import ABC

from ghoshell_moss import ChannelClient
from ghoshell_moss.concepts.channel import Channel, ChannelMeta, Builder, R
from ghoshell_moss.concepts.errors import CommandError
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
    ):
        self.root_name = root_local_channel.name()
        self.origin_root_name = ""
        self.session_id = session_id
        self.container = container or Container(name="duplex channel context container")
        self.connection = connection
        self.root_local_channel = root_local_channel
        self.meta_map: Dict[str, ChannelMeta] = {}
        self.runtime_channels: Dict[str, Channel] = {}
        self.started = False
        self.stop_event = ThreadSafeEvent()
        self.client_event_queue: asyncio.Queue[ChannelEvent | None] = asyncio.Queue()
        self.server_event_queue_map: Dict[str, asyncio.Queue[ChannelEvent | None]] = {}
        self._main_task: Optional[asyncio.Task] = None
        self._logger: logging.Logger | None = None

    async def send_event_to_server(self, event: ChannelEvent):
        await self.client_event_queue.put(event)

    def get_server_event_queue(self, name: str) -> asyncio.Queue[ChannelEvent | None]:
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
        await self._bootstrap()
        self._main_task = asyncio.create_task(self._main())

    async def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        for queue in self.server_event_queue_map.values():
            queue.put_nowait(None)
        if self._main_task:
            await self._main_task

    def is_available(self, name: str) -> bool:
        return self.is_running() and name in self.meta_map and self.meta_map[name].available

    def is_running(self) -> bool:
        return self.started and not self.stop_event.is_set()

    def get_children(self, name: str) -> Dict[str, Channel]:
        result = {}
        if name not in self.meta_map:
            local_channel = self.root_local_channel.get_channel(name)
            if local_channel is not None:
                return local_channel.children()
            # 返回空结果.
            return result
        else:
            meta = self.meta_map[name]
            result = {}
            for child_name in meta.children:
                child_channel = self.get_channel(child_name)
                result[child_name] = child_channel
            return result

    def get_channel(self, name: str) -> Optional[Channel]:
        if name == self.root_name:
            name = self.origin_root_name

        if name not in self.meta_map:
            return None
        elif name in self.runtime_channels:
            return self.runtime_channels[name]

        server_meta = self.meta_map.get(name)
        runtime_channel = DuplexChannelStub(
            name=name,
            server_meta_name=server_meta.name,
            ctx=self,
        )
        self.runtime_channels[name] = runtime_channel

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
            gathered = asyncio.gather(self._main_receiving_loop(), self._main_sending_loop())
            wait_task = asyncio.create_task(self.stop_event.wait())
            done, pending = await asyncio.wait([gathered, wait_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await gathered
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError:
            self.logger.info(f"Channel {self.root_name} Connection closed")
            self.stop_event.set()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop_event.set()

    async def send_event(self, channel_event: ChannelEvent) -> None:
        try:
            await self.client_event_queue.put(channel_event)
        except asyncio.CancelledError:
            pass

    async def update_meta(self, event: ChannelMetaUpdateEvent) -> None:
        self.origin_root_name = event.root_chan
        # 更新 meta map.
        meta_map = {}
        for meta in event.metas:
            meta_map[meta.name] = meta.model_copy()
            if meta.name not in self.server_event_queue_map:
                self.server_event_queue_map[meta.name] = asyncio.Queue()

        for meta in self.meta_map.values():
            if meta.name not in meta_map:
                meta.available = False
        self.meta_map.update(meta_map)
        root_meta = self.meta_map.get(self.origin_root_name)
        # 更新剩余的 meta.
        self._update_channel_with_meta(root_meta, self.root_name)

    def _update_channel_with_meta(self, meta: ChannelMeta, alias: str | None = None):
        if meta.name not in self.runtime_channels:
            alias = alias or meta.name
            runtime_channel = DuplexChannelStub(
                name=alias,
                server_meta_name=meta.name,
                ctx=self,
            )
            self.runtime_channels[meta.name] = runtime_channel
        for child_name in meta.children:
            child_meta = self.meta_map.get(child_name)
            self._update_channel_with_meta(child_meta)

    async def _main_receiving_loop(self) -> None:
        while not self.stop_event.is_set():
            event = await self.connection.recv()
            if event is None:
                self.stop_event.set()
                break

            if update_meta := ChannelMetaUpdateEvent.from_channel_event(event):
                await self.update_meta(update_meta)
                continue
            elif server_error := ChannelMetaUpdateEvent.from_channel_event(event):
                self.logger.error(f'Channel {self.root_name} error: {server_error}')
                continue

            if "chan" in event['data']:
                chan = event['data']['chan']
                # 检查是否是已经注册的 channel.
                if chan not in self.meta_map:
                    self.logger.error(f'Channel {self.root_name} error: {chan} not found')
                    continue

                if chan not in self.server_event_queue_map:
                    self.server_event_queue_map[chan] = asyncio.Queue()
                queue = self.server_event_queue_map[chan]
                await queue.put(event)
            else:
                self.logger.error(f'Channel {self.root_name} receive unknown event : {event}')

    async def _main_sending_loop(self) -> None:
        while not self.stop_event.is_set():
            item = await self.client_event_queue.get()
            await self.connection.send(item)


class DuplexChannelStub(Channel):

    def __init__(
            self,
            *,
            name: str,
            ctx: DuplexChannelContext,
            server_meta_name: str = "",
    ) -> None:
        self._name = name
        self._server_meta_name = server_meta_name
        self._ctx = ctx
        self._client: ChannelClient | None = None
        self._local_channel = ctx.root_local_channel.get_channel(name) or PyChannel(name=name)
        self._started = False

    def name(self) -> str:
        return self._name

    def _get_server_meta(self) -> Optional[ChannelMeta]:
        server_meta_name = self._server_meta_name or self._ctx.origin_root_name
        return self._ctx.meta_map.get(server_meta_name)

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._client

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        return self._local_channel.with_children(*children, parent=parent)

    def new_child(self, name: str) -> Self:
        return self._local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        result = self._local_channel.children()
        meta = self._get_server_meta()
        if meta is None:
            return result
        for child_meta_name in meta.children:
            channel = self._ctx.get_channel(child_meta_name)
            result[channel.name()] = channel
        return result

    def is_running(self) -> bool:
        if self._ctx.stop_event.is_set():
            return False
        meta = self._ctx.meta_map.get(self._server_meta_name)
        return self._started and meta is not None

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self} has already been started.')

        running_client = DuplexChannelClient(
            name=self._name,
            is_root=False,
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
            is_root: bool,
            ctx: DuplexChannelContext,
            local_channel: Channel,
            container: Optional[IoCContainer] = None,
    ) -> None:
        """
        """
        self.container = container or ctx.container
        self.id = uuid()
        self._is_root = is_root
        self._name = name
        self._ctx = ctx
        self._local_channel = local_channel

        # meta 的讯息.
        self._cached_meta: Optional[ChannelMeta] = None

        # 运行时参数
        self._started = False
        self._logger: logging.Logger | None = None
        self._remote_stop_event = ctx.stop_event

        self._self_close_event = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._main_loop_done_event = ThreadSafeEvent()
        # runtime
        self._pending_server_command_calls: Dict[str, CommandTask] = {}

    @property
    def server_meta_name(self) -> str:
        return self._name if not self._is_root else self._ctx.origin_root_name

    def is_running(self) -> bool:
        return self._started and self._ctx.is_running()

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
        meta_name = self.server_meta_name
        meta = self._ctx.meta_map.get(meta_name)
        if meta is None:
            return ChannelMeta(
                name=self._name,
                channel_id=self.id,
                available=False,
            )
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
        return self.is_running() and self._ctx.is_available(self.server_meta_name)

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
        if self._local_channel is not None:
            command = self._local_channel.client.get_command(name)
            if command is not None:
                self.logger.info(f"Channel {self._name} found command `{meta.name}` from local")
                return command.__call__
        session_id = self._ctx.session_id

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            task: CommandTask | None = None
            try:
                task = CommandTask.get_from_context()
            except LookupError:
                pass
            cid = task.cid if task else uuid()
            event = CommandCallEvent(
                session_id=session_id,
                name=name,
                # channel 名称使用 server 侧的名称
                chan=self.server_meta_name,
                command_id=cid,
                args=list(args),
                kwargs=dict(kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
            )
            wait_result_task = BaseCommandTask(
                meta=meta,
                func=None,
                cid=event.command_id,
                tokens=event.tokens,
                args=event.args,
                kwargs=event.kwargs,
                context=event.context,
            )
            self._pending_server_command_calls[cid] = wait_result_task
            try:
                await self._ctx.send_event_to_server(event.to_channel_event())
                # 等待异步返回结果.
                await wait_result_task.wait()
            except asyncio.CancelledError:
                if not wait_result_task.done():
                    wait_result_task.cancel()
                    await self._ctx.send_event_to_server(event.cancel().to_channel_event())
                raise
            except Exception as e:
                if not wait_result_task.done():
                    wait_result_task.fail(e)
                    await self._ctx.send_event_to_server(event.cancel().to_channel_event())
                self.logger.exception(e)
                raise

            finally:
                if cid in self._pending_server_command_calls:
                    _ = self._pending_server_command_calls.pop(cid)

            # 正常返回.
            wait_result_task.raise_exception()
            return wait_result_task.result()

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
            raise LookupError(f'Channel {self._name} can find command {task.meta.name}')
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            if self._local_channel is not None:
                await self._local_channel.client.policy_run()

            event = RunPolicyEvent(
                session_id=self._ctx.session_id,
                chan=self.server_meta_name,
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
                chan=self.server_meta_name,
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
                chan=self.server_meta_name,
            )
            await self._ctx.send_event_to_server(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _consume_server_event_loop(self):
        try:
            while not self._remote_stop_event.is_set() and not self._self_close_event.is_set():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            self._self_close_event.set()

    async def _command_peek_loop(self):
        while not self._remote_stop_event.is_set() and not self._self_close_event.is_set():
            if len(self._pending_server_command_calls) > 0:
                tasks = self._pending_server_command_calls.copy()
                for task in tasks.values():
                    peek_event = CommandPeekEvent(
                        chan=task.meta.chan,
                        command_id=task.cid,
                    )
                    await self._ctx.send_event_to_server(peek_event.to_channel_event())
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
            queue = self._ctx.get_server_event_queue(self.server_meta_name)

            item = await queue.get()
            if item is None:
                self._self_close_event.set()
                return

            if model := CommandDoneEvent.from_channel_event(item):
                await self._handle_command_done(model)
            elif model := RunPolicyDoneEvent.from_channel_event(item):
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
        if self._is_root:
            # 启动 ctx.
            await self._ctx.start()

        await asyncio.to_thread(self.container.bootstrap)
        if self._local_channel is not None:
            await self._local_channel.bootstrap(self.container).start()
        self._main_loop_task = asyncio.create_task(self._main_loop())

    async def close(self) -> None:
        if self._self_close_event.is_set():
            await self._main_loop_done_event.wait()
        self._self_close_event.set()
        self._main_loop_task.cancel()
        try:
            await self._main_loop_task
        except asyncio.CancelledError:
            pass

        if self._local_channel is not None and self._local_channel.is_running():
            await self._local_channel.client.close()
        await asyncio.to_thread(self.container.shutdown)
        # 关闭结束 ctx.
        if self._is_root:
            await self._ctx.close()


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
        if self._ctx is None:
            return self._local_channel.children()
        return self._ctx.get_children(self._ctx.origin_root_name)

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
            is_root=True,
            ctx=self._ctx,
            local_channel=self._local_channel,
            container=container,
        )
        self._client = client
        return client

    @property
    def build(self) -> Builder:
        return self._local_channel.build
