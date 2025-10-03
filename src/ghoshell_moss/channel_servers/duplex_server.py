import asyncio
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
    """开始运行 policy"""
    event_type: ClassVar[str] = "moss.channel.client.policy.run"
    chan: str = Field(description="channel name")


class PausePolicyEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.client.policy.pause"
    chan: str = Field(description="channel name")


class ClearCallEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.client.clear.call"
    chan: str = Field(description="channel name")


class CommandCallEvent(ChannelEventModel):
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
        )

    def done(self, result: Any, errcode: int, errmsg: str) -> "CommandDoneEvent":
        return CommandDoneEvent(
            command_id=self.command_id,
            errcode=errcode,
            errmsg=errmsg,
            data=result,
        )

    def not_found(self, msg: str = "") -> "CommandDoneEvent":
        return CommandDoneEvent(
            command_id=self.command_id,
            errcode=CommandErrorCode.NOT_FOUND.value,
            errmsg=msg or f"command `{self.chan}:{self.name}` not found",
            data=None,
        )


class CommandCancelEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.client.command.cancel"
    command_id: str = Field(description="command id")


class SyncChannelMetasEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.meta.sync"
    channels: List[str] = Field(default_factory=list, description="channel names to sync")


# --- server event --- #

class CommandDoneEvent(ChannelEventModel):
    event_type: ClassVar[str] = "moss.channel.server.command.done"
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
    pass


class Connection(ABC):
    @abstractmethod
    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        pass

    @abstractmethod
    async def send(self, event: ChannelEvent) -> None:
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass


class Authenticate(ABC):

    @abstractmethod
    def verify(self, access_token: str) -> bool:
        pass


# --- event handlers --- #

ServerEventHandler = Callable[[Channel, ChannelEvent], Coroutine[None, None, bool]]


class BaseDuplexChannelServer(ChannelServer, ABC):

    def __init__(
            self,
            container: Container,
            client_connection: Connection,
            server_event_handlers: Dict[str, ServerEventHandler] | None = None,
    ):
        self.container = container
        self.connection = client_connection
        self.channel: Channel | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._logger: logging.Logger | None = None

        self._closing_event: ThreadSafeEvent = ThreadSafeEvent()
        self._closed_event: ThreadSafeEvent = ThreadSafeEvent()
        self._started: bool = False

        self._server_event_handlers: Dict[str, ServerEventHandler] = server_event_handlers or {}
        self._running_command_tasks: Dict[str, CommandTask] = {}
        self._running_command_tasks_lock = asyncio.Lock()

        self._policy_run_tasks: Dict[str, asyncio.Task] = {}
        self._policy_pause_task: asyncio.Task | None = None
        self._clearing_task: asyncio.Task | None = None

        self._has_running_task_event: asyncio.Event = asyncio.Event()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    async def arun_until_closed(self, channel: Channel) -> None:
        if self.loop is not None:
            raise RuntimeError(f'{self} is already running')

        self.loop = asyncio.get_running_loop()
        self.channel = channel
        await asyncio.to_thread(self.container.bootstrap)
        await self._bootstrap_channels()
        await self.connection.start()
        self._started = True
        task = await asyncio.create_task(self._main())
        await self.wait_closed()
        if not task.done():
            task.cancel()
        await task

    async def _bootstrap_channels(self) -> None:
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
            done, pending = asyncio.wait([consume_loop_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            try:
                await consume_loop_task
            except asyncio.CancelledError:
                # todo: log
                pass
            finally:
                await self.connection.close()

        except asyncio.CancelledError:
            # todo: log
            pass
        except ConnectionClosedError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
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
            _ = asyncio.create_task(self._consume_single_event(event))

    async def _consume_single_event(self, event: ChannelEvent) -> None:
        try:
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
        try:
            self.logger.info("Received event: %s", event)
            event_type = event['event_type']
            if event_type in self._server_event_handlers:
                handler = self._server_event_handlers[event_type]
                go_on = await handler(self.channel, event)
                if not go_on:
                    return
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

    async def _handel_clear(self, event: ClearCallEvent):
        channel_name = event.chan
        try:
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return
            await channel.client.clear()
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
            await self.connection.send(server_error.to_channel_event())
        finally:
            response = ClearDoneEvent(
                session_id=event.session_id,
                chan=channel_name,
            )
            await self.connection.send(response.to_channel_event())

    async def _handle_run_policy(self, event: RunPolicyEvent) -> None:
        channel_name = event.chan
        try:
            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return

            run_policy_task = asyncio.create_task(channel.client.policy_run())
            self._policy_run_tasks[channel_name] = run_policy_task
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
            if channel_name in self._policy_run_tasks:
                _ = self._policy_run_tasks.pop(channel_name)
            response = PausePolicyDoneEvent(session_id=event.session_id, chan=channel_name)
            await self.connection.send(response.to_channel_event())

    async def _handle_pause_policy(self, event: PausePolicyEvent) -> None:
        channel_name = event.chan
        try:
            if channel_name in self._policy_run_tasks:
                # remove from the waiting one, but not send response
                task = self._policy_run_tasks.pop(channel_name)
                task.cancel()
                # 可以跨协程 wait
                try:
                    # 目的是阻塞到运行结束.
                    await task
                except asyncio.CancelledError:
                    pass

            channel = self.channel.get_channel(channel_name)
            if channel is None or not channel.is_running():
                return
            if not channel.client.is_available():
                return

            await channel.client.policy_pause()
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
            response = PausePolicyDoneEvent(session_id=event.session_id, chan=channel_name)
            await self.connection.send(response.to_channel_event())

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
            # todo: log
            task.cancel()

    async def _handle_command_call(self, event: CommandCallEvent) -> None:
        channel = self.channel.get_channel(event.chan)
        if channel is None:
            response = event.not_available("channel %s not found" % event.chan)
            await self.connection.send(response.to_channel_event())
            return
        elif not self.channel.is_running():
            response = event.not_available("channel %s is not running" % event.chan)
            await self.connection.send(response.to_channel_event())
            return

        command = channel.client.get_command(event.name)
        if command is None or not command.is_available():
            response = event.not_available()
            await self.connection.send(response.to_channel_event())
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
            await self._remove_pending_task(task)
            # todo: 通讯如果存在问题, 会导致阻塞. 需要思考.
            result = task.result()
            response = event.done(result, task.errcode, task.errmsg)
            await self.connection.send(response.to_channel_event())

    async def _execute_task(self, task: CommandTask) -> None:
        try:
            execution = asyncio.create_task(task.dry_run())
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
            self._has_running_task_event.set()
        finally:
            self._running_command_tasks_lock.release()

    async def _remove_pending_task(self, task: CommandTask) -> None:
        await self._running_command_tasks_lock.acquire()
        try:
            cid = task.cid
            if cid in self._running_command_tasks:
                del self._running_command_tasks[cid]
        finally:
            self._running_command_tasks_lock.release()


class AbsDuplexChannelClient(ChannelClient):

    def __init__(
            self,
            session_id: str,
            container: IoCContainer,
            server_meta: ChannelMeta,
            server_connection: Connection,
            server_event_queue: asyncio.Queue[ChannelEvent | None],
            py_channel: Optional[PyChannel],
    ) -> None:
        self.session_id = session_id
        self.py_channel = py_channel
        self.name = py_channel.name()
        self.container = Container(parent=container, name=f"moss/duplex_channel/{self.name}")
        self.connection = server_connection
        self.server_event_queue = server_event_queue
        self._started = False
        self._logger: logging.Logger | None = None
        self._stop_event = ThreadSafeEvent()
        self._main_loop_done_event = ThreadSafeEvent()

        self._server_meta: ChannelMeta = server_meta
        self._cached_meta: Optional[ChannelMeta] = None

        # runtime
        # todo: 内存泄漏检查, 要定期清理.
        self._server_command_calls: Dict[str, CommandTask] = {}
        self._run_policy_done_event = ThreadSafeEvent()
        self._pause_policy_done_event = ThreadSafeEvent()
        self._clear_done_event = ThreadSafeEvent()

    def is_running(self) -> bool:
        self_running = self._started and not self._stop_event.is_set()
        return self_running and self._server_meta is not None and not self.connection.is_closed()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel client {self.name} is not running')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        if self._cached_meta is not None and not no_cache:
            return self._cached_meta.model_copy()
        self._cached_meta = self._build_meta()
        return self._cached_meta.model_copy()

    def _build_meta(self) -> ChannelMeta:
        self._check_running()
        meta = self._server_meta.model_copy()
        commands = {}
        for command_meta in self._server_meta.commands:
            commands[command_meta.name] = command_meta
        if self.py_channel is not None:
            local_meta = self.py_channel.client.meta()
            for command_meta in local_meta.commands:
                commands[command_meta.name] = command_meta
        meta.commands = list(commands.values())
        return meta

    def is_available(self) -> bool:
        return self.is_running() and self._server_meta is not None and self._server_meta.available

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        meta = self.meta()
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        if self.py_channel is not None:
            command = self.py_channel.client.get_command(name)
            if command is not None:
                return command.__call__

        async def _call_server(*args, **kwargs):
            task = CommandTask.get_from_context()
            cid = task.cid if task else uuid()
            event = CommandCallEvent(
                session_id=self.session_id,
                name=name,
                chan=self.name,
                command_id=cid,
                args=list(args),
                kwargs=dict(kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
            )
            try:
                await self.connection.send(event.to_channel_event())
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
                self._server_command_calls[cid] = wait_result_task
                # 等待异步返回结果.
                await wait_result_task.wait()
                wait_result_task.raise_exception()
                return wait_result_task.result()
            finally:
                if cid in self._server_command_calls:
                    t = self._server_command_calls.pop(cid)
                    t.cancel()

        return _call_server

    def get_command(self, name: str) -> Optional[Command]:
        meta = self.meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        if task.meta.chan != self.name:
            raise ValueError(f'command task of {task.meta} executed by wrong channel {self.name}')

        func = self._get_command_func(task.meta)
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            self._run_policy_done_event.clear()
            if self.py_channel is not None:
                await self.py_channel.client.policy_run()

            event = RunPolicyEvent(
                session_id=self.session_id,
                chan=self.name,
            )
            await self.connection.send(event.to_channel_event())
            await self._run_policy_done_event.wait()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self._run_policy_done_event.set()

    async def policy_pause(self) -> None:
        self._check_running()
        try:
            self._pause_policy_done_event.clear()
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = PausePolicyEvent(
                session_id=self.session_id,
                chan=self.name,
            )
            await self.connection.send(event.to_channel_event())
            await self._pause_policy_done_event.wait()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self._pause_policy_done_event.set()

    async def clear(self) -> None:
        self._check_running()
        try:
            self._clear_done_event.clear()
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = ClearCallEvent(
                session_id=self.session_id,
                chan=self.name,
            )
            await self.connection.send(event.to_channel_event())
            await self._clear_done_event.wait()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self._clear_done_event.set()

    async def _main_consume_server_event_loop(self):
        try:
            while not self._stop_event.is_set():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()
        finally:
            self._main_loop_done_event.set()

    async def _consume_server_event(self):
        item = await self.server_event_queue.get()
        if item is None:
            self._stop_event.set()

        if model := CommandDoneEvent.from_channel_event(item):
            pass
        elif model := ChannelMetaUpdateEvent.from_channel_event(item):
            pass
        elif model := PausePolicyDoneEvent.from_channel_event(item):
            pass
        elif model := ClearCallEvent.from_channel_event(item):
            pass
        elif model := ServerErrorEvent.from_channel_event(item):
            pass

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AbsDuplexChannel(Channel, ABC):

    def __init__(
            self,
            server_connection: Connection,
            block: bool,
            name: str,
            description: str = "",
    ):
        self._name = name
        self._server_connection = server_connection
        self._py_channel = PyChannel(name=name, description=description, block=block)

    def name(self) -> str:
        return self._name

    @property
    def client(self) -> ChannelClient:
        pass

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        pass

    def new_child(self, name: str) -> Self:
        pass

    def children(self) -> Dict[str, "Channel"]:
        pass

    def descendants(self) -> Dict[str, "Channel"]:
        pass

    def get_channel(self, name: str) -> Optional[Self]:
        pass

    def is_running(self) -> bool:
        pass

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        pass

    @property
    def build(self) -> Builder:
        return self._py_channel.build
