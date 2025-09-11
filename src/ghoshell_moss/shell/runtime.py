from ghoshell_moss.concepts.channel import Channel, ChannelRuntime, ChannelMeta
from ghoshell_moss.concepts.command import BasicCommandTask, CommandTaskSeq
from ghoshell_moss.concepts.errors import StopTheLoop, FatalError
from ghoshell_container import IoCContainer
from typing import Dict, Optional, Set, Awaitable, List
from typing_extensions import Self
from collections import deque
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from anyio.abc import TaskGroup

import anyio
import asyncio
import threading


class ChannelRuntimeImpl(ChannelRuntime):

    def __init__(self, channel: Channel):
        self._chan: Channel = channel
        self._name = channel.name()

        # status
        self._running = False
        self._bootstrapped = False
        self._bootstrap_thread: Optional[threading.Thread] = None
        self._is_shutdown = False

        # runtime properties
        self._children: Dict[str, ChannelRuntimeImpl] = {}
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._command_task_queue: deque[BasicCommandTask] = deque()
        self._has_pending_task_event: Optional[asyncio.Event] = None
        self._defer_clear: bool = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._running_tasks: Set[BasicCommandTask] = set()

    def get_child(self, name: str) -> Optional["ChannelRuntimeImpl"]:
        self._check_running()
        if name in self._children:
            return self._children[name]

        child_chan = self._chan.get_channel(name)
        if child_chan is None:
            return None
        child_runtime = ChannelRuntimeImpl(child_chan)
        child_runtime.bootstrap()
        self._children[name] = child_runtime
        return child_runtime

    def bootstrap(self):
        if self._bootstrapped:
            return
        self._bootstrapped = True
        self._bootstrap_thread = threading.Thread(target=self._bootstrap, daemon=True)
        self._bootstrap_thread.start()

    def shutdown(self):
        if not self._bootstrapped or self._is_shutdown:
            return
        self._is_shutdown = True
        self._running_loop.call_soon_threadsafe(self._shutdown_event.set)
        for child in self._children.values():
            child.shutdown()

    def join(self):
        self.shutdown()
        for child in self._children.values():
            child.join()
        if self._bootstrap_thread is not None:
            self._bootstrap_thread.join()

    def _bootstrap(self):
        loop = asyncio.new_event_loop()
        try:
            self._running_loop = loop
            loop.run_until_complete(self._run_main_loop())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()

    def _check_running(self):
        if not self._running:
            raise FatalError(f"Channel {self._name} is not running")
        elif self._shutdown_event.is_set():
            raise FatalError(f"Channel {self._name} is shutdown")

    def name(self) -> str:
        return self._name

    def is_running(self) -> bool:
        """
        declare whether the channel is running
        """
        return self._running and not self._shutdown_event.is_set()

    def is_busy(self) -> bool:
        """
        declare whether the channel is busy
        """
        if not self.is_running():
            return False
        return self._has_pending_task_event.is_set() or len(self._running_tasks) > 0

    def append(self, *tasks: BasicCommandTask) -> None:
        if not self.is_running():
            # todo: log
            return
        tasks = list(tasks)
        if len(tasks) > 0:
            if self._defer_clear:
                # clear first
                self.clear()
            self._running_loop.call_soon_threadsafe(self._append, tasks)

    async def _append(self, tasks: List[BasicCommandTask]) -> None:
        for task in tasks:
            task.set_state('pending')
            self._command_task_queue.append(task)
        self._has_pending_task_event.set()

    def prepend(self, *tasks: BasicCommandTask) -> None:
        tasks = list(tasks)
        if len(tasks) > 0:
            if self._defer_clear:
                self.clear()
            self._running_loop.call_soon_threadsafe(self._prepend, *tasks)

    async def _prepend(self, tasks: List[BasicCommandTask]) -> None:
        for task in tasks:
            task.set_state('pending')
        self._command_task_queue.extendleft(tasks)
        self._has_pending_task_event.set()

    async def cancel_running(self, reason: str) -> None:
        for task in self._running_tasks:
            task.cancel(reason)

    def clear(self) -> None:
        if self.is_running():
            self._command_task_queue.clear()
            self._defer_clear = False
            # todo
            self._running_loop.call_soon_threadsafe(self.cancel_running, "Cancelled")
            for child in self._chan.children():
                # clear children
                if child.is_running():
                    child.runtime.clear()

    def defer_clear(self) -> None:
        self._defer_clear = True
        for child in self._chan.children():
            child.runtime.defer_clear()

    async def _run_main_loop(self):
        self._has_pending_task_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._running = True
        try:
            while not self._shutdown_event.is_set():
                # wait for new command
                await self._has_pending_task_event.wait()
                if len(self._command_task_queue) == 0:
                    self._has_pending_task_event.clear()
                    continue
                cmd_task = self._command_task_queue.popleft()
                if cmd_task.none_block:
                    task = asyncio.create_task(self._execute_task(cmd_task))
                    await asyncio.shield(task)
                    # do not block the main loop
                    continue
                else:
                    # wait until done
                    await self._execute_task(cmd_task)
        except Exception:
            # todo
            raise
        finally:
            self._running = False
            self._shutdown_event.set()

    async def _execute_task(self, cmd_task: BasicCommandTask) -> None:
        stack = [cmd_task]
        while len(stack) > 0:
            executing = stack.pop(0)
            result = await self._execute_single_task(executing)
            if result is not None:
                stack = result + stack

    async def _execute_single_task(self, cmd_task: BasicCommandTask) -> Optional[List[BasicCommandTask]]:
        try:
            if cmd_task.chan != self._name:
                child = self._chan.get_channel(cmd_task.chan)
                if child is not None:
                    child.runtime.append(cmd_task)
                else:
                    cmd_task.fail(f"function {cmd_task.name} channel {cmd_task.chan} not found")
                return
            cmd_task.set_state('running')
            self._running_tasks.add(cmd_task)
            command = self._chan.get_command(cmd_task.name)

            running_task = asyncio.create_task(command(*cmd_task.args, **cmd_task.kwargs))
            done, pending = await asyncio.wait(
                [running_task, cmd_task.wait_done()],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            if running_task not in done:
                running_task.cancel()
            result = await running_task
            # return a stack
            if isinstance(result, CommandTaskSeq):
                return list(result.tasks)

            cmd_task.resolve(result)
            return None

        except asyncio.CancelledError:
            return

        except (FatalError, StopTheLoop):
            # todo
            raise
        except Exception as exc:
            # todo
            cmd_task.fail(str(exc))
            return None
        finally:
            if not cmd_task.is_done():
                cmd_task.cancel()
            self._running_tasks.remove(cmd_task)
