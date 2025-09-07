import threading

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.concepts.channel import Channel
from ghoshell_moss.concepts.shell import MOSSShell, ShellRuntime
from ghoshell_moss.concepts.interpreter import AsyncInterpreter, Interpreter
from ghoshell_moss.concepts.command import CommandTask, TaskStack, Command
from ghoshell_moss.concepts.errors import StopTheLoop, FatalError, CommandError
from typing import Dict, Optional, Set, List, Tuple, Callable, Coroutine, Iterable
from collections import deque
from anyio.abc import TaskGroup

import anyio
import asyncio


class ChannelTask:

    def __init__(
            self,
            channel: Channel,
            dispatcher: Callable[[str, CommandTask], Coroutine],
            loop: asyncio.AbstractEventLoop,
            shell_closed: asyncio.Event,
    ):
        self.chan: Channel = channel
        self._name = channel.name()
        self._dispatcher = dispatcher
        self._loop = loop
        self._shell_closed = shell_closed

        # status
        self._starting = False
        self._started = False
        self._command_task_queue: deque[CommandTask] = deque()
        self._close_event: asyncio.Event = asyncio.Event()
        self._running_command_tasks: Set[asyncio.Task] = set()

        # runtime properties
        self._has_pending_task_event: asyncio.Event = asyncio.Event()
        self._defer_clear: bool = False
        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()
        self._main_loop_task: Optional[asyncio.Task] = None

    async def start(self):
        """
        bootstrap
        """
        if self._starting:
            return
        self._starting = True
        self._has_pending_task_event = asyncio.Event()
        self._close_event = asyncio.Event()
        await self.chan.start()
        self._started = True
        self._main_loop_task = self._loop.create_task(self.run_mail_loop())

    async def wait_until_idle(self):
        await self._idle_event.wait()

    async def close(self):
        if not self._started:
            raise FatalError(f'Channel {self._name} close but has not started yet.')
        if self._close_event.is_set():
            return
        self._close_event.set()
        await self.clear()
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            await self._main_loop_task
        await self.chan.close()

    async def run_mail_loop(self):
        try:
            while not self._close_event.is_set() and not self._shell_closed.is_set():
                await self._has_pending_task_event.wait()
                if len(self._command_task_queue) == 0:
                    self._has_pending_task_event.clear()
                    continue
                cmd_task = self._command_task_queue.popleft()
                if cmd_task.none_block:
                    _ = asyncio.create_task(self._execute_task(cmd_task))
                else:
                    # wait until done
                    await self._execute_task(cmd_task)
        except Exception:
            # todo
            raise
        finally:
            await self.clear()
            self._close_event.set()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return None

    def _check_started(self):
        if not self._starting:
            raise FatalError(f"Channel {self._name} is not running")
        elif self._close_event.is_set():
            raise FatalError(f"Channel {self._name} is shutdown")

    def name(self) -> str:
        return self._name

    def is_running(self) -> bool:
        """
        declare whether the channel is running
        """
        return self._started and not self._close_event.is_set()

    def is_idle(self) -> bool:
        """
        declare whether the channel is busy
        """
        if not self.is_running():
            return False
        return self._idle_event.is_set()

    async def append(self, *tasks: CommandTask) -> None:
        if not self.is_running():
            # todo: log
            return
        tasks = list(tasks)
        if len(tasks) > 0:
            self._idle_event.clear()
            if self._defer_clear:
                # clear first
                await self.clear()
            for task in tasks:
                task.set_state('pending')
                self._command_task_queue.append(task)
            self._has_pending_task_event.set()

    async def prepend(self, *tasks: CommandTask) -> None:
        if not self.is_running():
            # todo: log
            return
        tasks = list(tasks)
        if len(tasks) > 0:
            self._idle_event.clear()
            if self._defer_clear:
                await self.clear()
            for task in tasks:
                task.set_state('pending')
            self._command_task_queue.extendleft(tasks)
            self._has_pending_task_event.set()

    async def _cancel_running(self) -> None:
        tasks = list(self._running_command_tasks)
        self._running_command_tasks.clear()
        if len(tasks) > 0:
            for task in tasks:
                task.cancel()

    async def clear(self) -> None:
        self._defer_clear = False
        self._command_task_queue.clear()
        self._has_pending_task_event.clear()
        # todo
        await self._cancel_running()

    async def defer_clear(self) -> None:
        self._command_task_queue.clear()
        self._defer_clear = True

    async def _execute_task(self, cmd_task: CommandTask) -> None:
        self._idle_event.clear()
        task = self._loop.create_task(self._execute_task_stack(cmd_task))
        try:
            self._running_command_tasks.add(task)
            await task
        except asyncio.CancelledError:
            return
        finally:
            self._running_command_tasks.remove(task)
            if len(self._running_command_tasks) == 0 and len(self._command_task_queue) == 0:
                self._idle_event.set()

    async def _execute_task_stack(self, cmd_task: CommandTask) -> None:
        try:
            stack = [cmd_task]
            while len(stack) > 0:
                executing = stack.pop(0)
                temp_tasks = await self._execute_single_task(executing)
                if temp_tasks is not None:
                    temp_tasks.extend(stack)
                    stack = temp_tasks
        except asyncio.CancelledError:
            cmd_task.cancel(f"command {cmd_task.chan}:{cmd_task.name} cancelled")
            return

        except CommandError as e:
            cmd_task.fail(e)

        except (FatalError, StopTheLoop):
            # todo
            raise
        except Exception as exc:
            # todo
            cmd_task.fail(str(exc))
            return None
        finally:
            if not cmd_task.is_done():
                cmd_task.cancel("command {cmd_task.chan}:{cmd_task.name} cancelled")

    async def _execute_single_task(self, cmd_task: CommandTask) -> Optional[List[CommandTask]]:
        cmd_chan = cmd_task.chan
        if cmd_chan != self._name:
            children = self.chan.children()
            if cmd_chan in children:
                # directly send to child
                await self._dispatcher(cmd_task.name, cmd_task)
                # and break
                return None

            for child in children.values():
                if cmd_chan in child.descendants():
                    # send to some child
                    await self._dispatcher(child.name(), cmd_task)
                    return None
            # not children show have this command.
            cmd_task.fail(f"channel {cmd_task.chan} not found or command {cmd_task.name}")
            return
        elif not self.chan.runtime.available():
            cmd_task.cancel(f"Channel {self.name()} not available for command {cmd_task.name}")
            return

        cmd_task.set_state('running')
        # get from self channel
        command = self.chan.runtime.get_command(cmd_task.name)
        if command is None:
            cmd_task.fail(f"command {cmd_task.name} at channel {cmd_task.chan} not found")
            return None

        # cmd task may be canceled outside the loop
        running_task = self._loop.create_task(command(*cmd_task.args, **cmd_task.kwargs))
        done, pending = await asyncio.wait(
            [running_task, cmd_task.wait_done()],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if running_task not in done:
            running_task.cancel()
        result = await running_task
        # return a stack
        if isinstance(result, TaskStack):
            cmd_task.resolve(None)
            return list(result.tasks)

        cmd_task.resolve(result)
        return None


class AsyncMOSSShellRuntime(ShellRuntime):

    def __init__(
            self,
            main_chan: Channel,
            loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.loop = loop or asyncio.get_running_loop()
        self.main_chan = main_chan
        self._starting = False
        self._started = False
        self._closed_event = asyncio.Event()
        self._main_chan_task: Optional[ChannelTask] = None
        self._channel_tasks: Dict[str, ChannelTask] = {}
        self._interpreter: Optional[Interpreter] = None

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        waits = []
        for task in self._channel_tasks.values():
            waits.append(task.wait_until_idle())
        wait_task = asyncio.wait(waits)
        await asyncio.wait_for(wait_task, timeout)

    async def system_prompt(self) -> str:
        pass

    def commands(self) -> Dict[str, List[Command]]:
        result = {}
        channels = self.main_chan.descendants()
        channels[self.main_chan.name()] = self.main_chan

        for name, chan in channels.items():
            result[name] = list(chan.runtime.commands())
        return result

    async def clear(self, *chans: str) -> None:
        if self._interpreter is not None:
            self._interpreter.cancel()
            self._interpreter = None
        clearing = []
        clearing_names = self._get_descendants_channel_names(*chans)
        for name in clearing_names:
            if name in self._channel_tasks:
                task = self._channel_tasks[name]
                clearing.append(task.clear())
        await asyncio.gather(*clearing)

    async def defer_clear(self, *chans: str) -> None:
        if self._interpreter is not None:
            self._interpreter.cancel()
            self._interpreter = None
        clearing_names = self._get_descendants_channel_names(*chans)
        for name in clearing_names:
            if name in self._channel_tasks:
                task = self._channel_tasks[name]
                await task.defer_clear()

    def _get_descendants_channel_names(self, *chans: str) -> List[str]:
        names = list(chans)
        if len(names) == 0:
            return list(self.main_chan.descendants().keys())
        for name in names:
            if name in self._channel_tasks:
                chan_task = self._channel_tasks[name]
                names.extend(chan_task.chan.descendants().keys())
        return names

    def is_idle(self) -> bool:
        if not self._started or self._closed_event.is_set():
            return False
        elif not self._main_chan_task.is_idle():
            return False
        else:
            for task in self._channel_tasks.values():
                if not task.is_idle():
                    return False
        return True

    def is_running(self) -> bool:
        return self._started and not self._closed_event.is_set()

    async def start(self) -> None:
        if self._closed_event.is_set():
            raise FatalError(f"channel {self.main_chan.name} is already closed")
        elif self._starting:
            raise FatalError(f"channel {self.main_chan.name} is already started")
        self._starting = True
        await self._start_all_tasks()
        self._started = True

    async def stop(self) -> None:
        if self._closed_event.is_set():
            return
        self._closed_event.set()
        if self._interpreter is not None:
            self._interpreter.cancel()
            self._interpreter = None

        closing_tasks = []
        for task in self._channel_tasks.values():
            closing_tasks.append(task.close())
        await asyncio.gather(*closing_tasks)

    async def append(self, *tasks: CommandTask) -> None:
        await self._main_chan_task.append(*tasks)

    async def dispatch(self, channel_name: str, cmd_task: CommandTask) -> None:
        """
        确保不成环.
        """
        if channel_name in self._channel_tasks:
            task = self._channel_tasks[channel_name]
            await task.append(cmd_task)
        elif channel_name in self.main_chan.descendants():
            child = self.main_chan.descendants()[channel_name]
            task = ChannelTask(child, self.dispatch, self.loop, self._closed_event)
            _ = self.loop.create_task(task.start())

    def _new_child_task(self, child: Channel) -> ChannelTask:
        if not child.is_running():
            raise FatalError(f"channel {child.name} not running")
        child_task = ChannelTask(child, self.dispatch, self.loop, self._closed_event)
        return child_task

    async def _start_all_tasks(self) -> None:
        tasks = []
        self._main_chan_task = ChannelTask(self.main_chan, self.dispatch, self.loop, self._closed_event)
        self._channel_tasks[self.main_chan.name()] = self._main_chan_task
        tasks.append(self._main_chan_task.start())
        for child in self.main_chan.descendants().values():
            name = child.name()
            child_task = self._new_child_task(child)
            self._channel_tasks[name] = child_task
            tasks.append(child_task.start())
        _ = await asyncio.gather(*tasks, loop=self.loop)
