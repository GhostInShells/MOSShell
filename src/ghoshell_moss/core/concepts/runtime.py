import logging
from typing import Optional, AsyncIterable, AsyncIterator, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from .channel import ChannelBroker, Channel, ChannelFullPath, ChannelPaths, ChannelMeta
from .command import (
    CommandTask, CommandTaskStateType, CommandTaskStack, CommandUniqueName, Command, CommandWrapper,
    BaseCommandTask,
)
from .errors import CommandErrorCode, FatalError
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Container
import asyncio
import contextvars
from contextlib import asynccontextmanager
import threading


class ChannelRuntime(ABC):

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    async def children(self) -> dict[str, Self]:
        """
        children runtime
        """
        pass

    @abstractmethod
    async def wait_idled(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


_ChannelId = str
_BrokerId = str
_TaskId = str
_ChannelNames = list[str]
_TaskWithPaths = tuple[_ChannelNames, CommandTask]

ChannelRuntimeCtxVar = contextvars.ContextVar('MOSSChannelRuntimeCtx')


class ChannelTreeRuntime:
    """
    Channel 的运行时. 用来调度各种 task.
    目标是实现线程安全的 Runtime.
    """

    def __init__(
            self,
            path: ChannelFullPath,
            channel: Channel,
            container: IoCContainer,
            logger: LoggerItf | None = None,
    ):
        self._path = path
        self._channel = channel
        self._broker: Optional[ChannelBroker] = None
        self._name = channel.name()

        # 不创建递归的 Container.
        self._container = container
        self._logger: LoggerItf | None = logger or container.get(LoggerItf) or logging.getLogger('moss')

        # 运行时的 children runtime.
        self._children_runtimes: dict[_ChannelId, ChannelTreeRuntime] = {}
        self._children_name_to_ids: dict[str, _ChannelId] = {}

        # runtime
        self._block_action_lock = asyncio.Lock()
        self._blocking_task_empty_event = asyncio.Event()
        self._pending_task_queue: asyncio.Queue[_TaskWithPaths | None] = asyncio.Queue(1000)
        self._handling_task: CommandTask | None = None
        self._paused_event = asyncio.Event()

        # 一次只能执行一个.
        self._executing_task_soon_queue: asyncio.Queue[CommandTask | None] = asyncio.Queue(1)
        self._defer_clear: bool = False

        self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        self._main_loop_task: asyncio.Task | None = None

        self._starting: bool = False
        self._started: bool = False
        self._stopping_event = asyncio.Event()
        self._stopped_event = asyncio.Event()
        self.log_prefix = ""

    @classmethod
    def bootstrap(cls, channel: Channel, container: IoCContainer | None = None) -> Self:
        container = Container(name="MossChannelTreeRuntimeContainer/{}".format(channel.name()), parent=container)
        runtime = cls(path="", channel=channel, container=container)
        return runtime

    @property
    def channel_fullpath(self) -> ChannelFullPath:
        return self._path

    @property
    def channel(self) -> Channel:
        return self._channel

    @property
    def broker(self) -> ChannelBroker:
        return self._broker

    @property
    def name(self) -> str:
        return self._name

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self._container.get(LoggerItf) or logging.getLogger('moss')
        return self._logger

    def is_running(self) -> bool:
        return self._started and not self._stopping_event.is_set() and self._broker and self.broker.is_running()

    def is_available(self) -> bool:
        return self._broker and self._broker.is_available()

    def is_blocking_task_empty(self) -> bool:
        return self.is_running() and self._blocking_task_empty_event.is_set()

    async def fetch_node(self, path: ChannelFullPath) -> Optional[Self]:
        paths = Channel.split_channel_path_to_names(path)
        return await self._fetch_node_by_paths(paths)

    async def _fetch_node_by_paths(self, paths: ChannelPaths) -> Optional[Self]:
        if len(paths) == 0:
            return self
        child_name = paths[0]
        further_paths = paths[1:]
        runtime = await self._fetch_child_runtime(child_name)
        if runtime is None:
            return None
        if len(further_paths) == 0:
            return runtime
        return runtime._fetch_node_by_paths(further_paths)

    async def wait_blocking_task_empty(self) -> None:
        if not self.is_running():
            return
        await self._blocking_task_empty_event.wait()

    async def refresh_all_metas(self, callback: bool = True) -> None:
        if not self.is_running():
            return
        await self._loop.create_task(self._broker.refresh_meta(callback))
        refreshing = []
        for child_name, child_channel in self._channel.children().items():
            runtime = await self._fetch_child_runtime_by_channel(child_name, child_channel)
            if runtime is not None:
                refreshing.append(self._loop.create_task(runtime.refresh_all_metas(callback)))
        done = await asyncio.gather(*refreshing)
        for r in done:
            if isinstance(r, Exception):
                self.logger.exception("%s failed to refresh meta: %s", self.log_prefix, r)

    def metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        if not self.is_running():
            return {}
        result = {self._path: self._broker.self_meta()}
        for runtime in self._children_runtimes.values():
            for path, meta in runtime.metas().items():
                result[path] = meta
        return result

    def get_command(self, name: str, *, chan: ChannelFullPath = "") -> Optional[Command]:
        paths = Channel.split_channel_path_to_names(chan)
        command = self.get_command_by_paths(paths, name)
        return CommandWrapper(
            meta=command.meta().model_copy(update={"chan": chan}),
            func=command.__call__,
            available_fn=command.is_available,
        )

    def create_command_task(
            self,
            name: str,
            *,
            chan: ChannelFullPath = "",
            args: tuple | None = None,
            kwargs: dict | None = None
    ) -> CommandTask:
        command = self.get_command(name, chan=chan)
        if command is None:
            raise LookupError(f'Could not find command "{name}"')
        args = args or ()
        kwargs = kwargs or {}
        return BaseCommandTask.from_command(command, *args, **kwargs)

    def commands(self, available_only: bool = False) -> dict[str, Command]:
        if not self.is_running():
            return {}
        result: dict[CommandUniqueName, Command] = {}
        for name, command in self._broker.self_commands(available_only=available_only).items():
            unique_name = Command.make_uniquename(self._path, name)
            result[unique_name] = CommandWrapper(
                meta=command.meta().model_copy(update={"chan": self._path}),
                func=command.__call__,
                available_fn=command.is_available,
            )
        for runtime in self._children_runtimes.values():
            sub_commands = runtime.commands(available_only)
            result.update(sub_commands)
        return result

    def get_command_by_paths(self, paths: ChannelPaths, name: str) -> Optional[Command]:
        if len(paths) == 0:
            command = self._broker.get_self_command(name)
            return command

        child_name = paths[0]
        further_paths = paths[1:]
        if child_name not in self._children_name_to_ids:
            return None
        child_id = self._children_name_to_ids[child_name]
        runtime = self._children_runtimes.get(child_id)
        if runtime is None:
            return None
        return runtime.get_command_by_paths(further_paths, name)

    async def put_task(self, *tasks: CommandTask) -> None:
        """
        入栈 task.
        """
        # 入栈检查.
        for task in tasks:
            task = self._check_task_runnable(task)
            if task is None:
                return
            paths = Channel.split_channel_path_to_names(task.meta.chan)
            await self.put_task_with_paths(paths, task)

    def _check_task_runnable(self, task: CommandTask) -> Optional[CommandTask]:
        if task.done():
            # 丢弃
            return None
        elif not self.is_running():
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"channel not running"))
            return None
        elif not self.broker.is_connected():
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"channel not connected"))
            return None
        elif not self.broker.is_available():
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"channel not available"))
            return None
        return task

    async def pause(self) -> None:
        """
        递归地暂停当前和所有的子 Channel.
        作为一种安全锁. pause 状态下仍然可以接受新的指令.
        """
        if not self.is_running():
            return
        # 递归清空所有子 runtime 和执行中的任务.
        self._paused_event.set()
        await self.clear()
        pause_tasks = [asyncio.create_task(self._broker.pause())]
        for runtime in self._children_runtimes.values():
            pause_tasks.append(asyncio.create_task(runtime.pause()))
        done = await asyncio.gather(*pause_tasks, return_exceptions=True)
        for t in done:
            if isinstance(t, Exception):
                self.logger.error("%s pause exception %r", self.log_prefix, t)

    async def put_task_with_paths(self, paths: _ChannelNames, task: CommandTask) -> None:
        """
        基于路径将任务入栈.
        """
        # 设置运行通道记录.
        task.send_through.append(self.name)

        # 有任何新命令进入, 则终止 pause 状态. pause 状态会阻止进入 idle 状态.
        self._paused_event.clear()
        # 设置 task id 到 pending map 里.
        try:
            # 是自己的, 而且是要立刻执行的任务.
            # call soon 这类任务
            if len(paths) == 0 and task.meta.call_soon:
                if task.meta.blocking:
                    # 需要立刻执行, 而且是一个阻塞类的任务, 则会清空所有运行中的任务. 这其中也递归地包含子节点的任务.
                    await self.clear()
                # 立刻将它放入 broker 的执行队列. 它会被尽快执行.
                await self._broker.push_task(task)
                # 并不阻塞等待结果, 而是立刻返回.
                return

            # 普通的任务, 则会被丢入阻塞队列中排队执行.
            _queue = self._pending_task_queue
            # 入栈.
            _queue.put_nowait((paths, task))
            # 标记有任务入栈.
            self._blocking_task_empty_event.clear()
        except asyncio.QueueFull:
            task.fail(CommandErrorCode.FAILED.error(f"channel queue is full, clear first"))

    async def clear(self):
        """
        清空自己和所有的子节点.
        """
        if not self.is_running():
            return
        # 先确认清空 pending, 面得有并行错误.
        await self._clear_self()
        # 清空自己自身的 broker.
        refresh_tasks = [asyncio.create_task(self._broker.clear())]
        # 清空子孙 runtime.
        for runtime in self._children_runtimes.values():
            # 先序遍历, 递归清空.
            refresh_tasks.append(asyncio.create_task(runtime.clear()))
        await asyncio.gather(*refresh_tasks)

    async def _clear_self(self) -> None:
        if not self.is_running():
            return
        self._defer_clear = False

        # 清空队列.
        pending_queue = self._pending_task_queue
        self._pending_task_queue = asyncio.Queue(100)
        while not pending_queue.empty():
            r = await pending_queue.get()
            if r is None:
                continue
            paths, task = r
            if task is not None and not task.done():
                task.fail(CommandErrorCode.CLEARED.error("channel cleared"))
        # 放入一个毒丸, 免得极端情况死锁.
        pending_queue.put_nowait(None)

        # 清空正在运行的任务.
        if self._handling_task is not None and not self._handling_task.done():
            self._handling_task.fail(CommandErrorCode.CLEARED.error(f"channel cleared"))
        self._handling_task = None

    async def _wait_children_blocking_done(self) -> None:
        wait_all = []
        for runtime in self._children_runtimes.values():
            wait_all.append(self._loop.create_task(runtime.wait_blocking_task_empty()))
        await asyncio.gather(*wait_all)

    async def _get_children_runtimes(self) -> dict[str, Self]:
        result = {}
        for name, cid in self._children_name_to_ids.items():
            if cid in self._children_runtimes:
                result[name] = self._children_runtimes[cid]
        return result

    async def _fetch_child_runtime(self, child_name: str) -> Optional[Self]:
        """
        在动态的 Channel 中查找子节点, 获取一个 Channel Runtime.
        """
        child_channel = self._channel.children().get(child_name)
        if child_channel is None:
            if child_name in self._children_name_to_ids:
                await self._remove_child_runtime(child_name)
            return None
        try:
            return await self._fetch_child_runtime_by_channel(child_name, child_channel)
        except Exception as exc:
            self.logger.exception(
                "%s fetch child runtime %s failed: %s",
                self.log_prefix, child_name, exc,
            )
            return None

    async def _fetch_child_runtime_by_channel(self, name: str, channel: Channel) -> Self:
        if name in self._children_name_to_ids:
            exists_id = self._children_name_to_ids[name]
            # 存在并且相等. 是同一个 channel 创建的.
            if exists_id == channel.id():
                runtime = self._children_runtimes[exists_id]
                if runtime is not None:
                    return runtime
            else:
                # 删除同名, 但是不存在了的 runtime.
                await self._remove_child_runtime(name)
        new_id = channel.id()
        new_runtime = ChannelTreeRuntime(
            path=Channel.join_channel_path(self._path, name),
            channel=channel,
            container=self._container,
        )
        # 启动 new_runtime.
        await self._loop.create_task(new_runtime.start())
        self._children_name_to_ids[name] = new_id
        self._children_runtimes[new_id] = new_runtime

    async def wait_connected(self) -> None:
        if not self.is_running():
            return
        await self._broker.wait_connected()
        await self.refresh_all_metas()

    async def execute_command(
            self,
            name: str,
            *,
            chan: ChannelFullPath = "",
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> Any:
        task = self.create_command_task(name, chan=chan, args=args, kwargs=kwargs)
        await self.put_task(task)
        return await task

    async def _remove_child_runtime(self, child_name: str) -> None:
        if child_name not in self._children_name_to_ids:
            return
        child_id = self._children_name_to_ids.pop(child_name)
        if child_id not in self._children_runtimes:
            return
        runtime = self._children_runtimes.pop(child_id)
        # 让它默默地关闭掉.
        _ = self._loop.create_task(runtime.stop())

    async def _is_children_blocking_task_done(self) -> bool:
        """
        递归判断子孙节点是否空了.
        """
        children = await self._get_children_runtimes()
        for runtime in children.values():
            if not runtime.is_blocking_task_empty():
                return False
        return True

    async def wait_blocking_task_done(self) -> None:
        """
        等待当前 runtime 和它所有子节点的运行都清空.
        """
        if not self.is_running():
            return
        await self._blocking_task_empty_event.wait()

    async def _consume_task_loop(self) -> None:
        try:
            while not self._stopping_event.is_set():
                _pending_queue = self._pending_task_queue
                # 如果队列是空的, 则要看看是否能够启动 idle.
                if _pending_queue.empty() and not self._blocking_task_empty_event.is_set():
                    get_next_cmd_task = asyncio.create_task(_pending_queue.get())
                    children_none_block = asyncio.create_task(self._wait_children_blocking_done())

                    done, pending = await asyncio.wait(
                        [get_next_cmd_task, children_none_block],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    # 先拿到了子孙节点都被清空了.
                    if children_none_block in done:
                        # 这种情况下就真的可以 idle 了.
                        if not self._paused_event.is_set():
                            await self._broker.idle()
                        self._blocking_task_empty_event.set()
                        continue
                    # 另一种情况, 就是先拿到了 Item.
                    item = await get_next_cmd_task
                else:
                    # 阻塞等待下一个结果.
                    get_item = asyncio.create_task(_pending_queue.get())
                    stop_task = asyncio.create_task(self._stopping_event.wait())
                    done, pending = await asyncio.wait([stop_task, get_item], return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()
                    item = await get_item

                # 可能拿到了 clear 清空后的毒丸.
                if item is None:
                    self.logger.info("%s receive none from pending task queue", self.log_prefix)
                    continue

                paths, task = item
                # handle task 函数是阻塞的, 这意味着:
                # 1. 它会阻塞后续拿到新的任务.
                # 2. 如果它执行了子任务, 其实不会阻塞.
                # 3. 如果它执行了 none-blocking 的任务, 也不会阻塞.
                # 4. 只有它执行的目标任务是自己的任务, 才会阻塞. 而且要阻塞等待儿孙们都执行完了, 才轮到自己执行.
                await self._handle_task(paths, task)
        except asyncio.CancelledError as e:
            # 允许被 cancel.
            self.logger.info("%s Cancel consuming pending task loop: %r", self.log_prefix, e)
        finally:
            self.logger.info("%s Finished executing loop", self.log_prefix)

    async def _dispatch_children_task(self, paths: ChannelPaths, task: CommandTask) -> None:
        child_name = paths[0]
        # 子节点在路径上不存在.
        runtime = await self._fetch_child_runtime(child_name)
        if runtime is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"channel `{task.meta.chan}` not found"))
            return
        # 直接发送给子树.
        further_paths = paths[1:]
        await runtime.put_task_with_paths(further_paths, task)
        return

    async def _handle_task(self, paths: _ChannelNames, task: CommandTask) -> None:
        """
        尝试运行一个 task. 这个运行周期是全局唯一, 阻塞的.
        """
        try:
            # 确保这个任务也可以被 clear 掉.
            self._handling_task = task
            await asyncio.sleep(0)
            # 检查是不是子节点的任务.
            if len(paths) > 0:
                await self._dispatch_children_task(paths, task)
                return

            # 任务是异步执行的, 则可以马上调度 broker 执行它.
            # 所以非阻塞任务任何时候都会优先执行. 它不会被子孙阻塞, 也不会阻塞后面的任务.
            if not task.meta.blocking:
                # 非阻塞任务立刻执行.
                await self._broker.push_task(task)
                # 而且不需要阻塞等待.
                return

            # 由于子孙轨道可以阻塞父轨道, 因此需要检查和等待.
            if not self._is_children_blocking_task_done():
                # 等待子孙节点的阻塞周期都完成.
                wait_children_done = self._loop.create_task(self._wait_children_blocking_done())
                wait_task_done_outside = self._loop.create_task(task.wait(throw=False))
                # 看看谁先到.
                done, pending = await asyncio.wait(
                    [wait_children_done, wait_task_done_outside],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                # 最先等到的不是儿孙们都执行完毕了, 这就意味着出了别的意外.
                if wait_task_done_outside in done:
                    # task 肯定 done 了.
                    return

            # 执行任务, 并且解决回调的问题.
            await asyncio.sleep(0)
            await self._execute_self_blocking_task(task)

        except asyncio.CancelledError:
            raise
        except FatalError as e:
            # 系统级别的致命异常都会终止运行.
            self.logger.info("%s handle pending task with fatal error: %r", self.log_prefix, e)
            self._stopping_event.set()
        except Exception as e:
            self.logger.info("%s handle pending task exception: %r", self.log_prefix, e)
            # 所有在执行 handle pending task 阶段抛出的异常, 都不向上中断.
        finally:
            self._handling_task = None

    async def _execute_self_blocking_task(self, task: CommandTask) -> None:
        """
        运行属于自己这个 channel 的 task, 让它进入到 executing group 中.
        """
        try:
            # 先不着急, 复制一份, 用来处理特殊的返回值逻辑.
            execute_task = task.copy()
            # 让 broker 去执行它.
            await self._broker.push_task(execute_task)
            # 等待 execute_task 运行结束.
            origin_task_done = asyncio.create_task(task.wait(throw=False))
            execute_task_done = asyncio.create_task(execute_task.wait(throw=False))
            done, pending = await asyncio.wait(
                [origin_task_done, execute_task_done],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if execute_task_done not in done:
                # origin task 已经运行结束.
                return

            if e := execute_task.exception():
                # 传递一下异常.
                task.fail(e)
                return

            result = await execute_task
            # 如果返回值是 stack, 则意味着要循环堆栈.
            if isinstance(result, CommandTaskStack):
                # 执行完所有的堆栈. 同时设置真实被执行的任务.
                await self._fulfill_task_with_its_result_stack(task, result)
            else:
                # 赋值给原来的 task.
                task.resolve(result)

        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            # 不会往上报 cancel.
            return
        except FatalError as e:
            self.logger.exception("%s execute task %s fatal: %s", self.log_prefix, task, e)
            if not task.done():
                task.fail(e)
            self._stopping_event.set()
            raise
        except Exception as e:
            # 没有到 Fatal Error 级别的都忽视.
            self.logger.exception("%s execute task %s failed: %s", self.log_prefix, task, e)
            if not task.done():
                task.fail(e)
        finally:
            if not task.done():
                self.logger.info("%s failed to ensure task done: %s", self.log_prefix, task)
                task.fail(CommandErrorCode.UNKNOWN_ERROR.error(f"execution failed"))

    async def _fulfill_task_with_its_result_stack(
            self,
            owner: CommandTask,
            stack: CommandTaskStack,
            depth: int = 0,
    ) -> None:
        try:
            self.logger.info(
                "%s Fulfilling task with stack, depth=%s task=%s",
                self.log_prefix, depth, owner,
            )
            # 非阻塞函数不能返回 stack
            if depth > 10:
                raise CommandErrorCode.INVALID_USAGE.error("stackoverflow")
            async for sub_task in stack:
                await asyncio.sleep(0)
                if owner.done():
                    # 不要继续执行了.
                    break
                paths = Channel.split_channel_path_to_names(sub_task.meta.chan)
                if len(paths) > 0:
                    # 发送给子孙了.
                    await self._dispatch_children_task(paths, sub_task)
                    continue
                # 非阻塞
                elif not sub_task.meta.blocking:
                    # 异步执行了.
                    await self._broker.push_task(sub_task)
                    continue

                # 阻塞.
                await self.channel.broker.push_task(sub_task)
                result = await sub_task
                if isinstance(result, CommandTaskStack):
                    # 递归执行
                    await self._fulfill_task_with_its_result_stack(sub_task, result, depth + 1)

            # 完成了所有子节点的调度后, 通知回调函数.
            # !!! 注意: 在这个递归逻辑中, owner 自行决定是否要等待所有的 child task 完成,
            #          如果有异常又是否要取消所有的 child task.
            await stack.callback(owner)
            return
        except Exception as e:
            # 不要留尾巴?
            # 有异常时, 同时取消所有动态生成的 task 对象. 包括发送出去的. 这样就不会有阻塞了.
            if not owner.done():
                self.logger.exception(
                    "%s Fulfill task stack failed, task=%s, exception=%s",
                    self.log_prefix, owner, e,
                )
                for child in stack.generated():
                    if not child.done():
                        child.fail(e)
                owner.fail(e)
            raise e

    async def _run_main_loop(self) -> None:
        """主循环"""
        # 消费输入的命令
        consume_pending_task = asyncio.create_task(self._consume_task_loop())
        closed_task = asyncio.create_task(self._stopping_event.wait())
        try:
            done, pending = await asyncio.wait(
                [consume_pending_task, closed_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await consume_pending_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s Channel main loop failed: %s", self.log_prefix, e)
        finally:
            await self._cleanup()
            self.logger.info("%s channel runtime main loop done", self.log_prefix)
            self._stopped_event.set()

    async def _cleanup(self) -> None:
        try:
            await self.clear()
            if self._broker:
                await self._broker.close()
            self._broker = None
            close_children = []
            for child in self._children_runtimes.values():
                close_children.append(self._loop.create_task(child.stop()))

            done = await asyncio.gather(*close_children, return_exceptions=True)
            for r in done:
                if isinstance(r, Exception):
                    self.logger.exception("%s clean sub runtime failed: %s", self.log_prefix, r)
            self._container = None
            self._channel = None
        except Exception as e:
            self.logger.exception("%s Channel main cleanup exception: %s", self.log_prefix, e)

    async def start(self):
        if self._starting:
            return
        self._starting = True
        self._loop = asyncio.get_event_loop()
        # bootstrap self
        # 确保已经被启动过. 不再递归启动.
        await asyncio.to_thread(self._container.bootstrap)
        self._broker = self._channel.bootstrap(self._container)
        await self._broker.start()

        start_children = []
        for channel in self._channel.children().values():
            child_name = channel.name()
            child_id = channel.id()
            self._children_name_to_ids[child_name] = child_id
            new_runtime = ChannelTreeRuntime(
                path=Channel.join_channel_path(self._path, child_name),
                channel=channel,
                container=self._container,
            )
            start_children.append(self._loop.create_task(new_runtime.start()))
            self._children_name_to_ids[child_name] = child_id
            self._children_runtimes[child_id] = new_runtime

        done = await asyncio.gather(*start_children, return_exceptions=True)
        for r in done:
            if isinstance(r, Exception):
                self.logger.exception("%s channel start sub runtime failed: %s", self.log_prefix, r)
        self._started = True
        self._main_loop_task = self._loop.create_task(self._run_main_loop())

    async def stop(self):
        if self._stopping_event.is_set():
            return
        self._stopping_event.set()
        if self._main_loop_task and not self._main_loop_task.done():
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        self._main_loop_task = None
        await self._stopping_event.wait()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._logger is not None:
            self._logger.exception("%s Channel exit with exception: %s", self.log_prefix, exc_val)
        await self.stop()
