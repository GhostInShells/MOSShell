import contextvars
import logging
from abc import ABC, abstractmethod

from typing import Dict, Optional, Iterable, Any
from typing_extensions import Self
from ghoshell_moss.concepts.channel import Channel
from ghoshell_moss.concepts.command import CommandTaskStack, CommandTask, Command
from ghoshell_moss.concepts.errors import FatalError, CommandError, InterpretError
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, ensure_tasks_done_or_cancel, TreeNotify
from ghoshell_container import IoCContainer
import asyncio


class ChannelRuntime(ABC):
    """
    管理 channel 的所有的 command task 运行时状态, 包括阻塞, 执行, 等待.
    """
    channel: Channel
    name: str

    @abstractmethod
    def append(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时的队列中.
        """
        pass

    @abstractmethod
    def commands(self, recursive: bool = True, available_only: bool = True) -> Iterable[Command]:
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空所有的运行任务和运行中的任务.
        递归清空.
        """
        pass

    @abstractmethod
    async def defer_clear(self) -> None:
        """
        设置 channel 为软重启. 当有一个属于当前 channel runtime 的 task 推送进来时, 清空自身和所有子节点.
        """
        pass

    @abstractmethod
    async def clear_pending(self) -> int:
        """
        清空自身和子节点队列中未执行的任务.
        """
        pass

    @abstractmethod
    async def cancel_executing(self) -> None:
        """
        取消正在运行的所有任务, 包括自身正在运行的任务, 和所有子节点的任务.
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        runtime 是否在运行中.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """
        是否正在运行任务, 或者队列中存在任务.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        运行直到结束.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        停止 runtime 运行.
        """
        pass


class ChannelRuntimeImpl(ChannelRuntime):
    """
    Channel 运行时的状态管理. 一个核心的技术思路是, channel runtime 自身不递归.
    """

    def __init__(
            self,
            container: IoCContainer,
            channel: Channel,
            *,
            is_idle_notifier: TreeNotify | None = None,
            logger: Optional[logging.Logger] = None,
            stop_event: Optional[ThreadSafeEvent] = None,
            depth: int = 0
    ):
        # 容器应该要已经运行过了. 关键的抽象也被设置过.
        # channel runtime 不需要有自己的容器. 也不需要关闭它.
        self.container = container
        self.logger = logger or logging.getLogger("moss")
        self.channel: Channel = channel
        self.name = channel.name()
        self.depth: int = depth
        self.loop: asyncio.AbstractEventLoop | None = None
        self.is_idle_notifier = is_idle_notifier or TreeNotify(self.name)
        # runtime 级别的关机事件. 会传递给所有的子节点.
        self._stop_event = stop_event or ThreadSafeEvent()

        self.children_runtimes: Dict[str, ChannelRuntime] = {}

        # status
        self._started = False
        self._stopped = False

        # 获取被启动时的 loop, 用来做跨线程的调度.
        self._running_event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 输入队列, 只是为了足够快地输入. 当执行 cancel 的时候, executing_queue 会被清空, 但 pending queue 不会被清空.
        # 这种队列是为了 call_soon 的特殊 feature 做准备, 同时又不会在执行时阻塞解析. 解析的速度要求是全并行的.
        self._pending_queue: asyncio.Queue[CommandTask | None] = asyncio.Queue()

        # 消费队列. 如果队列里的数据是 None, 表示这个队列被丢弃了.
        self._executing_queue: asyncio.Queue[CommandTask | None] = asyncio.Queue()
        self._executing_block_task: bool = False

        # main loop
        self._main_loop_task: Optional[asyncio.Task] = None

        # 是否是 defer clear 状态.
        # 用 flag 做标记, 因为一旦触发了 clear, 就会递归 clear.
        self._defer_clear: bool = False

        # 运行中的 task group, 方便整体 cancel. 由于版本控制在 3.10, 暂时无法使用 asyncio 的 TaskGroup.
        self._executing_task_group: set = set()
        self._executing_block_task: bool = False

    # --- lifecycle --- #

    async def start(self):
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        self._running_event_loop = loop
        # 自身的启动.
        # 最后才启动主循环.
        try:
            start_runtimes = [self._self_bootstrap()]

            for channel in self.channel.children().values():
                name = channel.name()
                if name not in self.children_runtimes:
                    runtime = self.make_child_runtime(channel)
                    start_runtimes.append(runtime.start())
                    self.children_runtimes[name] = runtime

            # 启动所有已知的 children 节点.
            await asyncio.gather(*start_runtimes)
        except Exception as e:
            raise FatalError(f"Failed to start channel {self.name}") from e

    async def _self_bootstrap(self):
        # 创建主任务.
        if not self.channel.is_running():
            # 启动自身的 channel. 不过这样是效率比较低, 最好提前都启动完了.
            client = self.channel.bootstrap(self.container)
            await client.start()
        self._main_loop_task = asyncio.create_task(self._run_main_loop())

    async def close(self):
        # 已经结束过了.
        if not self._started or self._stopped:
            return
        self._stopped = True
        if not self._stop_event.is_set():
            self._stop_event.set()
        await self._self_close()
        # 自身完成后, 再关闭所有的子节点.
        closing_children = []
        for runtime in self.children_runtimes.values():
            closing_children.append(runtime.close())
        await ensure_tasks_done_or_cancel(*closing_children)

    async def _self_close(self) -> None:
        # 等待自身的主循环结束. 同时关闭对 channel client 的调用.
        if not self._main_loop_task.done():
            self._main_loop_task.cancel()
        try:
            await self._main_loop_task
        except asyncio.CancelledError:
            pass
        if self.channel.is_running():
            await self.channel.client.close()

    def _check_running(self):
        if not self._started:
            raise RuntimeError(f"Channel `{self.name}` is not running")
        elif self._stop_event.is_set():
            raise RuntimeError(f"Channel `{self.name}` is shutdown")

    def is_running(self) -> bool:
        """
        判断 runtime 是否在运行.
        """
        return self._started and not self._stop_event.is_set() and self.channel.is_running()

    def is_available(self) -> bool:
        return self.is_running() and self.channel.is_running() and self.channel.client.is_available()

    def commands(self, recursive: bool = True, available_only: bool = True) -> Iterable[Command]:
        self._check_running()
        if not self.is_available():
            return []
        yield from self.channel.client.commands(available_only).values()
        if recursive:
            for child_runtime in self.children_runtimes.values():
                yield from child_runtime.commands(recursive, available_only)

    def is_busy(self) -> bool:
        """
        判断 runtime 是否是 busy 状态. 任何子节点在运行, 都会是 busy 状态.
        """
        if not self.is_running():
            return False
        return not self.is_idle_notifier.is_set()

    def make_child_runtime(self, channel: Channel) -> ChannelRuntime:
        child_runtime = ChannelRuntimeImpl(
            self.container,
            channel,
            logger=self.logger,
            stop_event=self._stop_event,
            is_idle_notifier=self.is_idle_notifier.child(channel.name()),
            depth=self.depth + 1,
        )
        return child_runtime

    async def get_or_create_child_runtime(self, channel: Channel) -> Self:
        if channel is self.channel:
            raise ValueError(f"Channel {self.name} register child but is itself")
        name = channel.name()
        if name in self.children_runtimes:
            runtime = self.children_runtimes[name]
            if runtime.channel is channel:
                # 直接返回.
                return runtime
            else:
                # 清空掉之前的.
                await runtime.close()

        if name not in self.channel.children():
            # 动态注册.
            self.channel.include_channels(channel)

        child_runtime = self.make_child_runtime(channel)
        if not child_runtime.is_running():
            await child_runtime.start()
        self.children_runtimes[child_runtime.name] = child_runtime
        return child_runtime

    async def get_chan_runtime(self, name: str, child_only: bool = False) -> Optional[Self]:
        try:
            # 检查是否已经启动了.
            self._check_running()
            if name == self.name:
                # 返回自身.
                return self
            children_channels = self.channel.children()
            if name in children_channels:
                # 从缓存里拿
                child_channel = children_channels[name]
                # 需要判断是不是最新的.
                runtime = self.get_or_create_child_runtime(child_channel)
                return runtime
            elif child_only:
                return None
            else:
                if name in children_channels:
                    # 返回一级 channel.
                    child_channel = children_channels[name]
                    runtime = await self.get_or_create_child_runtime(child_channel)
                    return runtime

                # 查看是否在血缘中.
                for child_channel in children_channels.values():
                    if name in child_channel.descendants():
                        runtime = await self.get_or_create_child_runtime(child_channel)
                        # 返回子孙 channel.
                        return runtime.get_chan_runtime(name)
                    else:
                        return None

        except Exception as e:
            self.logger.exception(e)
            raise FatalError("Failed to get child client") from e

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        return await asyncio.wait_for(self.is_idle_notifier.wait(), timeout)

    # --- append & pending --- #
    def append(self, *tasks: CommandTask) -> None:
        if not self.is_running():
            raise InterpretError(f"Channel {self.name} is not running")
        task_list = list(tasks)
        if len(task_list) == 0:
            return

        try:
            _queue = self._pending_queue
            for _task in tasks:
                if _task is None:
                    continue
                # 快速入队.
                elif _task.done():
                    # 丢弃掉已经被取消的任务.
                    # todo: log
                    continue
                _task.set_state('pending')
                self._running_event_loop.call_soon_threadsafe(_queue.put_nowait, _task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)

    async def clear_pending(self) -> None:
        """无锁的清空实现. """
        self._check_running()
        try:
            # 先清空自身的队列.
            # 同步阻塞清空.
            _pending_queue = self._pending_queue
            self._pending_queue = asyncio.Queue()
            while not _pending_queue.empty():
                task = await _pending_queue.get()
                if task and not task.done():
                    task.cancel("clear pending")
            _pending_queue.put_nowait(None)
            # 送入毒丸, 避免死锁.
            # 然后清空所有子节点的 pending 队列.
            clear_children_pending = []
            for child in self.children_runtimes.values():
                clear_children_pending.append(child.clear_pending())
            # 清空子节点.
            await asyncio.gather(*clear_children_pending)

        except Exception as exc:
            self.logger.exception(exc)
            # 所有没有管理的异常, 都是致命异常.
            self._stop_event.set()
            raise exc

    async def _consume_pending_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                _pending_queue = self._pending_queue
                item = await _pending_queue.get()
                if item is None:
                    continue
                await self._add_executing_task(item)
        except asyncio.CancelledError as e:
            self.logger.info("Cancelling pending task: %r", e)

        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()
        finally:
            self.logger.info('Finished executing loop')

    # --- executing loop --- #

    async def _add_executing_task(self, task: CommandTask) -> None:
        # 推送到等待队列中.
        # 需要在添加命令时就执行
        if task is None:
            return
        elif task.done():
            # todo: log
            # 丢弃掉完成的 task.
            return

        if self._defer_clear:
            try:
                await self.cancel_executing()
            finally:
                self._defer_clear = False

        try:
            # call soon
            if task.meta.call_soon:
                # 清空队列先.
                block = task.meta.block
                if block:
                    # 先清空.
                    await self.cancel_executing()
                    # 丢入执行队列中.
                    self._executing_queue.put_nowait(task)
                    return
                else:
                    # 立刻执行, 实际上会生成一个 none block 的 task.
                    # 虽然是 none-block 的, 但也会被 cancel executing 取消掉.
                    await self._execute_task(task)
                    return
            else:
                # 丢到阻塞队列里.
                self._executing_queue.put_nowait(task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()

    async def cancel_executing(self) -> None:
        self._check_running()
        try:
            # 准备并发 cancel 所有的运行.
            cancel_running = [self._cancel_self_executing()]
            for child in self.children_runtimes.values():
                # 子节点则是直接 clear. 对于父节点而言, 子节点的 pending 也是自己的 executing.
                cancel_running.append(child.clear())
            await asyncio.gather(*cancel_running)
        except asyncio.CancelledError:
            self.logger.error("channel %s cancel running but canceled", self.name)
        except Exception as exc:
            self.logger.exception(exc)
            self._stop_event.set()
            raise FatalError("channel %s cancel executing failed" % self.name) from exc

    async def _cancel_self_executing(self) -> None:
        """取消掉正在运行中的 task. """
        old_queue = self._executing_queue
        # 创建新队列.
        self._executing_queue = asyncio.Queue()
        # 取消掉所有未执行任务.
        while not old_queue.empty():
            task = await old_queue.get()
            if not task.done():
                task.cancel()

        # 发送毒丸.
        await old_queue.put(None)
        # 清除所有运行中的任务. 同步阻塞, 所以不用考虑锁的问题.
        if len(self._executing_task_group) > 0:
            for t in self._executing_task_group:
                t.cancel()
            self._executing_task_group.clear()

    async def _executing_loop(self) -> None:
        """主消费队列."""
        try:
            # 判断 policy 协议是否已经触发了.
            policy_is_running = False

            while not self._stop_event.is_set():
                try:
                    # 每次重新去获取 queue. 由于 queue 可能被丢弃, 所以一定要一次只执行一步.
                    _queue = self._executing_queue
                    item = None
                    if not _queue.empty() or self.is_idle_notifier.is_self_set():
                        # 用短时间来尝试获取.
                        item = await _queue.get()

                    if item is None:
                        if not policy_is_running:
                            # 启动 policy.
                            await self._start_self_policy()
                            policy_is_running = True
                        elif not self.is_idle_notifier.is_self_set():
                            # 设置已经闲了.
                            self.is_idle_notifier.set()
                        # 进入下个循环.
                        continue

                    # 有任务在执行, 怎么都 clear 一下.
                    self.is_idle_notifier.clear()
                    if policy_is_running:
                        # 阻塞等待 policy 停止运行.
                        await self._pause_self_policy()
                        policy_is_running = False

                    # 获取最早的一个任务.
                    # 运行一个任务. 理论上是很快的调度.
                    # 这个任务不运行结束, 不会释放运行状态. 它如果是同步阻塞的, 则阻塞后续的 task 消费.
                    await self._execute_task(item)
                except asyncio.CancelledError as e:
                    self.logger.error(f"channel {self.name} loop got cancelled: %s", e)
                except Exception as e:
                    self.logger.exception(e)
        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()

    async def _pause_self_policy(self) -> None:
        try:
            if not self.is_available():
                return
            await self.channel.client.policy_pause()
        except FatalError as e:
            self.logger.exception(e)
            self._stop_event.set()
            raise
        except Exception as e:
            self.logger.exception(e)

    async def _start_self_policy(self) -> None:
        try:
            if not self.is_available():
                return
            # 启动 policy.
            await self.channel.client.policy_run()
        except FatalError as e:
            self.logger.exception(e)
            self._stop_event.set()
            raise
        except Exception as e:
            self.logger.exception(e)

    async def _execute_task(self, cmd_task: CommandTask) -> None:
        """执行一个 task. 核心目标是最快速度完成调度逻辑, 或者按需阻塞链路.  """
        try:
            sent = await self._send_task_to_child_if_not_own(cmd_task)
            if sent:
                # 不是自己的任务, 快速分发.
                return
            block = cmd_task.meta.block
            if block:
                await self._execute_self_channel_task_within_group(cmd_task)
            else:
                # 非阻塞的 task, 异步执行. 但仍然可以统一 cancel.
                _ = asyncio.create_task(self._execute_self_channel_task_within_group(cmd_task))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # 不应该抛出任何异常.
            self.logger.exception(e)
            self._stop_event.set()

    async def _execute_self_channel_task_within_group(self, cmd_task: CommandTask) -> None:
        """运行属于自己这个 channel 的 task"""
        # 运行一个任务. 理论上是很快的调度.
        # 这个任务不运行结束, 不会释放运行状态.
        asyncio_task = asyncio.create_task(self._ensure_resolve_self_channel_task(cmd_task))
        try:
            # 通过 group 方便统一取消.
            self._executing_task_group.add(asyncio_task)
            wait_stop = asyncio.create_task(self._stop_event.wait())
            # 永远和 stop 做比较. 避免无法停止.
            done, pending = await asyncio.wait(
                [asyncio_task, wait_stop],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if asyncio_task not in done:
                asyncio_task.cancel()
            return await asyncio_task

        except asyncio.CancelledError:
            # 无所谓, 继续.
            return
        except Exception as e:
            # 没有到 Fatal Error 级别的都忽视.
            self.logger.exception(e)
        finally:
            if asyncio_task and asyncio_task in self._executing_task_group:
                self._executing_task_group.remove(asyncio_task)
            if not cmd_task.done():
                cmd_task.cancel()

    async def _ensure_resolve_self_channel_task(self, task: CommandTask) -> None:
        """在一个栈中运行 task. 要确保 task 的最终状态一定被更新了, 不是空. """
        try:
            # 真的轮到自己执行它了.
            task.set_state("running")
            # 先执行一次 command, 拿到可能的 command_seq, 主要用来做 resolve.
            result = await self._run_self_channel_task_with_context(task)
            if not isinstance(result, CommandTaskStack):
                # 返回一个栈, command task 的结果需要在栈外判断.
                # 等栈运行完了才会赋值.
                task.resolve(result)
                return result

            # 这里才真正赋值
            # 执行特殊的 stack 逻辑.
            await self._fulfill_task_with_its_result_stack(task, result)

        except asyncio.CancelledError as e:
            self.logger.info("execute command `%r` is cancelled: %s", task, e)
            task.cancel()
            # 冒泡.
            raise
        except FatalError as e:
            self.logger.exception(e)
            self._stop_event.set()
            raise
        except CommandError as e:
            self.logger.info("execute command `%r`error: %s", task, e)
            task.fail(e)
        except Exception as e:
            self.logger.exception(e)
            task.fail(e)
        finally:
            # 不要留尾巴?
            if not task.done():
                task.cancel()

    async def _fulfill_task_with_its_result_stack(
            self,
            owner: CommandTask,
            stack: CommandTaskStack,
            depth: int = 0,
    ) -> None:
        try:
            # 非阻塞函数不能返回 stack
            if not owner.meta.block:
                # todo: 这个是不是 fatal 的问题呢? 应该不是.
                raise CommandError(
                    CommandError.INVALID_USAGE,
                    f"none-block command {owner} returned a command stack which is not allowed",
                )
            elif depth > 5:
                raise CommandError(CommandError.INVALID_USAGE, "stackoverflow")

            async for sub_task in stack:
                if owner.done():
                    # 不要继续执行了.
                    break
                sent = await self._send_task_to_child_if_not_own(sub_task)
                if sent:
                    # 发送给子孙了.
                    continue

                # 非阻塞
                if not sub_task.meta.block:
                    # 异步执行了.
                    _ = asyncio.create_task(self._execute_self_channel_task_within_group(sub_task))
                    continue

                # 阻塞.
                result = await self._run_self_channel_task_with_context(sub_task)
                if isinstance(result, CommandTaskStack):
                    # 递归执行
                    await self._fulfill_task_with_its_result_stack(sub_task, result, depth + 1)
                else:
                    sub_task.resolve(result)

            # 完成了所有子节点的调度后, 通知回调函数.
            # !!! 注意, 在这个递归逻辑中, owner 自行决定是否要等待所有的 child task 完成, 如果有异常又是否要取消所有的 child task.
            await stack.success(owner)
            return
        except Exception:
            # 不要留尾巴?
            # 有异常时, 同时取消所有动态生成的 task 对象. 包括发送出去的. 这样就不会有阻塞了.
            for child in stack.generated():
                if not child.done():
                    child.cancel()
            raise
        finally:
            if not owner.done():
                owner.cancel()

    async def _send_task_to_child_if_not_own(self, cmd_task: CommandTask) -> bool:
        """判断一个 task 是不是自己的."""
        execution_channel_name = cmd_task.meta.chan
        if execution_channel_name == self.name:
            # 就是自己, 不用发送.
            return False
        elif execution_channel_name == "":
            # 主轨命令都是自身执行, 不管谁调度来的.
            return False

        children_channels = self.channel.children()
        if execution_channel_name in children_channels:
            child_channel = children_channels[execution_channel_name]
            child_runtime = await self.get_or_create_child_runtime(child_channel)
            child_runtime.append(cmd_task)
            return True

        for child_channel in children_channels.values():
            # 判断是不是在一个 child channel 的子孙节点里.
            if execution_channel_name in child_channel.descendants():
                # 如果是, 命令也只发送给子节点.
                child_runtime = await self.get_or_create_child_runtime(child_channel)
                child_runtime.append(cmd_task)
                return True

        # 丢弃掉不认识的.
        self.logger.error(f"channel {self.name} abundant orphan task {cmd_task}")
        # 记住要取消掉.
        cmd_task.cancel("channel %s not found" % execution_channel_name)
        return True

    async def _run_self_channel_task_with_context(self, cmd_task: CommandTask) -> Any:
        """真正运行一个 command task 了. """
        if cmd_task.done():
            cmd_task.raise_exception()
            return cmd_task.result()

        cmd_task.exec_chan = self.name
        # 准备好 ctx. 包含 channel 的容器, 还有 command task 的 context 数据.
        ctx = contextvars.copy_context()
        self.channel.set_context_var()
        ctx_ran_cor = ctx.run(cmd_task.dry_run)

        # 创建一个可以被 cancel 的 task.
        run_execution = asyncio.create_task(ctx_ran_cor)
        # 这个 task 是不是在运行出结果之前, 外部已经结束了.
        wait_outside_done = asyncio.create_task(cmd_task.wait())

        done, pending = await asyncio.wait(
            [run_execution, wait_outside_done],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if run_execution not in done and not run_execution.done():
            # 如果 run task 不在 Done 里, 说明 cmd task 在外部被先结束了.
            run_execution.cancel()

        return await run_execution

    # --- main loop --- #

    async def _run_main_loop(self) -> None:
        """主循环"""
        # 消费输入的命令
        consume_pending_task = asyncio.create_task(self._consume_pending_loop())
        # 消费确认可执行的命令.
        executing_task = asyncio.create_task(self._executing_loop())

        try:
            gathered = asyncio.gather(consume_pending_task, executing_task)
            stopped = asyncio.create_task(self._stop_event.wait())
            done, pending = await asyncio.wait([gathered, stopped], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            # 如果遇到问题就直接取消.
            await gathered
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.logger.info(f"channel {self.name} main loop done")

    async def clear(self) -> None:
        self._check_running()
        try:
            # 暂停所有的消费动作. 锁了自己, 也就锁了子节点.
            # 先清空队列. 递归地清空.
            await self.clear_pending()
            # 然后清空运行中的任务.
            await self.cancel_executing()
            # 通知自己所有的 channel 清空.
            await self._call_self_clear_callback()

        except asyncio.CancelledError:
            self.logger.info("channel %s clearing is cancelled", self.name)
            raise
        except FatalError as e:
            self.logger.exception(e)
            self._stop_event.set()
            raise
        except Exception as exc:
            self.logger.exception(exc)
            raise

    async def _call_self_clear_callback(self) -> None:
        """
        回调所有的 channel 已经执行了 clear.
        """
        try:
            if self.is_available():
                await self.channel.client.clear()
        except asyncio.CancelledError:
            self.logger.info(f"channel {self.name} clearing is cancelled")
        except Exception as exc:
            self.logger.exception(exc)

    async def defer_clear(self) -> None:
        """
        准备清空运行状态, 如果有指令输入的话.
        """
        await self.clear_pending()
        # defer clear 不需要递归. 因为所有子节点的任务来自父节点.
        self._defer_clear = True

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return None
