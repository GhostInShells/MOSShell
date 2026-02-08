import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelMeta
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandTaskStack
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode, FatalError
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

ChannelPath = list[str]
DispatchTaskCallback = Callable[[Channel, ChannelPath, CommandTask], Coroutine[None, None, None]]


class ChannelRuntime:
    """
    Channel 运行时的状态管理. 一个核心的技术思路是, channel runtime 自身不递归.
    """

    def __init__(
        self,
        container: IoCContainer,
        channel: Channel,
        dispatch_task_callback: DispatchTaskCallback,
        *,
        stop_event: Optional[ThreadSafeEvent] = None,
    ):
        # 容器应该要已经运行过了. 关键的抽象也被设置过.
        # channel runtime 不需要有自己的容器. 也不需要关闭它.
        self.container = container
        self.channel: Channel = channel
        self.name = channel.name()
        self._dispatch_task_callback = dispatch_task_callback
        self.loop: asyncio.AbstractEventLoop | None = None
        # runtime 级别的关机事件. 会传递给所有的子节点.
        self._stop_event = stop_event or ThreadSafeEvent()
        # status
        self._started = False
        self._stopped = False
        self._logger = None

        # 获取被启动时的 loop, 用来做跨线程的调度.
        self._running_event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 输入队列, 只是为了足够快地输入. 当执行 cancel 的时候, executing_queue 会被清空, 但 pending queue 不会被清空.
        # 这种队列是为了 call_soon 的特殊 feature 做准备, 同时又不会在执行时阻塞解析. 解析的速度要求是全并行的.
        self._pending_queue: asyncio.Queue[tuple[ChannelPath, CommandTask] | None] = asyncio.Queue()
        self._is_idle_event = asyncio.Event()
        self._is_idle_event.set()

        # 消费队列. 如果队列里的数据是 None, 表示这个队列被丢弃了.
        self._executing_queue: asyncio.Queue[tuple[ChannelPath, CommandTask] | None] = asyncio.Queue()
        self._executing_block_task: bool = False

        # main loop
        self._main_loop_task: Optional[asyncio.Task] = None

        # 是否是 defer clear 状态.
        # 用 flag 做标记, 因为一旦触发了 clear, 就会递归 clear.
        self._defer_clear: bool = False

        # 运行中的 task group, 方便整体 cancel. 由于版本控制在 3.10, 暂时无法使用 asyncio 的 TaskGroup.
        self._executing_task_group: set = set()
        self._executing_block_task: bool = False

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            logger = self.container.get(LoggerItf)
            if logger is None:
                logger = logging.getLogger("moss")
                self.container.set(LoggerItf, logger)
            self._logger = logger
        return self._logger

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
            await self._self_bootstrap()
        except Exception as e:
            raise FatalError(f"Failed to start channel {self.name}") from e

    async def _self_bootstrap(self):
        # 创建主任务.
        if not self.channel.is_running():
            # 启动自身的 channel. 不过这样是效率比较低, 最好提前都启动完了.
            broker = self.channel.bootstrap(self.container)
            await broker.start()
        self._main_loop_task = asyncio.create_task(self._run_main_loop())

    async def close(self):
        # 已经结束过了.
        if not self._started or self._stopped:
            return
        self._stopped = True
        if not self._stop_event.is_set():
            self._stop_event.set()
        await self._self_close()

    async def _self_close(self) -> None:
        # 等待自身的主循环结束. 同时关闭对 channel client 的调用.
        if not self._main_loop_task.done():
            self._main_loop_task.cancel()
        try:
            await self._main_loop_task
        except asyncio.CancelledError:
            pass
        if self.channel.is_running():
            await self.channel.broker.close()

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
        return self.is_running() and self.channel.broker.is_connected() and self.channel.broker.is_available()

    def commands(self, available_only: bool = True) -> dict[str, Command]:
        self._check_running()
        if not self.is_available():
            return {}
        return self.channel.broker.commands(available_only)

    def channel_meta(self) -> ChannelMeta:
        self._check_running()
        # 保持更新. 返回值自我应该复制, 保证不污染.
        return self.channel.broker.meta()

    def is_busy(self) -> bool:
        """
        判断 runtime 是否是 busy 状态. 任何子节点在运行, 都会是 busy 状态.
        """
        if not self.is_running():
            return False
        return not self._is_idle_event.is_set()

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._is_idle_event.wait(), timeout)

    # --- append & pending --- #

    def add_task(self, task: CommandTask) -> None:
        if task is None:
            return
        chan = task.meta.chan
        if chan in {"", self.name}:
            self.add_task_with_paths([], task)
        else:
            paths = Channel.split_channel_path_to_names(chan)
            self.add_task_with_paths(paths, task)

    def add_task_with_paths(self, channel_path: list[str], task: CommandTask) -> None:
        if not self.is_running():
            self.logger.error("Channel `%s` is not running, receiving task %s", self.name, task)
            return

        try:
            _queue = self._pending_queue
            task.set_state("pending")
            # 记录发送路径.
            task.send_through.append(self.name)
            _queue.put_nowait((channel_path, task))
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("Add task failed")

    async def clear_pending(self) -> None:
        """无锁的清空实现."""
        self._check_running()
        try:
            # 先清空自身的队列.
            # 同步阻塞清空.
            _pending_queue = self._pending_queue
            self._pending_queue = asyncio.Queue()
            while not _pending_queue.empty():
                path, task = await _pending_queue.get()
                if task and not task.done():
                    task.cancel("clear pending")
            _pending_queue.put_nowait(None)
            # 送入毒丸, 避免死锁.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception("Clear pending failed")
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
                paths, task = item
                await self._add_executing_task(paths, task)
        except asyncio.CancelledError as e:
            self.logger.info("Cancelling pending task: %r", e)

        except Exception:
            self.logger.exception("Consume pending loop failed")
            self._stop_event.set()
        finally:
            self.logger.info("Finished executing loop")

    # --- executing loop --- #

    @classmethod
    def is_self_path(cls, path: ChannelPath) -> bool:
        return len(path) == 0

    async def _add_executing_task(self, path: ChannelPath, task: CommandTask) -> None:
        # 推送到等待队列中.
        # 需要在添加命令时就执行
        if task is None:
            return
        elif task.done():
            self.logger.error("received executing task `%s` already done", task)
            return

        if self._defer_clear:
            try:
                await self.cancel_executing()
            finally:
                self._defer_clear = False

        try:
            # call soon
            if self.is_self_path(path) and task.meta.call_soon:
                # 清空队列先.
                block = task.meta.block
                if block:
                    # 先清空.
                    await self.cancel_executing()
                    # 丢入执行队列中.
                    self._executing_queue.put_nowait((path, task))
                    return
                else:
                    # 立刻执行, 实际上会生成一个 none block 的 task.
                    # 虽然是 none-block 的, 但也会被 cancel executing 取消掉.
                    await self._execute_task(task)
                    return
            else:
                # 丢到阻塞队列里.
                self._executing_queue.put_nowait((path, task))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Add executing task failed")
            self._stop_event.set()

    async def cancel_executing(self) -> None:
        self._check_running()
        try:
            # 准备并发 cancel 所有的运行.
            await self._cancel_self_executing()
        except asyncio.CancelledError:
            self.logger.exception("channel %s cancel running but canceled", self.name)
            raise
        except Exception as exc:
            # 理论上不会有异常抛出来.
            self.logger.exception("Cancel executing failed")
            self._stop_event.set()
            raise FatalError(f"channel {self.name} cancel executing failed") from exc

    async def _cancel_self_executing(self) -> None:
        """取消掉正在运行中的 task."""
        old_queue = self._executing_queue
        # 创建新队列.
        self._executing_queue = asyncio.Queue()
        # 发送毒丸.
        await old_queue.put(None)
        # 取消掉所有未执行任务.
        while not old_queue.empty():
            item = await old_queue.get()
            if item is None:
                continue
            paths, task = item
            if not task.done():
                task.cancel("cancel executing")

        # 清除所有运行中的任务. 同步阻塞, 所以不用考虑锁的问题.
        if len(self._executing_task_group) > 0:
            canceling = self._executing_task_group.copy()
            self._executing_task_group.clear()
            for t in canceling:
                t.cancel("cancel executing")
            # 等待所有的任务结束.
            await asyncio.gather(*canceling, return_exceptions=True)

    async def _executing_loop(self) -> None:
        """主消费队列."""
        try:
            # 判断 policy 协议是否已经触发了.
            policy_is_running = False

            while not self._stop_event.is_set():
                # 每次重新去获取 queue. 由于 queue 可能被丢弃, 所以一定要一次只执行一步.
                # 循环里的每次查找发生时, 都一定是没有阻塞任务在执行中的.
                _queue = self._executing_queue
                # 当队列不为空的时候, 或者已经完成了 policy 与 idle 设置的时候.
                if not _queue.empty() or (policy_is_running and self._is_idle_event.is_set()):
                    # 尽快消费队列.
                    item = await _queue.get()
                    if item is None:
                        # 拿到了毒丸.
                        continue

                    # 拿到了真实的任务.
                    paths, task = item
                    # 不是自己的任务, 就要分发给孩子们.
                    # 自己状态不变更.
                    if not self.is_self_path(paths):
                        await self._dispatch_child_task(paths, task)
                        continue

                    # 先取消 idle 状态.
                    self._is_idle_event.clear()

                    # 如果是自己的任务, 则不要立刻执行, 先关闭 policy.
                    if policy_is_running:
                        try:
                            await self._pause_self_policy()
                        finally:
                            policy_is_running = False
                    # 然后开始执行, 并且等待 (如果要等待的话)
                    await self._execute_task(task)
                    continue
                else:
                    # 这种情况下, 可知队列为空. 没有新的任务进入进来.
                    if not policy_is_running:
                        # 启动 policy.
                        try:
                            await self._start_self_policy()
                        finally:
                            policy_is_running = True
                        continue

                    if not self._is_idle_event.is_set():
                        self._is_idle_event.set()
                        continue

        except asyncio.CancelledError as e:
            self.logger.info("channel `%s` loop got cancelled: %s", self.name, e)
        except Exception:
            self.logger.exception("Executing loop failed")
            self._stop_event.set()

    async def _pause_self_policy(self) -> None:
        try:
            if not self.is_available():
                return
            await self.channel.broker.policy_pause()
        except asyncio.CancelledError:
            pass
        except FatalError:
            self.logger.exception("Pause policy failed with fatal error")
            self._stop_event.set()
            raise
        except Exception:
            self.logger.exception("Pause policy failed")

    async def _start_self_policy(self) -> None:
        try:
            if not self.is_available():
                return
            # 启动 policy.
            await self.channel.broker.policy_run()
        except asyncio.CancelledError:
            pass
        except FatalError:
            self.logger.exception("Start policy failed with fatal error")
            self._stop_event.set()
            raise
        except Exception:
            self.logger.exception("Start policy failed")

    async def _dispatch_child_task(self, path: ChannelPath, task: CommandTask) -> None:
        if len(path) == 0:
            self.logger.error("failed to dispatch child task with empty paths")
            return
        child_name = path.pop(0)
        children = self.channel.children()
        if child_name not in children:
            task.cancel("the channel not found")
            self.logger.error(
                "receive task from channel `%s` which not found at %s",
                task.meta.chan,
                self.name,
            )
            return
        child = children[child_name]
        await self._dispatch_task_callback(child, path, task)

    async def _execute_task(self, cmd_task: CommandTask) -> None:
        """执行一个 task. 核心目标是最快速度完成调度逻辑, 或者按需阻塞链路."""
        try:
            block = cmd_task.meta.block
            if block:
                await self._execute_self_channel_task_within_group(cmd_task)
            else:
                # 非阻塞的 task, 异步执行. 但仍然可以统一 cancel.
                _ = asyncio.create_task(self._execute_self_channel_task_within_group(cmd_task))
        except asyncio.CancelledError:
            raise
        except Exception:
            # 不应该抛出任何异常.
            self.logger.exception("Execute task failed")
            self._stop_event.set()

    async def _execute_self_channel_task_within_group(self, cmd_task: CommandTask) -> None:
        """运行属于自己这个 channel 的 task, 让它进入到 executing group 中."""
        # 运行一个任务. 理论上是很快的调度.
        # 这个任务不运行结束, 不会释放运行状态.
        asyncio_task = asyncio.create_task(self._ensure_self_task_done(cmd_task))
        try:
            # 通过 group 方便统一取消.
            self._executing_task_group.add(asyncio_task)
            wait_stop = asyncio.create_task(self._stop_event.wait())
            # 永远和 stop 做比较. 避免无法停止.
            done, pending = await asyncio.wait(
                [asyncio_task, wait_stop],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            if asyncio_task not in done:
                asyncio_task.cancel()
            return await asyncio_task

        except asyncio.CancelledError:
            # 无所谓, 继续.
            return
        except FatalError:
            raise
        except Exception:
            # 没有到 Fatal Error 级别的都忽视.
            self.logger.exception("Execute task loop failed")
        finally:
            if asyncio_task and asyncio_task in self._executing_task_group:
                self._executing_task_group.remove(asyncio_task)
            if not cmd_task.done():
                cmd_task.cancel()

    async def _ensure_self_task_done(self, task: CommandTask) -> None:
        """在一个栈中运行 task. 要确保 task 的最终状态一定被更新了, 不是空."""
        try:
            # 真的轮到自己执行它了.
            task.set_state("running")
            # 先执行一次 command, 拿到可能的 command_seq, 主要用来做 resolve.
            result = await self.channel.execute_task(task)
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
            task.cancel(f"cancelled: {e}")
            # 冒泡.
            raise
        except FatalError as e:
            self.logger.exception("Execute task failed with fatal error")
            self._stop_event.set()
            task.fail(e)
            raise
        except CommandError as e:
            self.logger.info("execute command `%r`error: %s", task, e)
            task.fail(e)
        except Exception as e:
            self.logger.exception("Execute task failed")
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
                raise CommandErrorCode.INVALID_USAGE.error(
                    f"none-block command {owner} returned a command stack which is not allowed",
                )
            elif depth > 5:
                raise CommandErrorCode.INVALID_USAGE.error("stackoverflow")

            async for sub_task in stack:
                if owner.done():
                    # 不要继续执行了.
                    break
                paths = Channel.split_channel_path_to_names(sub_task.meta.chan)
                if not self.is_self_path(paths):
                    # 发送给子孙了.
                    await self._dispatch_child_task(paths, sub_task)
                    continue

                # 非阻塞
                if not sub_task.meta.block:
                    # 异步执行了.
                    _ = asyncio.create_task(self._execute_self_channel_task_within_group(sub_task))
                    continue

                # 阻塞.
                result = await self.channel.execute_task(sub_task)
                if isinstance(result, CommandTaskStack):
                    # 递归执行
                    await self._fulfill_task_with_its_result_stack(sub_task, result, depth + 1)
                else:
                    sub_task.resolve(result)

            # 完成了所有子节点的调度后, 通知回调函数.
            # !!! 注意: 在这个递归逻辑中, owner 自行决定是否要等待所有的 child task 完成,
            #          如果有异常又是否要取消所有的 child task.
            await stack.success(owner)
            return
        except FatalError:
            raise
        except Exception as e:
            # 不要留尾巴?
            # 有异常时, 同时取消所有动态生成的 task 对象. 包括发送出去的. 这样就不会有阻塞了.
            self.logger.exception("Fulfill task stack failed")
            for child in stack.generated():
                if not child.done():
                    child.fail(e)
            owner.fail(e)

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
        except Exception:
            self.logger.exception("Channel main loop failed")
        finally:
            self.logger.info("channel %s main loop done", self.name)

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
        except FatalError:
            self.logger.exception("Clear failed with fatal error")
            self._stop_event.set()
            raise
        except Exception:
            self.logger.exception("Clear failed")
            raise

    async def _call_self_clear_callback(self) -> None:
        """
        回调所有的 channel 已经执行了 clear.
        """
        try:
            if self.is_available():
                await self.channel.broker.clear()
        except asyncio.CancelledError:
            self.logger.info("channel %s clearing is cancelled", self.name)
        except Exception:
            self.logger.exception("Clear callback failed")

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
