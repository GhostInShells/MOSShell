import logging

from ghoshell_moss.concepts.channel import Channel, Controller, ChannelMeta
from ghoshell_moss.concepts.command import BaseCommandTask, CommandTaskSeq, CommandTask
from ghoshell_moss.concepts.errors import StopTheLoop, FatalError, CommandError
from ghoshell_moss.concepts.shell import ChannelRuntime
from ghoshell_moss.helpers.event import ThreadSafeEvent
from ghoshell_container import IoCContainer
from typing import Dict, Optional, Set, Awaitable, List
from collections import deque
import threading
from anyio.abc import TaskGroup
import anyio
import asyncio
import threading


class ChannelRuntimeImpl(ChannelRuntime):

    def __init__(
            self,
            container: IoCContainer,
            channel: Channel,
            *,
            logger: Optional[logging.Logger] = None,
            stop_event: Optional[ThreadSafeEvent] = None,
    ):
        # 容器应该要已经运行过了.
        self.container = container
        self.logger = logger or logging.getLogger("moss")
        # runtime 级别的关机事件. 会传递给所有的子节点.
        self._stopped_event = stop_event or ThreadSafeEvent()
        self._chan: Channel = channel
        self._name = channel.name()

        # status
        self._started = False
        self._stopped = False

        # runtime properties
        self._children_runtime: Dict[str, ChannelRuntimeImpl] = {}
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None

        # 内部的 运行状态判断.
        self._consuming_loop_done_event: asyncio.Event = asyncio.Event()
        # 主任务.
        self._main_task: Optional[asyncio.Task] = None

        self._is_idle_event: ThreadSafeEvent = ThreadSafeEvent()
        self._allow_consume_pending_loop_event: asyncio.Event = asyncio.Event()

        # 消息队列.
        self._pending_command_task_queue: deque[CommandTask] = deque()
        self._has_pending_task_event: ThreadSafeEvent = ThreadSafeEvent()
        self._pending_queue_locker = asyncio.Lock()

        # 是否是 defer clear 状态.
        self._defer_clear: bool = False

        # 运行中的 task group, 方便整体 cancel. 由于版本控制在 3.10, 暂时无法使用 asyncio 的 TaskGroup.
        self._executing_task_group: Optional[TaskGroup] = None
        self._executing_block_task: bool = False

    async def get_child(self, name: str) -> Optional["ChannelRuntime"]:
        # 检查是否已经启动了.
        self._check_running()
        # 检查是否已经运行了.
        if name in self._children_runtime:
            return self._children_runtime[name]

        # 只能从直系血亲里查找.
        child_chan = self._chan.children().get(name)
        if child_chan is None:
            return None
        if not child_chan.is_running():
            # 启动这个 channel.
            await child_chan.run(self.container).start()

        child_runtime = ChannelRuntimeImpl(
            self.container,
            child_chan,
            logger=self.logger,
            stop_event=self._stopped_event,
        )
        # 考虑到要动态启动.
        await child_runtime.start()
        self._children_runtime[name] = child_runtime
        return child_runtime

    async def start(self):
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        self._running_loop = loop
        # 自身的启动.
        await self._self_bootstrap()
        # 启动所有的子 runtime.
        children_start = []
        for child_chan in self._chan.children().values():
            child_runtime = ChannelRuntimeImpl(
                self.container,
                child_chan,
                logger=self.logger,
                stop_event=self._stopped_event,
            )
            # 添加所有的 runtime 启动任务.
            children_start.append(child_runtime.start())
        # 并行启动所有的子节点.
        if len(children_start) > 0:
            await asyncio.gather(*children_start)

    async def _self_bootstrap(self):
        # 创建主任务.
        if not self._chan.is_running():
            # 启动自身的 channel. 不过这样是效率比较低, 最好提前都启动完了.
            await self._chan.run(self.container).start()
        self._main_task = asyncio.create_task(self._run_main_loop())
        # 启动 task group.
        self._executing_task_group = anyio.TaskGroup()
        await self._executing_task_group.__aenter__()

    async def stop(self):
        # 已经结束过了.
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stopped_event.set()
        await self._executing_task_group.__aexit__(None, None, None)
        # 分层递归结束.
        stop_children = []
        for child in self._children_runtime.values():
            stop_children.append(child.stop())
        # 等待所有的子节点结束. 从自己开始关闭.
        if len(stop_children) > 0:
            await asyncio.gather(*stop_children)

    def _check_running(self):
        if not self._started:
            raise FatalError(f"Channel {self._name} is not running")
        elif self._stopped_event.is_set():
            raise FatalError(f"Channel {self._name} is shutdown")

    def name(self) -> str:
        return self._name

    def is_running(self) -> bool:
        """
        判断 runtime 是否在运行.
        """
        return self._started and not self._stopped_event.is_set()

    def is_busy(self) -> bool:
        """
        判断 runtime 是否是 busy 状态. 任何子节点在运行, 都会是 busy 状态.
        """
        if not self.is_running():
            return False
        elif self._is_self_busy():
            return True
        else:
            # 任何一个子节点忙, 就都在忙了.
            for child in self._children_runtime.values():
                if child.is_busy():
                    return True
            return False

    async def _run_main_loop(self) -> None:
        main_task = asyncio.create_task(self._consume_pending_queue())

        async def cancel_main_task():
            await self._stopped_event.wait()
            if not main_task.done():
                main_task.cancel()

        try:
            await asyncio.gather(main_task, cancel_main_task(), return_exceptions=False)
            # todo log
        except Exception as e:
            self.logger.exception(e)
            # 关闭运行.
            self._stopped_event.set()
        finally:
            self.logger.info(f"channel {self._name} main loop done")
            await self._consuming_loop_done_event.wait()

    async def _consume_pending_queue(self) -> None:
        try:
            policy_is_running = False
            while not self._stopped_event.is_set():
                # 其它的逻辑可能导致消费暂时终止.
                if not self._allow_consume_pending_loop_event.is_set():
                    await self._allow_consume_pending_loop_event.wait()
                    continue

                # 等待新的命令进入.
                if len(self._pending_command_task_queue) == 0:
                    if self._has_pending_task_event.is_set():
                        # 重置错误的标识.
                        self._has_pending_task_event.clear()

                    if not policy_is_running:
                        policy_is_running = True
                        # 运行 policy.
                        await self._chan.controller.policy_run()
                        continue

                    # 等待新命令的输入.
                    try:
                        # 不要做完全的阻塞, 无法感知到 stop
                        await self._has_pending_task_event.wait_until_timeout(0.1)
                    except asyncio.TimeoutError:
                        pass
                    finally:
                        continue
                # 有命令存在.
                else:
                    if policy_is_running:
                        # 阻塞等待 policy 停止运行.
                        await self._chan.controller.policy_pause()

                    # 获取最早的一个任务.
                    cmd_task = self._pending_command_task_queue.popleft()
                    # 运行一个任务. 理论上是很快的调度.
                    # 这个任务不运行结束, 不会释放运行状态.
                    await self._execute_task(cmd_task, blocking=self._chan.controller.is_blocking())
        except Exception as e:
            self.logger.exception(e)
            self._stopped_event.set()
        finally:
            self._consuming_loop_done_event.set()

    async def _execute_task(self, cmd_task: CommandTask, blocking: bool) -> None:
        try:
            # 运行一个任务. 理论上是很快的调度.
            # 这个任务不运行结束, 不会释放运行状态.
            if not blocking:
                # 非阻塞, 直接异步运行. 这种方式不支持 stack.
                self._executing_task_group.start_soon(self._execute_single_task, cmd_task)
            else:
                # 用 task group 启动, 方便统一中断.
                self._executing_block_task = True
                await self._executing_task_group.start(self._execute_task_in_stack, cmd_task, blocking)

        except asyncio.CancelledError:
            # 无所谓, 继续.
            return
        except FatalError as e:
            # 终止队列.
            self.logger.exception(e)
            self._stopped_event.set()
        except CommandError as e:
            # 所有的 command error 都忽视.
            self.logger.info("execute command `%r`error: %s", cmd_task, e)
        except Exception as e:
            # 没有到 Fatal Error 级别的都忽视.
            self.logger.exception(e)
        finally:
            if blocking:
                self._executing_block_task = False

    async def _execute_task_in_stack(self, task: CommandTask) -> None:
        try:
            task.set_state("running")
            stack = deque([task])
            while len(stack) > 0:
                first = stack.popleft()

                # 先做血缘调度.
                execution_channel = first.meta.chan
                # 根本不是自己的, 要尽快调度. 但如果 task 的 channel 为空, 表示是主轨, 则也可立刻运行.
                if not execution_channel or execution_channel != self._name:
                    await self._send_to_child(execution_channel, first)
                    continue

                new_stack = await self._execute_single_task(first)
                # 栈操作, 插入到队首.
                if new_stack:
                    stack.extendleft(new_stack)

        except asyncio.CancelledError:
            # 忽视错误.
            # todo: log
            return
        except FatalError as e:
            self.logger.exception(e)
            self._stopped_event.set()
        except CommandError as e:
            self.logger.info("execute command `%r`error: %s", e, e)
        finally:
            # 不要留尾巴?
            if not task.done():
                task.cancel()

    async def _send_to_child(self, execution_channel: str, cmd_task: CommandTask) -> None:
        # 怕万一没启动呢?
        child_runtime = await self.get_child(execution_channel)
        if child_runtime is not None:
            # 直接发送给子节点.
            await child_runtime.append(cmd_task)
            return

        for child_channel in self._chan.children().values():
            # 判断在不在哪个子节点的所有子孙节点里. 当 channel 特别多的时候, 有性能损耗.
            # 但子 channel 会不会动态注册孙 channel, 目前还不能确认.
            if execution_channel in child_channel.descendants():
                # 还是发送给这个儿女.
                child_runtime = await self.get_child(child_channel.name())
                await child_runtime.append(cmd_task)
                return
        # 丢弃掉不认识的.
        self.logger.error(f"channel {self._name} executing found orphan task {cmd_task}")

    async def _execute_single_task(self, cmd_task: CommandTask) -> Optional[List[CommandTask]]:

        try:
            if cmd_task.done():
                cmd_task.raise_exception()
                return cmd_task.result

            cmd_task.set_state('running')

            async def _resolve():
                _result = cmd_task.func(*cmd_task.args, **cmd_task.kwargs)
                cmd_task.resolve(_result)

            async def _wait_done():
                await cmd_task.wait()
                # 实际上可能没有.
                cmd_task.raise_exception()

            await asyncio.gather(_resolve(), _wait_done())

            return cmd_task.result

        except asyncio.CancelledError:
            # todo: log
            cmd_task.cancel()
            return
        except FatalError as e:
            # todo: log
            self.logger.exception(e)
            self._stopped_event.set()
            return None
        except CommandError as e:
            cmd_task.fail(e)
            return None
        except Exception as exc:
            # 忽视其它的异常. 仅仅记录.
            self.logger.exception(exc)
            cmd_task.fail(str(exc))
            return None
        finally:
            if not cmd_task.done():
                cmd_task.cancel()

    async def append(self, *tasks: CommandTask) -> None:
        if not self.is_running():
            # todo: log
            return
        pending = []
        for task in tasks:
            if task.meta.call_soon and task.meta.block:
                # 先判断要不要清空.
                pending = []
            else:
                pending.append(task)
        if len(pending) > 0:
            try:
                for task in pending:
                    # 一次只加入一个.
                    await self._append(task)
            except asyncio.CancelledError:
                self.logger.info(f"channel {self._name} pending tasks cancelled")
                raise
            except Exception as exc:
                self.logger.exception(exc)
                raise

    async def _append(self, task: CommandTask) -> None:
        # 必须先上锁, 否则不知道发生什么事情.
        await self._pending_queue_locker.acquire()
        # 推送到等待队列中.
        task.set_state('pending')
        #  需要在添加命令时就执行
        if task.meta.call_soon:
            # 清空队列先.
            block = task.meta.block
            if block:
                await self._clear()
            await self._execute_task(task, blocking=task.meta.block)
            return

        try:
            self._pending_command_task_queue.append(task)
            # 通知消费队列.
            self._has_pending_task_event.set()
        finally:
            self._pending_queue_locker.release()

    def clear_pending(self) -> None:
        self._check_running()
        self._running_loop.call_soon_threadsafe(self._clear_pending)

    async def _clear_pending(self) -> None:
        try:
            await self._pending_queue_locker.acquire()
            # 先清空自身的队列.
            await self._clear_self_pending()
            clear_children = []
            for child in self._children_runtime.values():
                clear_children.append(child.clear_pending())
            if len(clear_children) > 0:
                await asyncio.gather(*clear_children)
        except asyncio.CancelledError:
            self.logger.info("channel %s clear pending but canceled", self._name)
            raise
        except Exception as exc:
            self.logger.exception(exc)
            # 所有没有管理的异常, 都是致命异常.
            self._stopped_event.set()
            raise exc
        finally:
            self._pending_queue_locker.release()

    async def _clear_self_pending(self) -> int:
        if len(self._pending_command_task_queue) > 0:
            cleared = len(self._pending_command_task_queue)
            self._pending_command_task_queue.clear()
            return cleared
        return 0

    def cancel_running(self) -> None:
        self._check_running()
        self._running_loop.call_soon_threadsafe(self._cancel_running)

    async def _cancel_running(self) -> None:
        # 准备并发 cancel 所有的运行.
        cancel_running = [self._cancel_self_running()]
        for child in self._children_runtime.values():
            cancel_running.append(child.cancel_running())
        await asyncio.gather(*cancel_running)

    def clear(self) -> None:
        self._check_running()
        self._running_loop.call_soon_threadsafe(self._clear)

    async def _clear(self) -> None:
        self._check_running()
        try:
            # 暂停所有的消费动作.
            self._allow_consume_pending_loop_event.clear()
            # 先清空队列. 递归地清空.
            has_pending = await self._clear_pending()
            # 然后清空运行中的任务.
            cancel_running = await self._cancel_running()
            # 通知自己所有的 channel 清空.
            if has_pending or cancel_running:
                await self._run_clear_callback()

        except asyncio.CancelledError:
            self.logger.info("channel %s clearing is cancelled", self._name)
            raise
        except Exception as exc:
            self.logger.exception(exc)
            raise
        finally:
            self._allow_consume_pending_loop_event.set()

    async def _run_clear_callback(self) -> None:
        """
        回调所有的 channel 已经执行了 clear.
        """
        clear_callbacks = [self._chan.controller.clear()]
        for child in self._children_runtime.values():
            clear_callbacks.append(child._run_clear_callback())
        await asyncio.gather(*clear_callbacks)

    def defer_clear(self) -> None:
        """
        准备清空运行状态, 如果有指令输入的话.
        """
        # defer clear 不需要递归. 因为所有子节点的任务来自父节点.
        self._defer_clear = True

    def destroy(self) -> None:
        """
        自身清理掉, 防止有人继续运行它.
        """
        pass

    async def _cancel_self_running(self) -> None:
        if self._executing_task_group is not None:
            # 清除所有运行中的任务.
            self._executing_task_group.cancel_scope.cancel()

    def _is_self_busy(self) -> bool:
        return len(self._pending_command_task_queue) > 0 or self._executing_block_task
