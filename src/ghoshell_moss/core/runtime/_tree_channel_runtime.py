import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Any, TypeVar

from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    CommandUniqueName,
    Command,
    CommandTaskState,
)
from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
    Channel,
    ChannelFullPath,
    ChannelPaths,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_common.contracts import LoggerItf
import janus
from ._base_channel_runtime import AbsChannelRuntime

__all__ = ["AbsChannelTreeRuntime"]

_ChannelId = str
CHANNEL = TypeVar("CHANNEL", bound=Channel)
_TaskId = str
_TaskIdWithPaths = tuple[ChannelPaths, _TaskId]


class AbsChannelTreeRuntime(AbsChannelRuntime, ABC):
    # --- main loop --- #

    def __init__(self, *, channel: CHANNEL, container: IoCContainer | None = None, logger: LoggerItf | None = None):
        super().__init__(
            channel=channel,
            container=container,
            logger=logger,
        )
        # 线程安全队列.
        self._blocking_action_queue = janus.Queue()

        self._blocking_action_lock = asyncio.Lock()
        # 通知有 pending task 的队列.
        self._pending_task_queue: asyncio.Queue[_TaskIdWithPaths | None] = asyncio.Queue()
        # 运行状态池.
        # 生命周期任务.
        self._lifecycle_task: asyncio.Task | None = None
        # 在队列中阻塞的任务.
        self._pending_tasks: dict[_TaskId, CommandTask] = {}
        # 在执行中的异步任务.
        self._executing_self_tasks: dict[_TaskId, CommandTask] = {}
        # 在执行中的非异步任务.
        self._executing_blocking_task: CommandTask | None = None
        # is self idle event
        self._idled_event = asyncio.Event()

    @abstractmethod
    def sub_channels(self) -> dict[str, Channel]:
        """
        当前持有的子 Channel.
        """
        pass

    def commands(self, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        commands = self.own_commands(available_only).copy()
        result = {"": commands}
        for name, child in self.sub_channels().items():
            child_runtime = self.importlib.get_channel_runtime(child)
            if child_runtime and child_runtime.is_running():
                child_commands = child_runtime.commands(available_only)
                for further_path, command_map in child_commands.items():
                    new_full_path = Channel.join_channel_path(name, further_path)
                    result[new_full_path] = command_map
        return result

    @abstractmethod
    def get_own_command(self, name: str) -> Optional[Command]:
        pass

    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        chan, command_name = Command.split_unique_name(name)
        if chan == "":
            return self.get_own_command(command_name)
        runtime = self.importlib.recursively_find_runtime(self, chan)
        if runtime is None:
            return None
        return runtime.get_command(command_name)

    async def wait_idle(self) -> None:
        """
        阻塞等待到闲时.
        """
        if not self.is_running():
            return
        wait_1 = asyncio.create_task(self._idled_event.wait())
        wait_2 = asyncio.create_task(self._closing_event.wait())
        done, pending = await asyncio.wait([wait_1, wait_2], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

    # --- lifecycle --- #

    async def _idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Runtime 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        if not self.is_running():
            return
        await self._clear_lifecycle_task()
        await self._blocking_action_lock.acquire()
        try:
            await asyncio.sleep(0.0)
            ctx = ChannelCtx(self)
            on_idle_cor = ctx.run(self.on_idle)
            # idle 是一个在生命周期中单独执行的函数.
            task = asyncio.create_task(on_idle_cor)
            self._lifecycle_task = task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception("%s idle task failed %s", self.log_prefix, exc)
            # 不返回.
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s idling, pending tasks %d", self.log_prefix, len(self._pending_tasks))

    @abstractmethod
    async def on_idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Runtime 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        pass

    async def _clear_lifecycle_task(self) -> None:
        """
        终止进行中的生命周期函数.
        """
        # 终止阻塞中的任务.
        self._idled_event.clear()
        await self._blocking_action_lock.acquire()
        try:
            if self._lifecycle_task and not self._lifecycle_task.done():
                self._lifecycle_task.cancel()
                await self._lifecycle_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s clear lifecycle task failed: %s", self.log_prefix, e)
        finally:
            self._lifecycle_task = None
            self._blocking_action_lock.release()

    async def wait_children_idled(self) -> None:
        async def wait_child_empty(_child: Channel):
            runtime = await self._importlib.get_or_create_channel_runtime(_child)
            if runtime and runtime.is_running():
                await runtime.wait_idle()
            return

        wait_all = []
        children = self.sub_channels()
        if len(children) > 0:
            for child in children.values():
                wait_all.append(wait_child_empty(child))
            _ = await asyncio.gather(*wait_all)

    def _is_children_idled(self) -> bool:
        children = self.sub_channels()
        if len(children) > 0:
            for child in children.values():
                runtime = self.importlib.get_channel_runtime(child)
                if not runtime or not runtime.is_running():
                    continue
                elif not runtime.is_idle():
                    return False
        return True

    def is_idle(self) -> bool:
        return self.is_running() and self._idled_event.is_set()

    async def _main_loop(self) -> None:
        try:
            # 等待启动再开始.
            await self.wait_started()
            while not self._closing_event.is_set():
                # 确保让出.
                await asyncio.sleep(0.0)
                _pending_queue = self._pending_task_queue
                # 如果队列是空的, 则要看看是否能够启动 idle.
                if _pending_queue.empty() and not self._idled_event.is_set():
                    # 存在执行中的任务, 继续去拉取.
                    if self._executing_blocking_task or len(self._pending_tasks) > 0:
                        continue
                    # 可以执行 idle 了.
                    if self._is_children_idled():
                        # 这种情况下就真的可以 idle 了. 速度应该很快.
                        await self._idle()
                        self._idled_event.set()
                        continue
                # 阻塞等待下一个结果.
                try:
                    item = await asyncio.wait_for(_pending_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # 可能拿到了 clear 清空后的毒丸.
                if item is None:
                    self.logger.info("%s receive none from pending task queue", self.log_prefix)
                    continue
                # 拿到新命令后, 就清空生命周期函数.
                paths, task_id = item
                # consume 动作认为是阻塞的, 它会快速执行, 然后去拉下一个 task.
                # 它唯一的目标就是快速消费.
                await self._consume_task(paths, task_id)
        except asyncio.CancelledError as e:
            # 允许被 cancel.
            self.logger.info("%s Cancel consuming pending task loop: %r", self.log_prefix, e)
        finally:
            self._closing_event.set()
            self.logger.info("%s Finished executing loop", self.log_prefix)

    async def _dispatch_children_task(self, paths: ChannelPaths, task: CommandTask) -> None:
        await asyncio.sleep(0)
        if task.done():
            return
        child_name = paths[0]
        # 子节点在路径上不存在.
        child = self.sub_channels().get(child_name)
        if child is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"channel `{task.chan}` not found"))
            return

        runtime = await self.importlib.get_or_create_channel_runtime(child)
        if runtime is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"channel `{task.chan}` not found"))
            return
        task.send_through.append(child_name)
        # 直接发送给子树.
        further_paths = paths[1:]
        await runtime.push_task_with_paths(further_paths, task)

    async def _consume_task(self, paths: ChannelPaths, task_id: str) -> None:
        """
        尝试运行一个 task. 这个运行周期是全局唯一, 阻塞的.
        """
        if task_id not in self._pending_tasks:
            return None
        consuming = None
        try:
            # consuming 过程中让出一次.
            await asyncio.sleep(0)
            # 阻塞任务存在的时候, 必须等到阻塞任务完成, 或者它被取消.
            # 这里不做优先级检查, 因为入队时做过了.
            if self._executing_blocking_task is not None and not self._executing_blocking_task.done():
                # 等待阻塞任务因为任何原因完成.
                await self._executing_blocking_task.wait(throw=False)
                # 只有 consuming 环节可以控制 executing blocking task
                self._executing_blocking_task = None

            try:
                consuming = self._pending_tasks.pop(task_id)
            except KeyError:
                return None
            if consuming.done():
                consuming = None
                return None

            is_self_task = len(paths) == 0
            is_blocking_task = consuming.meta.blocking
            # 检查是不是子节点的任务.
            if not is_self_task:
                # 分配给子节点.
                await self._dispatch_children_task(paths, consuming)
                consuming = None
                return None

            if is_blocking_task:
                # 只有 consume 层可以设置 blocking task. 协程安全操作.
                self._executing_blocking_task = consuming
            # 执行自己的任务. 但并不阻塞.
            await self._clear_lifecycle_task()
            await self._execute_self_task_none_block(consuming)
            consuming = None

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception("%s handle pending task exception: %r", self.log_prefix, e)
        finally:
            # 这个时候, consuming_command_task 正常应该都设置为 None 了.
            if consuming is not None:
                # 不合法的情况, 要检查原因.
                self.logger.error(
                    "%s consuming task not handled: %r",
                    self.log_prefix,
                    consuming,
                )
                consuming.cancel()

    async def _get_task_result(self, task: CommandTask) -> Any:
        # 准备执行.
        await asyncio.sleep(0)
        self.logger.info("%s start task %s", self.log_prefix, task.cid)
        # 初始化函数运行上下文.
        # 使用 dry run 来管理生命周期.
        async with ChannelCtx(self, task).in_ctx():
            # dry run 不会清空 task 状态.
            return await task.dry_run()

    async def _execute_self_task_none_block(self, task: CommandTask, depth: int = 0) -> asyncio.Task | None:
        """
        阻塞完成一个任务的运行准备.
        这里没有让出逻辑.
        task 虽然被执行了, 但
        """
        # 又要检查一次.
        if task is None or task.done():
            return None
        if depth > 10:
            task.fail(CommandErrorCode.INVALID_USAGE.error("stackoverflow"))
            return None
        # 确保 task 被加入了状态池.
        await self._add_executing_task(task)
        # 非阻塞函数不能返回 stack
        # 确保 task 被执行了. 但是不要阻塞主链路.
        return self._loop.create_task(self._ensure_task_executed(task, depth, throw=False))

    async def _add_executing_task(self, task: CommandTask) -> None:
        await self._blocking_action_lock.acquire()
        try:
            cid = task.cid
            if cid in self._executing_self_tasks:
                return
            self._executing_self_tasks[cid] = task
            if cid in self._pending_tasks:
                del self._pending_tasks[cid]
            task.set_state(CommandTaskState.executing)
            # 设置 channel id 来标记执行者.
            task.exec_chan = self.channel.id()
            task.add_done_callback(self._on_executing_task_done)
        finally:
            self._blocking_action_lock.release()

    def _on_executing_task_done(self, task: CommandTask) -> None:
        if not self.is_running():
            return
        # 确保垃圾回收.
        cid = task.cid
        try:
            _ = self._executing_self_tasks.pop(cid)
        except KeyError:
            pass

    async def _ensure_task_executed(self, task: CommandTask, depth: int, throw: bool) -> None:
        """
        运行属于自己这个 channel 的 task, 让它进入到 executing group 中.
        """
        # 由于是异步执行的, 再检查一次.
        task = self._parse_task(task)
        if task is None:
            return
        await self._add_executing_task(task)

        get_result_from_task = self._loop.create_task(self._get_task_result(task))
        try:
            origin_task_done = asyncio.create_task(task.wait(throw=False))
            wait_runtime_close = asyncio.create_task(self._closing_event.wait())
            done, pending = await asyncio.wait(
                [origin_task_done, get_result_from_task, wait_runtime_close],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if origin_task_done in done:
                # origin task 已经运行结束.
                return
            elif wait_runtime_close in done:
                task.fail(CommandErrorCode.NOT_RUNNING.error("runtime closed"))
                return
            result = await get_result_from_task
            # 如果返回值是 stack, 则意味着要循环堆栈.
            if isinstance(result, CommandStackResult):
                # 执行完所有的堆栈. 同时设置真实被执行的任务.
                (await self._fulfill_task_with_its_result_stack(task, result, depth=depth),)
            else:
                # 赋值给原来的 task.
                task.resolve(result)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            if throw:
                raise
        except Exception as e:
            if not task.done():
                task.fail(e)
            self.logger.error("%s task %s failed: %s", self.log_prefix, task.cid, e)
            if throw:
                raise e
        finally:
            if not task.done():
                self.logger.info("%s failed to ensure task done: %s", self.log_prefix, task)
                task.fail(CommandErrorCode.UNKNOWN_ERROR.error(f"execution failed"))
            # 还要确保 get result 这个函数被清空了.
            if task is self._executing_blocking_task:
                self._executing_blocking_task = None
            if task.cid in self._pending_tasks:
                del self._pending_tasks[task.cid]
            if not get_result_from_task.done():
                try:
                    get_result_from_task.cancel()
                    # 确保函数执行到了 finally
                    await get_result_from_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.exception(
                        "%s task %s cancel get result failed: %s",
                        self.log_prefix,
                        task,
                        e,
                    )

    async def _fulfill_task_with_its_result_stack(
            self,
            owner: CommandTask,
            stack: CommandStackResult,
            depth: int = 0,
    ) -> None:
        result = stack
        while result is not None:
            get_stack_result = asyncio.create_task(
                self._run_result_stack(owner, result, depth=depth),
            )
            self_done = asyncio.create_task(owner.wait(throw=False))
            done, pending = await asyncio.wait(
                [get_stack_result, self_done],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            result = await get_stack_result

    async def _run_result_stack(
            self,
            owner: CommandTask,
            stack: CommandStackResult,
            depth: int = 0,
    ) -> CommandStackResult | None:
        result = None
        try:
            if not owner.meta.blocking:
                owner.fail(CommandErrorCode.INVALID_USAGE.error(f"invalid command: none blocking task return stack"))
                return None
            if depth > 10:
                owner.fail(CommandErrorCode.INVALID_USAGE.error("stackoverflow"))
                return None

            self.logger.info(
                "%s Fulfilling task with stack, depth=%s task=%s",
                self.log_prefix,
                depth,
                owner,
            )
            # 遍历生成的新栈.
            async with stack:
                async for sub_task in stack:
                    await asyncio.sleep(0)
                    if owner.done():
                        # 不要继续执行了.
                        break
                    paths = Channel.split_channel_path_to_names(sub_task.chan)
                    if len(paths) > 0:
                        # 发送给子孙了.
                        await self._dispatch_children_task(paths, sub_task)
                        continue

                    # 递归阻塞等待任务被执行.
                    if sub_task.meta.blocking:
                        # 自己的任务仍然要阻塞一下.
                        await self._ensure_task_executed(sub_task, depth=depth + 1, throw=True)
                    else:
                        _ = asyncio.create_task(self._ensure_task_executed(sub_task, depth=depth, throw=False))

                # 完成了所有子节点的调度后, 通知回调函数.
                # !!! 注意: 在这个递归逻辑中, owner 自行决定是否要等待所有的 child task 完成,
                #          如果有异常又是否要取消所有的 child task.
                result = await stack.callback(owner)
                return result
        except asyncio.CancelledError:
            if not owner.done():
                owner.cancel()
            raise
        except Exception as e:
            # 有异常时, 同时取消所有动态生成的 task 对象. 包括发送出去的. 这样就不会有阻塞了.
            self.logger.exception(
                "%s Fulfill task stack failed, task=%s, exception=%s",
                self.log_prefix,
                owner,
                e,
            )
            for child in stack.generated():
                if not child.done():
                    child.fail(e)
            owner.fail(e)
        finally:
            # owner 结束时, 子任务可能并未完成.
            if result is None and not owner.done():
                owner.cancel()

    async def _push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        基于路径将任务入栈.
        入栈是高优的同步任务.
        """
        try:
            # 是自己的, 而且是要立刻执行的任务.
            task = self._parse_task(task)
            if task is None:
                return
            task_id = task.cid
            # set pending
            task.set_state(CommandTaskState.pending.value)
            # 确认是自身的任务, 并且 call soon.
            is_self_task = len(paths) == 0
            is_blocking_task = task.meta.blocking
            # 阻塞等待 compiled. 等得过久怎么办? 就得靠 shell clear 了.
            priority = task.meta.priority
            # 进入 pending 列表.
            if is_self_task:
                # 清理运行中的 lifecycle task
                await self._clear_lifecycle_task()
                # call soon
                if task.meta.call_soon:
                    if is_blocking_task:
                        # 需要立刻执行, 而且是一个阻塞类的任务, 则会清空所有运行中的任务.
                        # 设置清空等级为最高.
                        priority = None
                    else:
                        # 立刻将它执行, none blocking 任务确认会进入到并行运行.
                        await self._execute_self_task_none_block(task, depth=0)
                        # 并不阻塞等待结果, 而是立刻返回.
                        return
            # 来一次优先级的 pk.
            if is_blocking_task:
                self._clear_own_task_by_priority(task.chan, task.cid, priority)
            self._pending_tasks[task_id] = task
            # 普通的任务, 则会被丢入阻塞队列中排队执行.
            _queue = self._pending_task_queue
            # 入栈.
            _queue.put_nowait((paths, task_id))
        except asyncio.QueueFull:
            task.fail(CommandErrorCode.FAILED.error(f"channel queue is full, clear first"))

    def _clear_own_task_by_priority(self, chan: str, cid: str, priority: int | None):
        """
        根据优先级清空自身的任务.
        如果 priority 为空, 表示最高优先级, 不做比较.
        """

        reason = "interrupted by higher priority command"
        if self._executing_blocking_task is not None and not self._executing_blocking_task.done():
            # < 0 的 task 任何时候都会被取消.
            if self._executing_blocking_task.meta.priority < 0:
                self._executing_blocking_task.cancel(reason)

        # 接下来只有 priory > 0 的才有资格去取消任务.
        if priority is not None and priority <= 0:
            # 误操作, 没有资格做比较.
            return
        if self._executing_blocking_task is not None and not self._executing_blocking_task.done():
            if self._executing_blocking_task.cid == cid:
                pass
            elif priority is None or self._executing_blocking_task.meta.priority < priority:
                self._executing_blocking_task.cancel(reason)
        for task in self._pending_tasks.values():
            # 预先清空队列中优先级低于自身的命令.
            if task.chan != chan or task.cid == cid:
                continue
            if priority is None or (task.meta.blocking and task.meta.priority < priority):
                if not task.done():
                    task.cancel(reason)

    async def clear_own(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        仅仅清空自身的运行中状态.
        """
        if not self._started.is_set() or self._closed_event.is_set():
            return
        await self._blocking_action_lock.acquire()
        try:
            clear_err = CommandErrorCode.CLEARED.error("cleared by runtime")
            if len(self._pending_tasks) > 0:
                pending_tasks = self._pending_tasks.copy()
                self._pending_tasks.clear()
                for task in pending_tasks.values():
                    if not task.done():
                        task.fail(clear_err)
            # 清空存在的 tasks. 避免内存泄漏. 虽然有队列在拉取.
            self._pending_tasks.clear()

            # 并行执行的 task 也需要被清除.
            if len(self._executing_self_tasks) > 0:
                executing_tasks = self._executing_self_tasks.copy()
                self._executing_self_tasks.clear()
                for t in executing_tasks.values():
                    if not t.done():
                        t.fail(clear_err)
        except Exception as e:
            self.logger.exception("%s clear self failed: %s", self.log_prefix, e)
            raise
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s cleared", self.log_prefix)
