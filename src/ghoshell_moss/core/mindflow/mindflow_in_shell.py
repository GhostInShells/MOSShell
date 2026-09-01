from abc import ABC, abstractmethod
import asyncio
import contextlib
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, Thinking, Action, Signal, StatementExitedException,
    Attention, NucleusMeta
)
from ghoshell_moss.core.blueprint.moment import Moment
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.concepts.errors import FatalError, InterpretError
from ghoshell_moss.core.concepts.shell import InterpreterKind
from ghoshell_moss.core.concepts.interpreter import Interpretation
import logging

__all__ = ["MindflowInShell"]


class MindflowInShell(ABC):
    """ Mindflow 三循环的标准装线逻辑. """

    @property
    @abstractmethod
    def mindflow(self) -> Mindflow:
        ...

    # ── 生命周期 ──────────────────────────────────

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @abstractmethod
    def _collect_nuclei_metas(self) -> Iterable[NucleusMeta]:
        ...

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        ...

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        ...

    @property
    @abstractmethod
    def shell_trajectory(self) -> MShellTrajectory:
        ...

    @abstractmethod
    def _when_signal_added(self, callback: Callable[[Signal], None]):
        ...

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        ...

    @abstractmethod
    async def _refresh_shell(self) -> None:
        ...

    @abstractmethod
    def _is_thinking_gated(self) -> bool:
        ...

    async def _approve_logos(self, logos: str) -> tuple[bool, str]:
        """gated 模式下裁决完整 logos. 返回 (approved, message). 默认放行。"""
        return (True, '')

    @abstractmethod
    async def _articulate_from_thinking(self, thinking: Thinking) -> None:
        ...

    @abstractmethod
    async def _fire_interpreter_result(
            self,
            interpretation: Interpretation
    ) -> None:
        ...

    @abstractmethod
    async def _enter_async_context(self, manager: contextlib.AbstractAsyncContextManager) -> None:
        """把 async context manager 注册到宿主生命周期, 保证退出时反卷.

        mindflow 的启动/关闭由此装线进宿主的 exit stack, 使错误/任务取消之外的
        正常关闭路径也能确定性触发 mindflow.__aexit__.
        """
        ...

    def _on_mindflow_loop_task(self, future: asyncio.Future, *, name: str | None = None) -> None:
        pass

    def _on_thinking_start(self, thinking: Thinking) -> None:
        pass

    def _on_thinking_exited(self, thinking: Thinking, err: BaseException | None) -> None:
        pass

    def _on_logos_delta(self, delta: str) -> None:
        pass

    def _on_mindflow_error(self, error: BaseException | str) -> None:
        pass

    async def _wire_mindflow(self) -> None:
        container = self.container
        mindflow = self.mindflow
        for nucleus_meta in self._collect_nuclei_metas():
            try:
                nucleus = nucleus_meta.factory(container)
            except NotImplementedError:
                self.logger.warning(
                    "%r nucleus %s is a stub (NotImplementedError), skipping",
                    self, nucleus_meta.name(),
                )
                continue
            except Exception as e:
                self.logger.exception(
                    "%r failed to create nucleus %s, skipping on errr: %s",
                    self, nucleus_meta.name(), e,
                )
                continue
            mindflow.with_nucleus(nucleus, override=True)

        # session signal → mindflow 路由.
        # zenoh 存活周期比 ghost/mindflow 长, 关闭期间 session 仍可能收到信号,
        # 所以闭包内检查 mindflow.is_running() 做兜底丢弃.
        def _route_signal_to_mindflow(signal: Signal):
            if mindflow.is_running():
                mindflow.add_signal(signal)

        # mindflow 装线 shell trajectory, 注册观测函数.
        def _on_moments_observing(moment: Moment) -> None:
            frame = self.shell_trajectory.pop_frame()
            if moment.previous is not None:
                # 将轨迹中保存的数据作为回声插入.
                messages = frame.project(with_dynamic=False)
                moment.previous.add_echoes(messages, frame.need_observe)
                moment.previous.need_observe = frame.need_observe

        def _notify_moments_need_observe(e):
            # 仅仅通知观测应该发生. 真实的观测数据, 会在 moment 创建时回调构建.
            mindflow.moments.add_echoes([], need_observe=True)

        # 注册回调, 当发生 observe 事件时, 通知 mindflow observer.
        self.shell_trajectory.when_need_observe(_notify_moments_need_observe)
        # 注册回调, 当 observer 触发观察动作时, 更新数据.
        mindflow.moments.on_moment_created(_on_moments_observing)
        # 反向绑定: moments.new_epoch → trajectory.new_epoch (刷新 baseline 快照).
        # 先 on_epoch_creating (刷新) 再 with_epoch_baseline (从已刷新 baseline 产字符串).
        mindflow.moments.on_epoch_creating(lambda _epoch: self.shell_trajectory.new_epoch())
        # epoch 起点全量 facade 走 baseline 槽位 (非 recap): facade 是 shell 表面,
        # 不是前情提要. 首帧 diff 与之同源, 天然去重.
        mindflow.moments.with_epoch_baseline(
            "facade",
            lambda: self.shell_trajectory.epoch_start_point(refresh=False),
        )

        # mindflow 生命周期装线到宿主 exit stack: 启动由宿主保证, 关闭确定性反卷.
        await self._enter_async_context(mindflow)

        # 三循环托管给 matrix. mindflow 自身的内部循环已由 __aenter__ 启动,
        # 宿主只消费 thinking/action 两个生产循环.
        self._on_mindflow_loop_task(asyncio.create_task(self._thinking_loop()), name="mindflow_thinking_loop")
        self._on_mindflow_loop_task(asyncio.create_task(self._action_loop()), name="mindflow_action_loop")
        # 等待应该发生在循环外侧.
        await self.mindflow.wait_started()
        # ignore any signals before started
        self._when_signal_added(_route_signal_to_mindflow)

    # ── 三循环 ────────────────────────────────────

    async def _thinking_loop(self) -> None:
        """queue → ghost.articulate(articulator) → send_nowait + pub_logos.

        output 时序:
          - articulator 入队 → output('moment', log=...)  ghost 感知到了什么
          - delta 产出       → pub_logos(delta)           实时流, 外部通过 get_logos() 消费
          - 结束 (成功/失败) → ghost.on_articulate_exit()  调试附着点
        """
        mindflow = self.mindflow
        await mindflow.wait_started()
        # 组装 mindflow channel.
        if channel := mindflow.as_channel():
            self.shell.main_channel.add_virtual_channel(channel)
        # 首次要刷新, 获取关键帧.
        await self.shell.refresh_metas()
        try:
            while mindflow.is_running():
                last_attention: Attention | None = None
                async for thinking in self.mindflow.thinking_loop():
                    interrupt_first = False
                    if last_attention and thinking.attention.id != last_attention.id:
                        last_attention = thinking.attention
                        interrupt_first = last_attention.draw_from().interrupt
                    # 按规则中断.
                    if interrupt_first or thinking.moment.previous_stop_reason():
                        await self.shell.clear()
                    await self._run_thinking(thinking)
        finally:
            self.logger.info("%r thinking loop finished", self)

    async def _run_thinking(self, thinking: Thinking) -> None:
        err = None
        try:
            # 每次开始运行时必须刷新, 但如果刚刚刷新过, 在阈值内, 可以快速跳过.
            await self._refresh_shell()
            # 启动 thinking 生命周期.
            self._on_thinking_start(thinking)
            async with thinking:
                tasks = []
                if self._is_thinking_gated():
                    thinking.register_gate(self._approve_logos)

                moment = thinking.moment
                # 发送已经执行的命令.
                if moment.command_logos:
                    # 发送 command logos 作为第一波.
                    async with thinking.articulator(replan=False, wait_action_done=False) as articulator:
                        command_logos = moment.command_logos
                        articulator.send_nowait(command_logos)
                        self._on_logos_delta(command_logos)
                        await articulator.wait_compiled()
                # 如果
                if thinking.effort() == 'none':
                    return

                # -- 需要阻塞执行完的逻辑完成 -- #

                fut = asyncio.create_task(self._articulate_from_thinking(thinking))
                tasks.append(fut)
                # 同步阻塞到结束. 如果 thinking 提前结束, 也会中断所有的 tasks.
                await thinking.wait_until_done(*tasks)

        except FatalError as e:
            err = e
            self.logger.error("%r run thinking fatal error", self)
            raise
        except asyncio.CancelledError as e:
            err = e
            pass
        except BaseException as e:
            err = e
            self.logger.error("%r thinking loop error: %s", self, e)
            self._on_mindflow_error(e)
        finally:
            self._on_thinking_exited(thinking, err)

    async def _action_loop(self) -> None:
        """queue → action.received_logos() → interpreter → action.outcome().

        Interpreter 三阶段:
          1. feed    — 流式送入 delta, throw=True 确保异常立刻打断循环
          2. compile — commit() + wait_compiled() 检查 CTML 语法/语义
          3. execute — wait_stopped() 等待所有 CommandTask 执行完毕

        异常分级 (决定 as_messages 内容和 observe 返回值):
          1. InterpretError — 可管理中断 (模型 CTML 错误 / shell.clear).
             interpreter 内部设 observe=True + 取消 pending tasks.
             模型在下一轮 Moment 看到错误后可自我纠正.
          2. Task 级失败 — 单个命令执行异常. 捕获在 failed_tasks,
             task_result().observe 决定是否触发观察. 不中断整体解释.
          3. 静默失败 — 非关键组件异常. 应 log 到 matrix 但不呈现给模型.
          4. 致命异常 — shell/matrix 崩溃. 向外传播, 由 matrix task 管理器处理.
        """
        mindflow = self.mindflow
        await mindflow.wait_started()
        last_task = None
        try:
            while mindflow.is_running():
                async for action in mindflow.action_loop():
                    # 同一时间, 只能有一个 action task 运行.
                    if last_task is not None and not last_task.done():
                        last_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await last_task

                    # 为每个 action 创建 task, 然后继续拉最新的 action.
                    # 永远都是最新的 action 顶掉旧的.
                    last_task = asyncio.create_task(self._run_action(action))
                    # 由于是队列逻辑, 仍然要让出让 last task 执行.
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except FatalError:
            self.logger.error("%r action loop fatal error", self)
            raise
        except Exception as e:
            self.logger.exception("%sr action loop error: %s", self, e)
            self._on_mindflow_error(e)
        finally:
            if last_task is not None and not last_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await last_task

    async def _run_action(self, action: Action) -> None:
        try:
            async with action:
                # 是否要 shield 到这里?
                await action.wait_ready()
                if action.is_aborted():
                    # 在产生副作用之前关闭.
                    return
                # 启动 interpreter.
                await self._run_interpreter_with_action(action)
        except FatalError:
            self.logger.exception("%r action fatal error", self)
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 未捕获的异常都要导致停止.
            self.logger.exception(
                "%r action loop error: %s", self, e, exc_info=True,
            )
            self._on_mindflow_error(e)
            # 未兜底异常需要清空状态.
            await self.shell.clear()

    async def _run_interpreter_with_action(self, action: Action) -> None:
        """流式执行: action.received_logos() → interpreter.feed(delta) → 结算.

        返回 (as_messages, observe) 闭合 observe 回路.
        logos 已走 session stream 实时广播, 此处只发射 command-output/result.
        InterpretError 被捕获 — interpretation 已保留 partial results.

        Attention abort 传播: 在 feed/compile/execute 各阶段结束后检查
        action.is_aborted(), 发现后调用 shell.clear() 取消 pending command,
        返回部分结果.
        """
        shell = self.shell
        if not shell.is_running():
            self.logger.error(
                "%sr ghost runtime received action but shell is not running",
                self,
            )
            self._on_mindflow_error('received action but shell is not running')
            return

        if action.replaned:
            kind: InterpreterKind = 'clear'
        else:
            kind: InterpreterKind = 'append'

        interpreter = await shell.interpreter(kind=kind, clear_after_exit=False)
        interpretation = interpreter.interpretation()

        logger = self.logger

        async def _abort_clear() -> bool:
            """action 已 abort 时 clear shell, 取消 pending command. 返回是否触发.

            interpreter 以 ``clear_after_exit=False`` 退出, 不会取消运行中的 task;
            取消 pending command 是 action loop 的责任 — 由本函数显式触发 shell.clear.
            """
            if not action.is_aborted():
                return False
            await shell.clear()
            return True

        try:
            async with interpreter:
                try:
                    first_delta = True
                    # ── 阶段 1: feed — 流式送入 ──
                    async for delta in action.logos():
                        if first_delta:
                            logger.debug("action loop received first logos delta")
                            first_delta = False
                        interpreter.feed(delta)
                    interpreter.commit()
                    logger.debug("logos stream committed, waiting compile")
                    # ── 阶段 2: wait compiled — 等待编译完成 ──
                    await interpreter.wait_compiled()
                    # 通知编译已经完成.
                    action.set_compiled()
                except InterpretError as err:
                    # 级别 1: 可管理中断. interpretation 已保留 partial results +
                    # observe=True. 同步产出到 output 总线.
                    self._on_mindflow_error(err)
                    # 编译的错误, 直接退出 thinking.
                    action.abort_thinking()
                    return
                except StatementExitedException:
                    # feed 阶段 action 退出 (action.logos 抛 ActionExitedException):
                    # 仍需 clear 取消已启动的命令.
                    await _abort_clear()
                    return

                # 阶段 2 后检查 abort — 取消已解析但未执行完的命令.
                if await _abort_clear():
                    return

                # ── 阶段 3: wait stopped — 等待执行完成 ──
                task = asyncio.create_task(interpreter.wait_stopped())
                # 等待任务结束.
                await action.wait_until_done(task)

                # 阶段 3 后检查 abort — 兜底 execute 阶段的中断.
                if await _abort_clear():
                    return
        except asyncio.CancelledError:
            raise
        finally:
            # fire and forget
            asyncio.create_task(self._post_action_refresh())
            asyncio.create_task(self._fire_interpreter_result(interpretation))

    async def _post_action_refresh(self) -> None:
        """fire-and-forget refresh, 内部捕获异常防 task 静默崩溃.
        未来时序敏感点会加统一关键字 trace, 这里只做 warning 兜底.
        """
        try:
            await self._refresh_shell()
        except Exception as e:
            self.logger.warning(
                "%r post-action refresh_metas failed: %s",
                self,
                e,
                exc_info=True,
            )
