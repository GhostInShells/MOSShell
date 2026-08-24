import asyncio
import contextlib
from typing import Callable, Type

import janus
from typing_extensions import Self

from ghoshell_moss.core.blueprint.host import IGhostRuntime, IShellRuntime, LoopHealth, LoopStatus, SafeMode
from ghoshell_moss.host.pause_controller import PauseController
from ghoshell_moss.host.safe_mode import SafeModeImpl
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, Thinking, Action, Signal, StatementExitedException,
    ActionGate
)
from ghoshell_moss.core.blueprint.moment import Moment
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.concepts.errors import FatalError, InterpretError
from ghoshell_moss.core.concepts.shell import InterpreterKind
from ghoshell_moss.core.concepts.interpreter import Interpretation
from ghoshell_moss.message import Message
import pathlib

__all__ = ["GhostRuntimeImpl"]

_Observe = bool


class GhostRuntimeImpl(IGhostRuntime):
    """GhostRuntime 默认实现 — 编排 MossRuntime + Ghost 生命周期.

    wiring 顺序:
        1. 预注入 ghost providers → container
        2. MossRuntime.__aenter__ (matrix → shell → apps)
        3. GhostMeta.factory(container) → ghost
        4. ghost.__aenter__
        5. Mindflow 解析 + nuclei 注册 + 三循环托管给 matrix.create_task
    """

    def __init__(
            self,
            *,
            moss_runtime: IShellRuntime,
            ghost_meta: GhostMeta,
            source_path: pathlib.Path | None,
    ):
        if moss_runtime.is_running():
            raise RuntimeError(
                "MossRuntime already started. "
                "Pass a not-yet-entered instance — GhostRuntime owns the lifecycle."
            )
        self._moss_runtime = moss_runtime
        # todo: 未来迁移到 config type 中.
        # 0.5s 是 refresh_metas 的 freshness 窗口 — 人类感知阈值内,
        # 同时大于典型 articulate 首句时长 (保证 action 出口预热在下一轮
        # articulator 入口时命中 _refresh_meta_stale_time). 慢通道理论上应自行改推模式,
        # 不该让 0.5s 阈值承担其延迟.
        self._default_shell_prepare_timeout: float = 0.5
        self._refresh_meta_stale_time: float = 1.0
        self._source_path = source_path
        self._ghost_meta = ghost_meta
        self._ghost_instance: Ghost | None = None
        self._mindflow: Mindflow | None = None
        self._pause_ctrl = PauseController()
        self._safe_mode: SafeModeImpl | None = None  # 懒加载, 未开启时零开销
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        # 启动前注册的观察回调 — __aenter__ (matrix 就绪后) 优先装线到 session.
        self._output_listeners: list[Callable] = []
        self._signal_listeners: list[Callable] = []
        self._loop_status: LoopHealth = LoopHealth(
            mindflow="not_started",
            thinking="not_started",
            action="not_started",
        )

        self._shell_trajectory: MShellTrajectory | None = None
        self._log_prefix: str = f"<GhostRuntime cls={self.__class__} ghost={ghost_meta.name()} mode={self._moss_runtime.mode.name}>"

    # ── GhostRuntime ABC ──────────────────────────

    @property
    def moss(self) -> IShellRuntime:
        return self._moss_runtime

    @property
    def ghost(self) -> Ghost:
        if self._ghost_instance is None:
            raise RuntimeError("Ghost not started. Call __aenter__ first.")
        return self._ghost_instance

    @property
    def meta(self) -> GhostMeta:
        return self._ghost_meta

    @property
    def mindflow(self) -> Mindflow:
        if self._mindflow is None:
            raise RuntimeError("GhostRuntime not started. Call __aenter__ first.")
        return self._mindflow

    def is_running(self) -> bool:
        return self._started

    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        """注册 output 监听 — 生命周期无关. 启动前缓冲, 启动后直挂 session."""
        if self._started:
            self._moss_runtime.session.on_output(callback)
        else:
            self._output_listeners.append(callback)

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """注册 signal 监听 — 生命周期无关. 语义同 on_output."""
        if self._started:
            self._moss_runtime.session.on_signal(callback)
        else:
            self._signal_listeners.append(callback)

    # ── 生命周期 ──────────────────────────────────

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("GhostRuntime already started")

        container = self._moss_runtime.container
        logger = self.moss.logger

        # 1. 预注入 ghost providers → container
        logger.debug("%s step 1/5: registering ghost providers", self._log_prefix)
        for provider in self._ghost_meta.providers():
            container.register(provider)
        # 校验 IoC 容器中注册依赖是否能满足 Ghost 的需要.
        self._ghost_meta.contracts().validate(container)

        # 2. MossRuntime.__aenter__ (Matrix 从 IoC 注入 LoggerItf 或 fallthrough 到 project.logger)
        logger.debug("%s step 2/5: entering MossRuntime", self._log_prefix)
        await self._async_exit_stack.__aenter__()
        await self._async_exit_stack.enter_async_context(self._moss_runtime)
        # 默认注册 shell trajectory.
        self._shell_trajectory = MShellTrajectory(self._moss_runtime.shell)
        await self._async_exit_stack.enter_async_context(self._shell_trajectory)
        self._moss_runtime.container.set(MShellTrajectory, self._shell_trajectory)

        logger = self.moss.logger

        # 2.5: 装线启动前注册的观察回调 (matrix 已就绪) — 优先装, 先于 ghost.__aenter__,
        # 才能捕获 ghost 启动阶段 (stubs sync / dsh 启动) 发出的 output/signal.
        for callback in self._output_listeners:
            self._moss_runtime.session.on_output(callback)
        self._output_listeners.clear()
        for callback in self._signal_listeners:
            self._moss_runtime.session.on_signal(callback)
        self._signal_listeners.clear()

        # 3. GhostMeta.factory(container) → ghost
        logger.debug("%s step 3/5: building ghost instance", self._log_prefix)
        self._ghost_instance = self._ghost_meta.factory(container)

        # 4. ghost.__aenter__
        logger.debug("%s step 4/5: entering ghost", self._log_prefix)
        await self._async_exit_stack.enter_async_context(self._ghost_instance)

        # 5. Mindflow wiring
        logger.debug("%s step 5/5: wiring mindflow", self._log_prefix)
        await self._wire_mindflow()

        # 急停级联控制器 — mindflow 和 shell 都已就绪
        self._pause_ctrl.bind(self._mindflow, self.moss.shell)

        self._started = True
        logger.info("%s started", self._log_prefix)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._started = False
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self.moss.logger.exception(
                "%s error during teardown: %s", self._log_prefix, e
            )

    def is_paused(self) -> bool:
        return self._pause_ctrl.is_paused()

    def pause(self, toggle: bool = True, callback: Callable[[], None] | None = None) -> None:
        """急停 — 幂等, 设值. callback 在级联完成后同步 fire (done 语义).

        PauseController 负责状态机 + mindflow/shell 级联.
        callback 必须自行保证线程安全 (可能跨 loop 或跨线程调用).
        """
        self._pause_ctrl.pause(toggle)
        if callback:
            callback()

    def safe_mode(self) -> SafeMode:
        """SafeMode 懒加载单例. 未开启时零开销 — 首次调用才实例化 SafeModeImpl."""
        if self._safe_mode is None:
            self._safe_mode = SafeModeImpl()
        return self._safe_mode

    def close(self) -> None:
        logger = self.moss.logger
        logger.debug("%s closing moss runtime", self._log_prefix)
        self._moss_runtime.close()
        if self._mindflow is not None:
            logger.debug("%s closing mindflow", self._log_prefix)
            self._mindflow.close()
        logger.debug("%s closed", self._log_prefix)

    def inspect_loop_health(self) -> LoopHealth:
        return self._loop_status.copy()

    # ── Mindflow wiring ───────────────────────────

    def _collect_nuclei_manifests(self):
        """收集 matrix 和 mode 两层的 manifests, 用于 nuclei 发现."""
        # matrix 层
        try:
            yield self._moss_runtime.matrix.project.project_manifests()
        except Exception:
            self.moss.logger.exception(
                "%s failed to load matrix manifests, skipping matrix nuclei",
                self._log_prefix,
            )
        # mode 层
        try:
            yield self._moss_runtime.mode.manifests()
        except Exception:
            self.moss.logger.exception(
                "%s failed to load mode manifests, skipping mode nuclei",
                self._log_prefix,
            )

    async def _wire_mindflow(self) -> None:
        ghost = self._ghost_instance
        matrix = self._moss_runtime.matrix
        container = matrix.container

        # 解析: ghost.mindflow() > IoC > new_default_mindflow()
        mindflow = ghost.mindflow()
        if mindflow is None:
            mindflow = container.get(Mindflow)
        if mindflow is None:
            from ghoshell_moss.core.mindflow import new_default_mindflow
            mindflow = new_default_mindflow(logger=self.moss.logger)

        container.set(Mindflow, mindflow)

        nuclei_factories = {}
        # 从 matrix manifests 和 mode manifests 一起收集 nuclei
        for manifests in self._collect_nuclei_manifests():
            if manifests is None:
                continue
            for nucleus_manifest in manifests.nuclei():
                if nucleus_manifest.is_error():
                    self.moss.logger.warning(
                        "%s skip nucleus manifest with error: %s (%s)",
                        self._log_prefix, nucleus_manifest.name(), nucleus_manifest.error(),
                    )
                    continue
                nuclei_factories[nucleus_manifest.name()] = nucleus_manifest.value()

        # 注册 nuclei — 从 meta 工厂生成，add 到 mindflow
        for ghost_nucleus_factory in self._ghost_meta.nuclei_metas():
            nuclei_factories[ghost_nucleus_factory.name] = ghost_nucleus_factory

        for nucleus_meta in nuclei_factories.values():
            try:
                nucleus = nucleus_meta.factory(container)
            except NotImplementedError:
                self.moss.logger.warning(
                    "%s nucleus %s is a stub (NotImplementedError), skipping",
                    self._log_prefix, nucleus_meta.name(),
                )
                continue
            except Exception:
                self.moss.logger.exception(
                    "%s failed to create nucleus %s, skipping",
                    self._log_prefix, nucleus_meta.name(),
                )
                continue
            mindflow.with_nucleus(nucleus, override=True)

        self._mindflow = mindflow
        await self._async_exit_stack.enter_async_context(mindflow)

        # session signal → mindflow 路由.
        # zenoh 存活周期比 ghost/mindflow 长, 关闭期间 session 仍可能收到信号,
        # 所以闭包内检查 mindflow.is_running() 做兜底丢弃.
        def _route_signal_to_mindflow(signal: Signal):
            if mindflow.is_running():
                mindflow.add_signal(signal)

        # mindflow 装线 shell trajectory, 注册观测函数.
        def _on_moments_observing(moment: Moment) -> None:
            frame = self._shell_trajectory.pop_frame()
            if moment.previous is not None:
                # 将轨迹中保存的数据作为 result 插入.
                messages = frame.project(with_dynamic=False)
                moment.previous.add_result(messages, frame.need_observe)
                moment.previous.need_observe = frame.need_observe

        def _notify_moments_need_observe(e):
            # 仅仅通知观测应该发生. 真实的观测数据, 会在 moment 创建时回调构建.
            mindflow.moments.add_result([], need_observe=True)

        def _shell_trajectory_epoch_refresh():
            return [
                Message.new(tag='moss-full-facade', timestamp=True).with_content(
                    self._shell_trajectory.epoch_start_point()
                )
            ]

        # 注册回调, 当发生 observe 事件时, 通知 mindflow observer.
        self._shell_trajectory.when_need_observe(_notify_moments_need_observe)
        # 注册回调, 当 observer 触发观察动作时, 更新数据.
        mindflow.moments.when_moment_created(_on_moments_observing)
        mindflow.moments.with_epoch_recap("ShellTrajectoryEpoch", _shell_trajectory_epoch_refresh)

        # 三循环托管给 matrix
        matrix.create_task(self._mindflow_loop(), stop_matrix_on_error=True)
        matrix.create_task(self._thinking_loop(), stop_matrix_on_error=True)
        matrix.create_task(self._action_loop(), stop_matrix_on_error=True)
        # 等待应该发生在循环外侧.
        await self._mindflow.wait_started()
        # ignore any signals before started
        matrix.session.on_signal(_route_signal_to_mindflow)

    # ── 三循环 ────────────────────────────────────

    async def _mindflow_loop(self) -> None:
        """mindflow.loop() → Attention → (Articulator, Action) → queues."""
        status: LoopStatus = 'running'
        self._loop_status["mindflow"] = status
        try:
            async with self._mindflow:
                await self._mindflow.wait_close()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._moss_runtime.logger.exception("%s mindflow loop error: %s", self._log_prefix, e)
        finally:
            status = 'stopped'
            self._loop_status["mindflow"] = status

    async def _thinking_loop(self) -> None:
        """queue → ghost.articulate(articulator) → send_nowait + pub_logos.

        output 时序:
          - articulator 入队 → output('moment', log=...)  ghost 感知到了什么
          - delta 产出       → pub_logos(delta)           实时流, 外部通过 get_logos() 消费
          - 结束 (成功/失败) → ghost.on_articulate_exit()  调试附着点
        """
        mindflow = self._mindflow
        await mindflow.wait_started()
        # 组装 mindflow channel.
        if channel := self._mindflow.as_channel():
            self.moss.shell.main_channel.add_virtual_channel(channel)
        # 首次要刷新, 获取关键帧.
        await self.moss.shell.refresh_metas()
        status: LoopStatus = 'running'
        self._loop_status["thinking"] = status
        try:
            while mindflow.is_running():
                async for thinking in self._mindflow.thinking_loop():
                    await self._run_thinking(thinking)
        finally:
            status = 'stopped'
            self._loop_status["thinking"] = status

    async def _run_thinking(self, thinking: Thinking) -> None:
        try:
            # 每次开始运行时必须刷新, 但如果刚刚刷新过, 在阈值内, 可以快速跳过.
            prepare_timeout = self._default_shell_prepare_timeout
            session = self._moss_runtime.session
            await self.moss.shell.refresh_metas(prepare_timeout, stale_time=self._refresh_meta_stale_time)
            # 启动 thinking 生命周期.
            async with thinking:
                # 获取 moment 的首帧. 应该在它生成首帧时, 才会刷新汲取数据.
                moment = thinking.moment
                if moment.previous_stop_reason():
                    # 被强行中断的时候, 需要清空 shell.
                    await self._moss_runtime.shell.clear()
                tasks = []
                gated_mode = self._safe_mode is not None and self._safe_mode.is_enabled()
                if gated_mode:
                    gate = thinking.gate()
                    gate_task = asyncio.create_task(self._run_action_gate(thinking, gate))
                    tasks.append(gate_task)

                moment = thinking.moment
                # 发送已经执行的命令.
                if moment.command_logos:
                    # 发送 command logos 作为第一波.
                    async with thinking.articulator(replan=False, wait_action_done=False) as articulator:
                        command_logos = moment.command_logos
                        articulator.send_nowait(command_logos)
                        session.pub_logos(command_logos)
                        await articulator.wait_compiled()
                # 发送首帧提示.
                session.output(
                    'moment',
                    *moment.percepts_messages(),
                    log=f"moment {moment.id}: {len(moment.percepts)} percepts",
                )
                # 如果
                if thinking.effort() == 'none':
                    return

                # -- 需要阻塞执行完的逻辑完成 -- #

                fut = asyncio.create_task(self._run_thinking_with_ghost(thinking))
                tasks.append(fut)
                # 同步阻塞到结束. 如果 thinking 提前结束, 也会中断所有的 tasks.
                await thinking.wait_until_done(*tasks)

        except FatalError:
            self.moss.logger.exception("%s articulate fatal error", self._log_prefix)
            raise
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.moss.logger.exception("%s articulate loop error: %s", self._log_prefix, e)

    async def _run_action_gate(self, thinking: Thinking, gate: ActionGate) -> None:
        try:
            moment = thinking.moment
            # 注入动态数据.
            moment.with_dynamic_context(
                'safemode',
                [Message.new().with_content(
                    "<safemode-active>\n"
                    "Gate active on thinking→action path. Your logos is "
                    "reviewed by a human before dispatch. Rejected logos will "
                    "NOT be executed by the body; only your utterance stays "
                    "in your own history. Feedback arrives next frame as "
                    "<safemode-approval-note> or <safemode-rejection>.\n"
                    "</safemode-active>"
                )],
            )
            while thinking.is_running():
                request = await gate.wait_request()
                if request is None:
                    # 运行结束.
                    break
                # 提交完整 logos 给 SafeMode gate, 等 TUI 裁决.
                verdict_future = self._safe_mode.submit(request.logos)

                async def _await_gate_verdict():
                    return await asyncio.wrap_future(verdict_future)

                verdict = await _await_gate_verdict()
                if verdict.kind == 'approved':
                    # 回放 buffered logos → 走原来的 send_nowait 路径.
                    await request.approve(
                        "<safemode-approval-note>\n"
                        "Previous logos approved and executed. Human note:\n"
                        f"{verdict.message}\n"
                        "</safemode-approval-note>"
                    )
                elif verdict.kind == 'rejected':
                    # 否决反馈: 不回传被拒 logos (ghost 自己的 history 已有),
                    # 只标记事实 + 理由, 靠 ghost 自身消化. 同时会强制退出 Action.
                    await request.reject(
                        "<safemode-rejection>\n"
                        "Previous logos rejected by human review; body did not execute.\n"
                        f"Reason: {verdict.message}\n"
                        "</safemode-rejection>"
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.moss.logger.error("%s action gate error: %s", self._log_prefix, e)
            raise e
        finally:
            # 幂等: 已被 approve/reject 结算时 no-op; abort 兜底.
            if self._safe_mode is not None:
                self._safe_mode.cancel_current()

    async def _run_thinking_with_ghost(self, thinking: Thinking) -> None:
        session = self._moss_runtime.session
        ghost = self._ghost_instance

        logos_parts: list[str] = []
        error: Exception | None = None
        try:
            # 将权限移交给 ghost.
            async for delta in ghost.think(thinking):
                session.pub_logos(delta)
                logos_parts.append(delta)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = e
            self.moss.logger.exception("%s articulate error: %s", self._log_prefix, e)
            session.output('error', log=f"articulate error: {e}")
            raise e
        finally:
            logos = "".join(logos_parts)
            if logos:
                # logos 同时发一条原子 OutputItem — headless 观测面 (on_output)
                # 看不到 stream (get_logos), 这里补上 articulate 返回值的可观测出口.
                session.output('logos', logos)
                session.pub_logos("\n\n")
            # 回调感知, ack 事件发生.
            ghost.on_thinking_exit(
                thinking,
                logos,
                error,
            )

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
        mindflow = self._mindflow
        await mindflow.wait_started()
        last_task = None
        try:
            status: LoopStatus = 'running'
            self._loop_status["action"] = status
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
        except Exception as e:
            self.moss.logger.exception("%s action loop error: %s", self._log_prefix, e)
        finally:
            if last_task is not None and not last_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await last_task
            status = 'stopped'
            self._loop_status["action"] = status

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
            self.moss.logger.exception("%s action fatal error", self._log_prefix)
            # 异常退出.
            self._moss_runtime.close()
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 未捕获的异常都要导致停止.
            self.moss.logger.exception(
                "%s action loop error: %s", self._log_prefix, e, exc_info=True,
            )
            # 未兜底异常需要清空状态.
            await self._moss_runtime.shell.clear()

    async def _post_action_refresh(self) -> None:
        """fire-and-forget refresh, 内部捕获异常防 task 静默崩溃.
        未来时序敏感点会加统一关键字 trace, 这里只做 warning 兜底.
        """
        try:
            await self.moss.shell.refresh_metas(self._default_shell_prepare_timeout)
        except Exception:
            self.moss.logger.warning(
                "%s post-action refresh_metas failed",
                self._log_prefix,
                exc_info=True,
            )

    async def _run_interpreter_with_action(self, action: Action) -> None:
        """流式执行: action.received_logos() → interpreter.feed(delta) → 结算.

        返回 (as_messages, observe) 闭合 observe 回路.
        logos 已走 session stream 实时广播, 此处只发射 command-output/result.
        InterpretError 被捕获 — interpretation 已保留 partial results.

        Attention abort 传播: 在 feed/compile/execute 各阶段结束后检查
        action.is_aborted(), 发现后调用 shell.clear() 取消 pending command,
        返回部分结果.
        """
        shell = self._moss_runtime.shell
        if not shell.is_running():
            self.moss.logger.error(
                "%s ghost runtime received action but shell is not running",
                self._log_prefix,
            )
            self.moss.session.output('error', 'received action but shell is not running')
            return

        if action.replaned:
            kind: InterpreterKind = 'clear'
        else:
            kind: InterpreterKind = 'append'

        interpreter = await shell.interpreter(kind=kind, clear_after_exit=False)
        interpretation = interpreter.interpretation()

        logger = self.moss.logger
        session = self._moss_runtime.session

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
                    session.output('error', log=str(err))
                    # 编译的错误, 直接退出当前 Action.
                    return
                except StatementExitedException:
                    # 正常退出运行.
                    return

                # ── 阶段 3: wait stopped — 等待执行完成 ──
                task = asyncio.create_task(interpreter.wait_stopped())
                # 等待任务结束.
                await action.wait_until_done(task)
        except asyncio.CancelledError:
            raise
        finally:
            # fire and forget
            asyncio.create_task(self._post_action_refresh())
            asyncio.create_task(self._fire_interpreter_result(interpretation))

    async def _fire_interpreter_result(
            self,
            interpretation: Interpretation
    ) -> None:
        try:
            self._moss_runtime.session.output(
                'system',
                *interpretation.as_messages(),
                log=f"after interpreter {interpretation.id}",
            )
            self._moss_runtime.logger.info(
                "%s interpreter settled: %s",
                self._log_prefix,
                interpretation.id
            )
        except Exception as e:
            self.moss.logger.error("%s send interpreter frame failed: %s", self._log_prefix, e)
