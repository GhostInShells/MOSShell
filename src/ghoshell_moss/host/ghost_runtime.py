import asyncio
import contextlib
from typing import Type

import janus
from typing_extensions import Self

from ghoshell_moss.core.blueprint.host import GhostRuntime, MossRuntime, LoopHealth, LoopStatus
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta, GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import Mindflow, Articulator, Action, Signal
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.core.concepts.command import CommandTask
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss.message import Message
import pathlib

__all__ = ["GhostRuntimeImpl"]

_Observe = bool


class GhostRuntimeImpl(GhostRuntime):
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
            moss_runtime: MossRuntime,
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
        # articulator 入口时命中 stale_time). 慢通道理论上应自行改推模式,
        # 不该让 0.5s 阈值承担其延迟.
        self._default_shell_prepare_timeout: float = 0.5
        self._refresh_meta_stale_time: float = 0.5
        self._source_path = source_path
        self._ghost_meta = ghost_meta
        self._ghost_instance: Ghost | None = None
        self._mindflow: Mindflow | None = None
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        self._loop_status: LoopHealth = LoopHealth(
            main="not_started",
            articulate="not_started",
            action="not_started",
        )

        # 三循环队列: main loop → (articulate, action)
        self._articulate_queue: janus.Queue[Articulator] = janus.Queue()
        self._action_queue: janus.Queue[Action] = janus.Queue()
        self._log_prefix: str = f"<GhostRuntime cls={self.__class__} ghost={ghost_meta.name()} mode={self._moss_runtime.mode}>"

    # ── GhostRuntime ABC ──────────────────────────

    @property
    def moss(self) -> MossRuntime:
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
        if not container.bound(GhostWorkspace):
            container.register(GhostWorkspaceProvider(self._source_path))

        # 2. MossRuntime.__aenter__ (Matrix 从 IoC 注入 LoggerItf 或 fallthrough 到 env.logger)
        logger.debug("%s step 2/5: entering MossRuntime", self._log_prefix)
        await self._async_exit_stack.__aenter__()
        await self._async_exit_stack.enter_async_context(self._moss_runtime)
        logger = self.moss.logger

        # 3. GhostMeta.factory(container) → ghost
        logger.debug("%s step 3/5: building ghost instance", self._log_prefix)
        self._ghost_instance = self._ghost_meta.factory(container)

        # 4. ghost.__aenter__
        logger.debug("%s step 4/5: entering ghost", self._log_prefix)
        await self._async_exit_stack.enter_async_context(self._ghost_instance)

        # 5. Mindflow wiring
        logger.debug("%s step 5/5: wiring mindflow", self._log_prefix)
        await self._wire_mindflow()

        self._started = True
        # todo: hook — GhostRuntimeLifecycleHook.on_started(self)
        logger.info("%s started", self._log_prefix)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # todo: hook — GhostRuntimeLifecycleHook.on_stopping(self, exc_type, exc_val)
        self._started = False
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            self.moss.logger.exception(
                "%s error during teardown", self._log_prefix,
            )
        # todo: hook — GhostRuntimeLifecycleHook.on_stopped(self)

    def pause(self, toggle: bool = True) -> None:
        if self._mindflow is not None:
            self._mindflow.pause(toggle)

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
        for env_nucleus_info in self.moss.matrix.manifests.nuclei().values():
            nuclei_factories[env_nucleus_info.name] = env_nucleus_info.nucleus_meta

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

        # 三循环托管给 matrix
        matrix.create_task(self._main_loop(), stop_matrix_on_error=True)
        matrix.create_task(self._articulate_loop(), stop_matrix_on_error=True)
        matrix.create_task(self._action_loop(), stop_matrix_on_error=True)
        # 等待应该发生在循环外侧.
        await self._mindflow.wait_started()
        # ignore any signals before started
        matrix.session.on_signal(_route_signal_to_mindflow)

    # ── 三循环 ────────────────────────────────────

    def _moss_dynamic_messages(self) -> list[Message]:
        shell = self._moss_runtime.shell
        # 闭包在 shell running 时才取，shell 未启动时返回空列表.
        if shell.is_running():
            return shell.dynamic_messages()
        return []

    async def _main_loop(self) -> None:
        """mindflow.loop() → Attention → (Articulator, Action) → queues."""
        status: LoopStatus = 'running'
        self._loop_status["main"] = status
        try:
            async for attention in self._mindflow.loop():
                # per-attention 注册: ghost runtime 决定绑什么上下文.
                # mindflow 级注册留作将来更高层治理 (如多 ghost 共享 mindflow) 时设计.
                try:
                    impulse = attention.draw_from()
                    # 实现 interrupt 协议: 停止所有执行中的 logos.
                    # shell.clear() 是 stop_interpretation 的超集 —
                    # 关闭当前 interpreter + 清空 speech 缓冲 + 取消 runtime tree
                    # 上 pending 的 command tasks. 单调 stop_interpretation 只
                    # 关 interpreter, 留下半截状态.
                    if impulse.interrupt:
                        await self.moss.shell.clear()
                    async with attention:
                        async for articulate, action in attention.loop():
                            self._articulate_queue.sync_q.put_nowait(articulate)
                            self._action_queue.sync_q.put_nowait(action)
                except FatalError:
                    self.moss.logger.exception("%s main loop fatal error", self._log_prefix)
                    # todo: hook — MindflowErrorHook.on_fatal(error)
                    raise
                except Exception:
                    self.moss.logger.exception("%s main loop attention error", self._log_prefix)
                    # todo: hook — MindflowErrorHook.on_attention_error(error)
                    # 长时间运行要做异常感知, 而不能轻易破坏生命周期. 继续下一个 attention.
        finally:
            status = 'stopped'
            self._loop_status["main"] = status
            self._articulate_queue.shutdown(immediate=True)
            self._action_queue.shutdown(immediate=True)

    async def _articulate_loop(self) -> None:
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
        status: LoopStatus = 'running'
        self._loop_status["articulate"] = status
        try:
            while mindflow.is_running():
                try:
                    articulator = await self._articulate_queue.async_q.get()
                except janus.AsyncQueueShutDown:
                    break
                try:
                    await self._run_articulator(articulator)
                except FatalError:
                    self.moss.logger.exception("%s articulate fatal error", self._log_prefix)
                    raise
                except Exception:
                    self.moss.logger.exception("%s articulate loop error", self._log_prefix)
                    # 非关键路径异常 (session.output / on_articulate_exit 等). 不中断循环.
        finally:
            status = 'stopped'
            self._loop_status["articulate"] = status

    async def _run_articulator(self, articulator: Articulator) -> None:
        session = self._moss_runtime.session
        ghost = self._ghost_instance
        async with articulator:
            prepare_timeout = self._default_shell_prepare_timeout
            # 每次开始运行时必须刷新.
            await self.moss.shell.refresh_metas(prepare_timeout, stale_time=self._refresh_meta_stale_time)
            moment = articulator.moment
            # 发送已经执行的命令.
            if moment.command_logos:
                articulator.send_nowait(moment.command_logos)

            session.output(
                'moment',
                *moment.as_request_messages(),
                log=f"moment {moment.id}: {len(moment.percepts)} percepts",
            )

            if articulator.thinking_effort() == 'none':
                ghost.on_articulate_exit(articulator, '', None)
                return

            logos_parts: list[str] = []
            error: Exception | None = None
            # 等待刷新结束.
            moment.with_perspective(
                'moss_dynamic',
                self.moss.shell.dynamic_messages(available_only=True, stale_time=self._refresh_meta_stale_time),
            )
            try:
                async for delta in ghost.articulate(articulator):
                    articulator.send_nowait(delta)
                    session.pub_logos(delta)
                    logos_parts.append(delta)
            except Exception as e:
                error = e
                self.moss.logger.exception("%s articulate error: %s", self._log_prefix, e)
                session.output('error', log=f"articulate error: {e}")
            finally:
                logos = "".join(logos_parts)
                articulator.moment.logos = logos
                ghost.on_articulate_exit(
                    articulator,
                    logos,
                    error,
                )
                session.pub_logos("\n\n")

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
        status: LoopStatus = 'running'
        self._loop_status["action"] = status
        try:
            while mindflow.is_running():
                try:
                    action = await self._action_queue.async_q.get()
                except janus.AsyncQueueShutDown:
                    break
                await self._run_action(action)
        finally:
            status = 'stopped'
            self._loop_status["action"] = status

    async def _run_action(self, action: Action) -> None:
        try:
            # todo: hook — ActionHook.on_action_enter(action)
            async with action:
                await action.wait_ready()
                if action.is_aborted():
                    return
                messages, observe = await self._stream_execute(action)
                action.outcome(*messages, observe=observe)
                # 时序契约: action 结束 fire-and-forget refresh_metas,
                # 预热下一轮 articulator 入口的 stale_time 检查.
                # 不 await — 让 action_loop 立即进下一轮.
                asyncio.create_task(self._post_action_refresh())
        except FatalError:
            self.moss.logger.exception("%s action fatal error", self._log_prefix)
            # todo: hook — MindflowErrorHook.on_fatal(error)
            raise
        except Exception:
            self.moss.logger.exception("%s action loop error", self._log_prefix)
            # 非关键路径异常. 不中断循环 — action 是消耗品, 丢掉当前 action 继续.

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

    async def _stream_execute(self, action: Action) -> tuple[list[Message], _Observe]:
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
            return [], False

        interpreter = await shell.interpreter(kind='append', clear_after_exit=False)
        interpretation = interpreter.interpretation()

        logger = self.moss.logger
        session = self._moss_runtime.session

        def _on_task_done(task: CommandTask) -> None:
            result = task.task_result()
            caller = task.caller_name()

            # command-output: 给人的消息
            if result.output:
                session.output('command-output', *result.output, log=f"{caller} output")

            # command-result: 给模型的消息
            msgs = result.as_messages()
            if msgs:
                session.output('command-result', *msgs, log=f"{caller} done")
            else:
                session.output('command-result', log=f"{caller} done")

        interpreter.on_task_done(_on_task_done)

        async def _check_abort_and_clear(phase: str) -> bool:
            """检查 attention abort 并清理 shell. 返回 True 表示已 abort."""
            if not action.is_aborted():
                return False
            logger.info(
                "%s attention aborted during %s, clearing shell",
                self._log_prefix, phase,
            )
            await shell.clear()
            return True

        async with interpreter:
            try:
                # ── 阶段 1: feed — 流式送入 ──
                first_delta = True
                async for delta in action.received_logos():
                    if first_delta:
                        logger.debug("action loop received first logos delta")
                        first_delta = False
                    interpreter.feed(delta)

                # feed 阶段结束即检查: 此时 abort 表示 logos 流被中途截断,
                # 已 fed 的 CTML 可能产生了 pending command, 需要 clear.
                if await _check_abort_and_clear("feed"):
                    return interpretation.as_messages(), interpretation.observe

                # ── 阶段 2: compile — 标记结束, 等待解析完成 ──
                interpreter.commit()
                logger.debug("logos stream committed, waiting compile")
                await interpreter.wait_compiled()

                # compile 后检查: abort 可能发生在解析期间, 已编译的 task
                # 未开始执行但已入队, clear 将它们标记为 INTERRUPTED.
                if await _check_abort_and_clear("compile"):
                    return interpretation.as_messages(), interpretation.observe

                # ── 阶段 3: execute — 等待全部 task 执行完毕 ──
                await interpreter.wait_stopped()

                # execute 后检查: abort 发生在命令执行期间, 未完成的 task
                # 被 clear 取消, 已完成的保留结果.
                if await _check_abort_and_clear("execute"):
                    return interpretation.as_messages(), interpretation.observe

            except InterpretError:
                # 级别 1: 可管理中断. interpretation 已保留 partial results +
                # observe=True. 同步产出到 output 总线.
                err = interpretation.exception or "interpret error"
                session.output('error', log=str(err))
                logger.warning(
                    "interpret error during stream execute: %s",
                    interpretation.exception,
                )

        # __aexit__ 已调 close(), interpretation.done = True
        messages = interpretation.as_messages()
        session.output('system', *interpretation.status_messages())
        logger.info(
            "interpreter settled: compiled=%d done=%d failed=%d cancelled=%d observe=%s",
            len(interpretation.compiled_tasks),
            len(interpretation.success_tasks),
            len(interpretation.failed_tasks),
            len(interpretation.cancelled_tasks),
            interpretation.observe,
        )
        return messages, interpretation.observe


class GhostWorkspaceProvider(Provider[GhostWorkspace]):

    def __init__(self, source_path: pathlib.Path | None) -> None:
        self._source_path = source_path

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[GhostWorkspace]:
        return GhostWorkspace

    def factory(self, con: IoCContainer) -> GhostWorkspace:
        from ghoshell_moss.core.blueprint.matrix import Matrix
        matrix = con.force_fetch(Matrix)
        home_path = matrix.ghost_home.abspath()
        return GhostWorkspace(home=home_path, source=self._source_path)
