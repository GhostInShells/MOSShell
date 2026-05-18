import contextlib
import asyncio

import janus
from typing_extensions import Self

from ghoshell_moss.core.blueprint.host import GhostRuntime, MossRuntime
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Mindflow, Articulator, Action
from ghoshell_moss.message import Message

__all__ = ["GhostRuntimeImpl"]


class GhostRuntimeImpl(GhostRuntime):
    """GhostRuntime 默认实现 — 编排 MossRuntime + Ghost 生命周期.

    wiring 顺序:
        1. 预注入 ghost providers → container
        2. MossRuntime.__aenter__ (matrix → shell → apps)
        3. GhostMeta.factory(container) → ghost
        4. ghost.__aenter__
        5. Mindflow 解析 + nuclei 注册 + 三循环托管给 matrix.create_task
    """

    def __init__(self, *, moss_runtime: MossRuntime, ghost_meta: GhostMeta):
        if moss_runtime.is_running():
            raise RuntimeError(
                "MossRuntime already started. "
                "Pass a not-yet-entered instance — GhostRuntime owns the lifecycle."
            )
        self._moss_runtime = moss_runtime
        self._ghost_meta = ghost_meta
        self._ghost_instance: Ghost | None = None
        self._mindflow: Mindflow | None = None
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False

        # 三循环队列: main loop → (articulate, action)
        self._articulate_queue: janus.Queue[Articulator] = janus.Queue()
        self._action_queue: janus.Queue[Action] = janus.Queue()

        # logos 缓冲: articulate loop 产出 → task done 检查点发射
        self._logos_buffer: list[str] = []

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

    # ── 生命周期 ──────────────────────────────────

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("GhostRuntime already started")

        container = self._moss_runtime.container

        # 1. 预注入 ghost providers → container
        for provider in self._ghost_meta.providers():
            container.register(provider)

        # 2. MossRuntime.__aenter__
        await self._async_exit_stack.enter_async_context(self._moss_runtime)

        # 3. GhostMeta.factory(container) → ghost
        self._ghost_instance = self._ghost_meta.factory(container)

        # 4. ghost.__aenter__
        await self._async_exit_stack.enter_async_context(self._ghost_instance)

        # 5. Mindflow wiring
        await self._wire_mindflow()

        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._started = False
        await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        self._moss_runtime.close()

    # ── Mindflow wiring ───────────────────────────

    async def _wire_mindflow(self) -> None:
        ghost = self._ghost_instance
        matrix = self._moss_runtime.matrix
        container = matrix.container

        # 解析: ghost.mindflow() > IoC > BaseMindflow
        mindflow = ghost.mindflow()
        if mindflow is None:
            mindflow = container.get(Mindflow)
        if mindflow is None:
            from ghoshell_moss.core.mindflow.base_mindflow import BaseMindflow
            mindflow = BaseMindflow(logger=matrix.logger)

        container.set(Mindflow, mindflow)

        # 注册 nuclei — 从 meta 工厂生成，add 到 mindflow
        for factory in self._ghost_meta.nuclei_metas():
            nucleus = factory.factory(container)
            await mindflow.add_nucleus(nucleus)

        self._mindflow = mindflow
        await self._async_exit_stack.enter_async_context(mindflow)

        # session signal → mindflow 路由.
        # zenoh 存活周期比 ghost/mindflow 长, 关闭期间 session 仍可能收到信号,
        # 所以闭包内检查 mindflow.is_running() 做兜底丢弃.
        def _route_signal(signal):
            if mindflow.is_running():
                mindflow.add_signal(signal)

        matrix.session.on_signal(_route_signal)

        # 三循环托管给 matrix
        matrix.create_task(self._main_loop())
        matrix.create_task(self._articulate_loop())
        matrix.create_task(self._action_loop())

    # ── 三循环 ────────────────────────────────────

    async def _main_loop(self) -> None:
        """mindflow.loop() → Attention → (Articulator, Action) → queues."""
        await self._mindflow.wait_started()
        try:
            async for attention in self._mindflow.loop():
                async with attention:
                    async for articulate, action in attention.loop():
                        self._articulate_queue.sync_q.put_nowait(articulate)
                        self._action_queue.sync_q.put_nowait(action)
        finally:
            self._articulate_queue.shutdown(immediate=True)
            self._action_queue.shutdown(immediate=True)

    async def _articulate_loop(self) -> None:
        """queue → ghost.articulate(articulator) → send_nowait + session.output.

        output 时序:
          - articulator 入队 → output('moment', log=...)   ghost 感知到了什么
          - 开始 articulate   → output('logos', log='articulating')
          - deltas 缓冲       → task done 检查点由 action loop 发射
        """
        ghost = self._ghost_instance
        mindflow = self._mindflow
        session = self._moss_runtime.session
        try:
            while mindflow.is_running():
                articulator = await self._articulate_queue.async_q.get()
                async with articulator:
                    moment = articulator.moment
                    session.output(
                        'moment',
                        log=f"moment {moment.id}: {len(moment.percepts)} percepts",
                    )
                    session.output('logos', log='articulating')
                    self._logos_buffer.clear()
                    try:
                        async for delta in ghost.articulate(articulator):
                            articulator.send_nowait(delta)
                            self._logos_buffer.append(delta)
                    except Exception as e:
                        session.output('error', log=f"articulate error: {e}")
        except janus.AsyncQueueShutDown:
            pass

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
        try:
            while mindflow.is_running():
                action = await self._action_queue.async_q.get()
                async with action:
                    messages, observe = await self._stream_execute(action)
                    action.outcome(*messages, observe=observe)
        except janus.AsyncQueueShutDown:
            pass

    async def _stream_execute(self, action) -> tuple[list, bool]:
        """流式执行: action.received_logos() → interpreter.feed(delta) → 结算.

        返回 (status_messages, observe) 闭合 observe 回路.
        task done 检查点按序发射 logos / command-output / command-result.
        InterpretError 被捕获 — interpretation 已保留 partial results.
        """
        from ghoshell_moss.core.concepts.errors import InterpretError
        from ghoshell_moss.core.concepts.command import CommandTask

        shell = self._moss_runtime.shell
        interpreter = await shell.interpreter(kind='clear', clear_after_exit=False)
        interpretation = interpreter.interpretation()
        logger = self._moss_runtime.matrix.logger
        session = self._moss_runtime.session

        # ── task 级实时输出 ──
        # output 不支持流式粘包, 利用 task done 检查点发射已缓冲 logos
        def _on_task_done(task: CommandTask) -> None:
            result = task.task_result()
            caller = task.caller_name()

            # logos: 检查点发射 articulate buffer
            buffered = list(self._logos_buffer)
            self._logos_buffer.clear()
            if buffered:
                full_text = ''.join(buffered)
                session.output('logos', Message.new(tag='logos').with_content(full_text))

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

        async with interpreter:
            try:
                # ── 阶段 1: feed — 流式送入 ──
                first_delta = True
                async for delta in action.received_logos():
                    if first_delta:
                        logger.debug("action loop received first logos delta")
                        first_delta = False
                    interpreter.feed(delta)

                # ── 阶段 2: compile — 标记结束, 等待解析完成 ──
                interpreter.commit()
                logger.debug("logos stream committed, waiting compile")
                await interpreter.wait_compiled()

                # ── 阶段 3: execute — 等待全部 task 执行完毕 ──
                await interpreter.wait_stopped()

                # final drain: task 全部完成后剩余 logos
                remaining = list(self._logos_buffer)
                self._logos_buffer.clear()
                if remaining:
                    session.output(
                        'logos',
                        Message.new(tag='logos').with_content(''.join(remaining)),
                    )

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
        status = interpretation.status_messages()
        session.output('system', *status)
        logger.info(
            "interpreter settled: compiled=%d done=%d failed=%d cancelled=%d observe=%s",
            len(interpretation.compiled_tasks),
            len(interpretation.success_tasks),
            len(interpretation.failed_tasks),
            len(interpretation.cancelled_tasks),
            interpretation.observe,
        )
        return status, interpretation.observe
