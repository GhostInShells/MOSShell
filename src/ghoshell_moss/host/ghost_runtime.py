import asyncio
import contextlib
import logging
from typing import Callable, Iterable

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss import MOSShell, Channel
from ghoshell_moss.core import NucleusMeta
from ghoshell_moss.core.blueprint.host import IGhostRuntime, MOSShellRuntime, LoopHealth, LoopStatus, SafeMode
from ghoshell_moss.host.pause_controller import PauseController
from ghoshell_moss.host.safe_mode import SafeModeImpl
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import (
    Mindflow, Thinking, Signal
)
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.core.concepts.errors import FatalError
from ghoshell_moss.core.concepts.interpreter import Interpretation
from ghoshell_moss.core.mindflow.mindflow_in_shell import MindflowInShell
from ghoshell_moss.message import Message
import pathlib

__all__ = ["GhostRuntimeImpl"]

_Observe = bool


class GhostInShellDrivenByMindflow(IGhostRuntime, MindflowInShell):
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
            moss_runtime: MOSShellRuntime,
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
        self._runtime_channels: dict[str, Channel] = {}

        self._shell_trajectory: MShellTrajectory | None = None
        self._log_prefix: str = f"<GhostRuntime cls={self.__class__} ghost={ghost_meta.name()} mode={self._moss_runtime.mode.name}>"

    def __repr__(self):
        return self._log_prefix

    # ── GhostRuntime ABC ──────────────────────────

    @property
    def moss(self) -> MOSShellRuntime:
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
        return self._started and self._moss_runtime.is_running()

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
        logger.debug("%r step 1/5: registering ghost providers", self)
        for provider in self._ghost_meta.providers():
            container.register(provider)
        # 校验 IoC 容器中注册依赖是否能满足 Ghost 的需要.
        self._ghost_meta.contracts().validate(container)

        # 2. MossRuntime.__aenter__ (Matrix 从 IoC 注入 LoggerItf 或 fallthrough 到 project.logger)
        logger.debug("%r step 2/5: entering MossRuntime", self)
        await self._async_exit_stack.__aenter__()
        # 注册 runtime 自身的系统 channel. 每次刷新时都会更新.

        await self._async_exit_stack.enter_async_context(self._moss_runtime)
        self._moss_runtime.shell.main_channel.build.virtual_children(self._get_runtime_channels)
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
        logger.debug("%r step 3/5: building ghost instance", self)
        self._ghost_instance = self._ghost_meta.factory(container)
        # 注册 ghost 错误观测 (on_error): ghost 内部检测到错误时 fire 回调 → 输出 error 讯息.
        # 先于 ghost.__aenter__ 注册, 使 dsh 启动期 (ghost.__aenter__ 内) 的错误也能被捕获.
        self._ghost_instance.on_error(self._on_ghost_error)

        # 4. ghost.__aenter__
        logger.debug("%r step 4/5: entering ghost", self)
        await self._async_exit_stack.enter_async_context(self._ghost_instance)
        if channel := self._ghost_instance.channel():
            self._runtime_channels['ghost'] = channel

        # 5. Mindflow wiring
        logger.debug("%r step 5/5: wiring mindflow", self)
        await self._wire_mindflow()
        if mindflow_channel := self._mindflow.as_channel():
            self._runtime_channels['mindflow'] = mindflow_channel

        # 急停级联控制器 — mindflow 和 shell 都已就绪
        self._pause_ctrl.bind(self._mindflow, self.moss.shell)

        self._started = True
        logger.info("%r started", self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._started = False
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            self.moss.logger.exception(
                "%s error during teardown: %s", self._log_prefix, e
            )
        self._loop_status["mindflow"] = 'stopped'

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

    def _is_thinking_gated(self) -> bool:
        return self.safe_mode().is_enabled()

    @property
    def logger(self) -> logging.Logger:
        return self._moss_runtime.logger

    @property
    def container(self) -> IoCContainer:
        return self._moss_runtime.container

    @property
    def shell_trajectory(self) -> MShellTrajectory:
        if self._shell_trajectory is None:
            raise RuntimeError("ShellTrajectory not set")
        return self._shell_trajectory

    def _on_mindflow_loop_task(self, future: asyncio.Future, *, name: str | None = None) -> None:
        self._moss_runtime.matrix.create_task(future, stop_matrix_on_error=True, name=name)

    def close(self) -> None:
        logger = self.moss.logger
        logger.debug("%r closing moss runtime", self)
        self._moss_runtime.close()
        if self._mindflow is not None:
            logger.debug("%r closing mindflow", self)
            self._mindflow.close()
            self._loop_status["mindflow"] = 'stopped'
        logger.debug("%r closed", self)

    def inspect_loop_health(self) -> LoopHealth:
        return self._loop_status.copy()

    # ── Mindflow wiring ───────────────────────────

    def _collect_nuclei_metas(self) -> Iterable[NucleusMeta]:
        # 从 matrix manifests 和 mode manifests 一起收集 nuclei
        nuclei_factories = {}
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
        return nuclei_factories.values()

    def _collect_nuclei_manifests(self):
        """收集 matrix 和 mode 两层的 manifests, 用于 nuclei 发现."""
        # project 层
        try:
            yield self._moss_runtime.matrix.project.project_manifests()
        except Exception as e:
            self.moss.logger.exception(
                "%r failed to load matrix manifests, skipping matrix nuclei: %s",
                self, e
            )
            self._send_error(e, 'moss project manifests error')
        # mode 层
        try:
            yield self._moss_runtime.mode.manifests()
        except Exception as e:
            self.moss.logger.exception(
                "%s failed to load mode manifests, skipping mode nuclei: %s",
                self, e
            )
            self._send_error(e, 'moss mode manifest error')

    def _send_error(self, error: Exception | str, log: str = '') -> None:
        if self._moss_runtime.session.is_running:
            self._moss_runtime.session.output('error', str(error), log=log)

    def _on_mindflow_error(self, error: BaseException | str) -> None:
        self._send_error(str(error), 'mindflow-error')

    def _on_ghost_error(self, error: Exception) -> None:
        """ghost 内部错误观测出口 — ghost 经 on_error 注册, 检测到错误时 fire 本回调.

        与 mindflow error 同走 session.output('error'), 但 log 标记 'ghost-error'
        区分错误来源 (ghost 自身 vs mindflow 仲裁).
        """
        self._send_error(error, 'ghost-error')

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
        self._mindflow = mindflow
        await super()._wire_mindflow()
        # mindflow 生命周期绑定到全局 lifecycle, 启动即 running, 终止由 close/__aexit__ 置 stopped.
        self._loop_status["mindflow"] = 'running'

    def _when_signal_added(self, callback: Callable[[Signal], None]):
        self._moss_runtime.session.on_signal(callback)

    async def _enter_async_context(self, manager: contextlib.AbstractAsyncContextManager) -> None:
        await self._async_exit_stack.enter_async_context(manager)

    # ── 三循环 ────────────────────────────────────

    async def _thinking_loop(self) -> None:
        """queue → ghost.articulate(articulator) → send_nowait + pub_logos.

        output 时序:
          - articulator 入队 → output('moment', log=...)  ghost 感知到了什么
          - delta 产出       → pub_logos(delta)           实时流, 外部通过 get_logos() 消费
          - 结束 (成功/失败) → ghost.on_articulate_exit()  调试附着点
        """
        status: LoopStatus = 'running'
        self._loop_status["thinking"] = status
        try:
            await super()._thinking_loop()
        except FatalError as e:
            self._send_error(e, 'thinking-loop-failed')
            self.close()
        finally:
            status = 'stopped'
            self._loop_status["thinking"] = status

    async def _approve_logos(self, logos: str) -> tuple[bool, str]:
        """SafeMode 裁决完整 logos (articulator commit 锁的回调). 返回 (approved, message).

        approved 时若带附言 (approve-with-note), 直接落到 mindflow.moments 轨迹,
        返回值 message 恒为空; rejected 时 message 作为 abort reason, 由 _commit
        abort 掉 action; cancelled (abort 兜底) message 也为空。
        """
        verdict_future = self._safe_mode.submit(logos)
        try:
            verdict = await asyncio.wrap_future(verdict_future)
            if verdict.kind == 'approved':
                # 空 note (纯 Enter 放行) 不带附言 — 区别于 approve-with-note.
                if verdict.message:
                    note = (
                        "<safemode-approval-note>\n"
                        "Previous logos approved and executed. Human note:\n"
                        f"{verdict.message}\n"
                        "</safemode-approval-note>"
                    )
                    self.mindflow.moments.add_result([note])
                return True, ''
            if verdict.kind == 'cancelled':
                # cancel 是 abort 兜底, 不是否决, 无 reason.
                return False, ''
            message = (
                "<safemode-rejection>\n"
                "Previous logos rejected by human review; body did not execute.\n"
                f"Reason: {verdict.message}\n"
                "</safemode-rejection>"
            )
            return False, message
        finally:
            # 幂等: 已被 approve/reject 结算时 no-op; abort/cancel 兜底清理 pending.
            self._safe_mode.cancel_current()

    def _on_logos_delta(self, delta: str) -> None:
        if self._moss_runtime.session.is_running():
            self._moss_runtime.session.pub_logos(delta)

    def _on_logos_end(self) -> None:
        """一段 logos (utterance) 结束 — 发 EOF 哨兵, 消费端据此冲刷尾段."""
        if self._moss_runtime.session.is_running():
            self._moss_runtime.session.pub_logos(end=True)

    def _get_runtime_channels(self) -> dict[str, Channel]:
        return self._runtime_channels

    async def _articulate_from_thinking(self, thinking: Thinking) -> None:
        session = self._moss_runtime.session
        ghost = self._ghost_instance

        logos_parts: list[str] = []
        error: Exception | None = None
        try:
            # 将权限移交给 ghost.
            async for delta in ghost.think(thinking):
                self._on_logos_delta(delta)
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
                self._on_logos_end()

    async def _action_loop(self) -> None:
        """封装 action loop"""
        try:
            status: LoopStatus = 'running'
            self._loop_status["action"] = status
            await super()._action_loop()
        except FatalError as e:
            self._send_error(e, 'action-loop-failed')
            self.close()
        finally:
            status = 'stopped'
            self._loop_status["action"] = status

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
                "%r interpreter settled: %s",
                self,
                interpretation.id
            )
        except Exception as e:
            self.moss.logger.error("%s send interpreter frame failed: %s", self._log_prefix, e)

    @property
    def shell(self) -> MOSShell:
        return self._moss_runtime.shell

    async def _refresh_shell(self) -> None:
        await self.shell.refresh_metas(
            timeout=self._default_shell_prepare_timeout,
            stale_time=self._refresh_meta_stale_time,
        )

    def _on_thinking_start(self, thinking: Thinking) -> None:
        # 首帧提示: 把思维帧的 percepts 落到 output 总线, 供 headless 观测面消费.
        moment = thinking.moment
        if self._is_thinking_gated():
            # 注入 safemode 动态上下文, 提示模型其 logos 将经人工审批.
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
        if self._moss_runtime.session.is_running():
            self._moss_runtime.session.output(
                'moment',
                *moment.percepts_messages(),
                log=f"moment {moment.id}: {len(moment.percepts)} percepts",
            )

    def _on_thinking_exited(self, thinking: Thinking, err: BaseException | None) -> None:
        if self._ghost_instance:
            self._on_logos_end()
            self._ghost_instance.handle_thinking_exit(
                thinking,
                err,
            )


GhostRuntimeImpl = GhostInShellDrivenByMindflow
