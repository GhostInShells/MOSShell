"""Ghost TUI — logos stream + output items split into separate states, debug via REPL inspector."""

import asyncio
import queue
from typing import Callable, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit.key_binding import KeyPressEvent
from ghoshell_moss.core.blueprint.host import IHost, IGhostRuntime
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.host.tui import TUIState, MossHostTUI, Renderable, RichCaller
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.repl.inspector_ghost import GhostInspector
from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector
from ghoshell_moss.host.repl.inspector_manifests import ManifestsInspector

__all__ = ["GhostLogosState", "GhostOutputState", "GhostTUI"]


class LogosStreamSink(RichCaller):
    """跨线程流式文本输出槽 — 原地重绘 RESPONSE panel.

    event loop 线程: send(delta) 把片段投进 queue.Queue.
    渲染线程: __call__(console) 阻塞消费, 攒到 buffer 后原地重绘一个
    " RESPONSE " panel (光标上移 + 清屏 + 重画), 片断逐次增长.

    毒丸 (None) 由 close() 幂等投递: 正常结束 (LOGOS_END) 与中断清理
    (finally) 都走 close(), 保证渲染线程的阻塞 get() 一定能退出.
    毒丸是普通队列元素而非 shutdown — 消费循环 break 后仍走最后一次
    _render_panel(), 最后一批 pending 不会丢 (旧 janus 版 shutdown 会让
    get() 抛异常, except 吞掉后跳过最后一次重绘, 才是最后一行不打印的根因).

    _rendered 区分首次消费与 replay: state 切换 replay 时直接重放完整 panel.
    """

    _POISON = None  # 毒丸哨兵: 结束流

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._stopped = False
        self._rendered = False
        self._buffer: list[str] = []
        self._rendered_lines = 0

    def send(self, delta: str) -> None:
        """投递一个 logos 片段. 已关闭则丢弃 (中断兜底)."""
        if self._closed:
            return
        self._queue.put(delta)

    def close(self) -> None:
        """投毒丸结束流, 幂等."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._POISON)

    def stop(self) -> None:
        """中断触达: 投毒丸并标记停止, 渲染线程醒来后立即退出, 不再画最后一次.

        与 close 的区别: close 是流正常结束 (LOGOS_END), 醒来后渲染完整 panel;
        stop 是外部中断 (interrupt / ctrl+c), 醒来后跳过最后一次重绘直接退出,
        避免与关闭流程的 console 写并发.
        """
        if self._stopped:
            return
        self._stopped = True
        self.close()

    @staticmethod
    def _panel(text: str) -> Panel:
        return Panel(
            Text(text),
            title=" RESPONSE ",
            title_align="left",
            border_style="cyan",
        )

    def _render_panel(self, console: Console) -> None:
        if self._rendered_lines > 0:
            # 光标上移到上一个 panel 的起始行 + 清屏
            console.file.write(f"\033[{self._rendered_lines}F")
            console.file.write("\033[J")
        with console.capture() as capture:
            console.print(self._panel("".join(self._buffer)))
        output = capture.get()
        self._rendered_lines = output.count("\n")
        console.file.write(output)
        console.file.flush()

    def __call__(self, console: Console) -> None:
        """渲染线程入口: 首次阻塞消费并原地重绘, replay 重放完整 panel."""
        if self._rendered:
            if self._buffer:
                console.print(self._panel("".join(self._buffer)))
            return

        pending: list[str] = []
        while True:
            item = self._queue.get()
            if item is self._POISON:
                break
            pending.append(item)
            if self._queue.empty():
                self._buffer.append("".join(pending))
                pending.clear()
                self._render_panel(console)

        if self._stopped:
            self._rendered = True
            return
        if pending:
            self._buffer.append("".join(pending))
            pending.clear()
        self._render_panel(console)
        self._rendered = True
        console.print("")


class _GhostStateBase(REPLState):
    """Shared base: session access + ghost inspectors for both logos and output states."""

    def __init__(self, ghost_runtime: IGhostRuntime, name: str):
        self._gr = ghost_runtime
        super().__init__(name)

    @property
    def _session(self):
        return self._gr.moss.session

    def _create_repl_inspectors(self) -> dict[str, object]:
        moss = self._gr.moss
        mode = moss.mode if moss.is_running() else None
        return {
            "ghost": GhostInspector(
                ghost_runtime=self._gr,
                ghost=self._gr.ghost,
                mindflow=self._gr.mindflow,
                shell=moss.shell,
            ),
            "matrix": MatrixInspector(moss.matrix),
            "manifests": ManifestsInspector(
                moss.project.project_manifests(),
                mode.manifests() if mode else None,
            ),
        }

    async def _on_text_input(self, console_input: str) -> None:
        self._session.add_input_signal(
            console_input,
            description="from ghost tui",
        )
        self.console.hint(f"signal sent: {console_input[:60]}...")

    def on_interrupt(self, event: KeyPressEvent) -> None:
        # 先停本地 REPL operation (文本输入处理), 再向 ghost session 发 interrupt
        # signal → InterruptNucleus → ghost_runtime shell.clear() 停生成.
        super().on_interrupt(event)
        self._session.add_signal(
            new_interrupt_signal(description="from ghost tui"),
        )
        self.console.hint("interrupt sent — generation stopped, shell cleared")


class GhostLogosState(_GhostStateBase):
    """Logos streaming display — text input drives signals, logos renders line by line."""

    def __init__(self, ghost_runtime: IGhostRuntime, name: str = "echo"):
        self._logos_task: asyncio.Task | None = None
        self._sink: LogosStreamSink | None = None
        super().__init__(ghost_runtime, name)

    def on_interrupt(self, event: KeyPressEvent) -> None:
        # 立即触达渲染侧: 停掉正在原地重绘的 sink, 不依赖 interrupt signal 走
        # mindflow 长链 (abort → LOGOS_END → close) 才让渲染停止. 生成侧仍由
        # super().on_interrupt 的 interrupt signal 收线.
        if self._sink is not None:
            self._sink.stop()
            self._sink = None
        super().on_interrupt(event)

    def output_on_switch(self, enter_else_leave: bool) -> None:
        if enter_else_leave:
            self.console.info(
                f"Ghost [{self._gr.ghost.meta.name()}] — "
                f"logos stream. Type to send input signals.\n"
                f"REPL: /ghost.health()  /ghost.pause()  /ghost.resume()  /ghost.faculties()"
            )
        else:
            self.console.info(f"Leave logos [{self._gr.ghost.meta.name()}]")

    async def __aenter__(self):
        self._logos_task = asyncio.get_running_loop().create_task(self._consume_logos())
        await super().__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._logos_task and not self._logos_task.done():
            self._logos_task.cancel()
            try:
                await self._logos_task
            except asyncio.CancelledError:
                pass
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _consume_logos(self) -> None:
        """消费 logos 流: 片断投递到 LogosStreamSink, 由渲染线程逐片打印.

        一个 utterance 对应一个 sink: 首个内容片断创建 sink 并 rprint 进渲染队列,
        后续片断 send 投递; LOGOS_END 关闭 sink 投毒丸. finally 兜底中断清理 —
        保证渲染线程的阻塞 get() 一定会被毒丸唤醒, 不会永久卡住.
        """
        try:
            async for delta in self._session.get_logos():
                if not delta:
                    continue
                if delta == self._session.LOGOS_END:
                    if self._sink is not None:
                        self._sink.close()
                        self._sink = None
                    continue
                if self._sink is None:
                    self._sink = LogosStreamSink()
                    self.console.rprint(self._sink)
                self._sink.send(delta)
        except asyncio.CancelledError:
            pass
        finally:
            if self._sink is not None:
                self._sink.close()
                self._sink = None


class GhostOutputState(_GhostStateBase):
    """Output item display — structured messages from ghost session."""

    def __init__(self, ghost_runtime: IGhostRuntime, name: str = "messages"):
        super().__init__(ghost_runtime, name)

    def output_on_switch(self, enter_else_leave: bool) -> None:
        if enter_else_leave:
            self.console.info(
                f"Ghost [{self._gr.ghost.meta.name()}] — "
                f"output items. Structured messages from session."
            )
        else:
            self.console.info(f"Leave output items [{self._gr.ghost.meta.name()}]")

    async def __aenter__(self):
        self._session.on_output(self._on_session_output)
        await super().__aenter__()

    def _on_session_output(self, item: OutputItem) -> None:
        if not item.messages:
            return
        self.console.output(item)


class GhostTUI(MossHostTUI[IGhostRuntime]):
    """Ghost TUI — logos stream, output items, and shell debug.

    Usage: GhostTUI().run()
    Start with ``moss-ghost run <name>`` or configure via Environment.
    """

    def __init__(self, host: IHost | None = None):
        super().__init__(host=host or IHost.discover())
        self._safe_mode_wired: bool = False

    def _get_runtime(self) -> IGhostRuntime:
        return self.host.run_ghost(self.host.env.ghost_name)

    def _get_session(self):
        return self.runtime.moss.session

    def _log_loop_exception(self, message: str, exception: BaseException | None) -> None:
        self.runtime.moss.matrix.logger.exception("%s: %s", message, exception)

    def _on_emergency_pause(self) -> None:
        target = not self.runtime.is_paused()
        self.runtime.pause(target, callback=self._on_pause_done)

    def _on_pause_done(self) -> None:
        if self._prompt_session and self._prompt_session.app:
            self._prompt_session.app.invalidate()

    def _prompt_status(self) -> list[tuple[str, str]]:
        parts = super()._prompt_status()
        if self.runtime.is_paused():
            parts.append(("fg:red bold", "[PAUSED] "))
        if self.runtime.safe_mode().is_enabled():
            parts.append(("fg:yellow bold", "[SAFE] "))
        return parts

    # ── SafeMode ──────────────────────────────────
    # 回合制交互: [SAFE] prefix 已把 shell 语义翻转, prompt 输入行随之重定向到审批.
    # 输入协议:
    #   ""            → approve (无 note)
    #   "!<text>"     → approve-with-note (text 作为附言给 ghost 下一帧内观)
    #   "<text>"      → reject with reason (text 作为否决理由)
    #   "/<cmd>"      → 走默认命令 (可 /safe 关掉 gate, /exit 等), pending 不消化
    # 默认 = reject 是刻意: 误发文本时不会误批准 (approve 是更坏后果).

    def _toggle_safe_mode(self) -> None:
        sm = self.runtime.safe_mode()
        target = not sm.is_enabled()
        if sm.set_enabled(target):
            # 首次开启时挂 invalidate 回调 — pending 变更时 placeholder 刷新.
            if target and not self._safe_mode_wired:
                sm.on_pending_changed(self._invalidate)
                self._safe_mode_wired = True
            self._invalidate()

    def _pre_handle_input(self, item: str) -> bool:
        p = self.runtime.safe_mode().pending()
        if p is None:
            return False
        # /-prefix 让路给默认命令 (关 gate / exit 等), 不消化 pending.
        if item.startswith('/'):
            return False
        sm = self.runtime.safe_mode()
        if item == '':
            sm.approve(p['uuid'])
        elif item.startswith('!'):
            sm.approve(p['uuid'], note=item[1:].lstrip())
        else:
            sm.reject(p['uuid'], item)
        return True

    def _get_input_placeholder(self):
        def _build():
            p = self.runtime.safe_mode().pending()
            if p is None:
                return ""
            return f"[SAFE {p['uuid'][:8]}] enter=approve · !<text>=approve-with-note · <text>=reject"

        return _build

    def _invalidate(self) -> None:
        if self._prompt_session and self._prompt_session.app:
            self._prompt_session.app.invalidate()

    def default_commands(self) -> dict[str, tuple[str, Callable[[], None]]]:
        cmds = super().default_commands()
        cmds["safe"] = ("toggle SafeMode gate on articulator→action", self._toggle_safe_mode)
        return cmds

    def _get_custom_intro(self) -> Renderable:
        from rich.text import Text
        return Text(
            f"\nGhost: {self.host.env.ghost_name}\n"
            f"Type anything to talk to the ghost. Ctrl+T to switch views.\n"
            f"/safe toggles approval gate — during pending: enter=approve, !<text>=approve-with-note, <text>=reject",
            style="dim italic",
        )

    def create_states(self) -> Iterable[TUIState]:
        yield GhostLogosState(self.runtime, name=self.host.env.ghost_name)
        yield GhostOutputState(self.runtime)
        from ghoshell_moss.host.tui_entries.moss_runtime_ui import MOSSRuntimeREPLState
        yield MOSSRuntimeREPLState(self.host, self.runtime.moss, name="shell")


if __name__ == "__main__":
    from ghoshell_moss.core.blueprint.environment import Environment
    from ghoshell_moss.host import Host

    env = Environment(ghost="echo")
    env.seal()
    tui = GhostTUI(host=Host(env=env))
    tui.run()
