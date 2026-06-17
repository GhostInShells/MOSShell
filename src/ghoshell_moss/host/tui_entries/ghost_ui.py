"""Ghost TUI — 主界面 logos 流式输出 + 文本输入，调试走 REPL inspector."""

import asyncio
from typing import Iterable

from ghoshell_moss.core.blueprint.host import MossHost, GhostRuntime, MossRuntime
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.host.tui import TUIState, MossHostTUI, RenderableType
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.repl.inspector_ghost import GhostInspector
from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector

__all__ = ["GhostREPLState", "GhostTUI"]


class GhostREPLState(REPLState):
    """Ghost 交互主界面 — 文本输入驱动信号，logos 流式渲染。"""

    def __init__(
            self,
            ghost_runtime: GhostRuntime,
            name: str = "echo",
    ) -> None:
        self._gr = ghost_runtime
        self._logos_task: asyncio.Task | None = None
        super().__init__(name)

    @property
    def _session(self):
        return self._gr.moss.session

    # ── REPLState overrides ──────────────────────

    def _create_repl_inspectors(self) -> dict[str, object]:
        return {
            "ghost": GhostInspector(
                ghost_runtime=self._gr,
                ghost=self._gr.ghost,
                mindflow=self._gr.mindflow,
                shell=self._gr.moss.shell,
            ),
            "matrix": MatrixInspector(self._gr.moss.matrix),
        }

    async def _on_text_input(self, console_input: str) -> None:
        self._session.add_input_signal(
            console_input,
            description="from ghost tui",
        )
        self.console.hint(f"signal sent: {console_input[:60]}...")

    def output_on_switch(self, enter_else_leave: bool) -> None:
        if enter_else_leave:
            self.console.info(
                f"Ghost [{self._gr.ghost.meta.name()}] — "
                f"type anything to send an input signal.\n"
                f"REPL: /ghost.health()  /ghost.pause()  /ghost.resume()  /ghost.faculties()"
            )
        else:
            self.console.info(f"Leave Ghost [{self._gr.ghost.meta.name()}]")

    # ── lifecycle ────────────────────────────────

    async def __aenter__(self):
        # 注册 session output 回调 — OutputItem 实时渲染到 TUI
        self._session.on_output(self._on_session_output)
        # 启动 logos 流消费
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

    # ── output / logos ───────────────────────────

    def _on_session_output(self, item: OutputItem) -> None:
        """session output 回调：将 OutputItem 渲染到 TUI。"""
        if not item.messages:
            return
        self.console.output(item)

    async def _consume_logos(self) -> None:
        """消费 logos 流，遇到换行立即输出。"""
        buffer = ""
        try:
            async for delta in self._session.get_logos():
                if not delta:
                    continue
                buffer += delta
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    self.console.rprint(line, spacing=False)
        except asyncio.CancelledError:
            pass
        finally:
            if buffer:
                self.console.rprint(buffer)


class GhostTUI(MossHostTUI[GhostRuntime]):
    """Ghost TUI — 组合 echo ghost state 和 Moss shell state。

    用法: GhostTUI().run()
    启动前通过 host.env.set_ghost_name("echo") 指定 ghost。
    """

    def __init__(self, host: MossHost | None = None):
        super().__init__(host=host or MossHost.discover())

    def _get_runtime(self) -> GhostRuntime:
        return self.host.run_ghost(self.host.env.ghost_name)

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
        return parts

    def _get_custom_intro(self) -> RenderableType | None:
        from rich.text import Text
        return Text(
            f"\nGhost: {self.host.env.ghost_name}\n"
            f"Type anything to talk to the ghost.",
            style="dim italic",
        )

    def create_states(self) -> Iterable[TUIState]:
        yield GhostREPLState(self.runtime, name=self.host.env.ghost_name)
        from ghoshell_moss.host.tui_entries.moss_runtime_ui import MOSSRuntimeREPLState
        yield MOSSRuntimeREPLState(self.host, self.runtime.moss, name="shell")


if __name__ == "__main__":
    from ghoshell_moss.host import Host

    env = Environment.discover()
    env.set_ghost_name("echo")
    tui = GhostTUI(host=Host(env=env))
    tui.run()
