"""Ghost TUI — 主界面 logos 流式输出 + 文本输入，调试走 REPL inspector."""

import asyncio
import re
from collections.abc import Iterable
from typing import Literal

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.host import GhostRuntime, MossHost
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.host.repl.inspector_ghost import GhostInspector
from ghoshell_moss.host.repl.inspector_manifests import ManifestsInspector
from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.tui import MossHostTUI, Renderable, TUIState

__all__ = ["GhostOutputMode", "GhostREPLState", "GhostTUI"]

GhostOutputMode = Literal["normal", "verbose", "trace"]
_OUTPUT_MODES = frozenset({"normal", "verbose", "trace"})
_NORMAL_OUTPUT_ROLES = frozenset({"command-output", "error"})
_VERBOSE_HIDDEN_ROLES = frozenset({"command-result"})
_PAIRED_COMMAND_RE = re.compile(
    r"<(?P<name>[A-Za-z_][\w.-]*:[\w.-]+)\b[^>]*>.*?</(?P=name)\s*>",
    re.DOTALL,
)
_SELF_CLOSING_TAG_RE = re.compile(r"<[A-Za-z_][\w.:-]*(?:\s+[^<>]*?)?/\s*>", re.DOTALL)
_TAG_RE = re.compile(r"</?[A-Za-z_][\w.:-]*(?:\s+[^<>]*?)?>", re.DOTALL)


def _normal_logos_text(logos: str) -> str:
    """Return user-facing text while keeping CTML control syntax private."""
    visible = logos
    previous = None
    while visible != previous:
        previous = visible
        visible = _PAIRED_COMMAND_RE.sub("", visible)
    visible = _SELF_CLOSING_TAG_RE.sub("", visible)
    visible = _TAG_RE.sub("", visible)
    lines = [line.rstrip() for line in visible.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    compact: list[str] = []
    for line in lines:
        if not line.strip() and compact and not compact[-1].strip():
            continue
        compact.append(line)
    return "\n".join(compact)


def _output_item_visible(role: str, mode: GhostOutputMode) -> bool:
    if mode == "normal":
        return role in _NORMAL_OUTPUT_ROLES
    if mode == "verbose":
        return role not in _VERBOSE_HIDDEN_ROLES
    return True


class GhostREPLState(REPLState):
    """Ghost 交互主界面 — 文本输入驱动信号，logos 流式渲染。"""

    def __init__(
            self,
            ghost_runtime: GhostRuntime,
            name: str = "echo",
            output_mode: GhostOutputMode = "normal",
    ) -> None:
        self._gr = ghost_runtime
        self._logos_task: asyncio.Task | None = None
        self._output_mode = output_mode
        super().__init__(name)

    @property
    def _session(self):
        return self._gr.moss.session

    # ── REPLState overrides ──────────────────────

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
                moss.project.matrix_manifests(),
                mode.manifests() if mode else None,
            ),
        }

    async def _on_text_input(self, console_input: str) -> None:
        self._session.add_input_signal(
            console_input,
            description="from ghost tui",
        )
        if self.operation_hints_enabled():
            self.console.hint(f"signal sent: {console_input[:60]}...")

    def operation_hints_enabled(self) -> bool:
        return self._output_mode != "normal"

    def set_output_mode(self, mode: GhostOutputMode) -> None:
        if mode not in _OUTPUT_MODES:
            raise ValueError(f"unsupported Ghost output mode: {mode}")
        self._output_mode = mode

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
        if not _output_item_visible(item.role, self._output_mode):
            return
        if self._output_mode == "normal":
            if item.role == "error":
                detail = item.messages_string() or item.log or "unknown runtime error"
                self.console.error(detail)
            else:
                content = item.messages_string()
                if content:
                    self.console.rprint(content, spacing=False)
            return
        if not item.messages:
            if item.log:
                self.console.hint(f"{item.role}: {item.log}")
            return
        self.console.output(item)

    async def _consume_logos(self) -> None:
        """Consume logos; normal mode emits only sanitized final user-facing text."""
        normal_buffer = ""
        line_buffer = ""
        try:
            async for delta in self._session.get_logos():
                if not delta:
                    continue
                if self._output_mode == "normal":
                    if line_buffer:
                        self.console.rprint(line_buffer, spacing=False)
                        line_buffer = ""
                    if delta == "\n\n":
                        visible = _normal_logos_text(normal_buffer)
                        if visible:
                            self.console.rprint(visible)
                        normal_buffer = ""
                    else:
                        normal_buffer += delta
                    continue

                if normal_buffer:
                    visible = _normal_logos_text(normal_buffer)
                    if visible:
                        self.console.rprint(visible)
                    normal_buffer = ""
                line_buffer += delta
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    self.console.rprint(line, spacing=False)
        except asyncio.CancelledError:
            pass
        finally:
            if normal_buffer:
                visible = _normal_logos_text(normal_buffer)
                if visible:
                    self.console.rprint(visible)
            if line_buffer:
                self.console.rprint(line_buffer)


class GhostTUI(MossHostTUI[GhostRuntime]):
    """Ghost TUI — 组合 echo ghost state 和 Moss shell state。

    用法: GhostTUI().run()
    启动前通过 Environment(ghost="echo").seal() 指定 ghost。
    """

    def __init__(self, host: MossHost | None = None, *, output_mode: GhostOutputMode = "normal"):
        if output_mode not in _OUTPUT_MODES:
            raise ValueError(f"unsupported Ghost output mode: {output_mode}")
        self._output_mode = output_mode
        self._ghost_repl_state: GhostREPLState | None = None
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
        if self._output_mode != "normal":
            parts.append(("fg:yellow", f"[{self._output_mode}] "))
        return parts

    def default_commands(self):
        commands = super().default_commands()
        commands.update({
            "normal": ("show only user-facing replies", lambda: self._set_output_mode("normal")),
            "verbose": ("show runtime summaries", lambda: self._set_output_mode("verbose")),
            "trace": ("show full internal command results", lambda: self._set_output_mode("trace")),
        })
        return commands

    def _set_output_mode(self, mode: GhostOutputMode) -> None:
        self._output_mode = mode
        if self._ghost_repl_state is not None:
            self._ghost_repl_state.set_output_mode(mode)
        self.console.notice(f"Ghost output mode: {mode}")

    def _get_custom_intro(self) -> Renderable:
        from rich.text import Text
        return Text(
            f"\nGhost: {self.host.env.ghost_name}\n"
            f"Type anything to talk to the ghost.",
            style="dim italic",
        )

    def create_states(self) -> Iterable[TUIState]:
        self._ghost_repl_state = GhostREPLState(
            self.runtime,
            name=self.host.env.ghost_name,
            output_mode=self._output_mode,
        )
        yield self._ghost_repl_state
        from ghoshell_moss.host.tui_entries.moss_runtime_ui import MOSSRuntimeREPLState
        yield MOSSRuntimeREPLState(self.host, self.runtime.moss, name="shell")


if __name__ == "__main__":
    from ghoshell_moss.core.blueprint.environment import Environment
    from ghoshell_moss.host import Host

    env = Environment(ghost="echo")
    env.seal()
    tui = GhostTUI(host=Host(env=env))
    tui.run()
