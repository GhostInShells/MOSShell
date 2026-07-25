"""Ghost TUI — logos stream + output items split into separate states, debug via REPL inspector."""

import asyncio
from typing import Callable, Iterable

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import ConditionalKeyBindings, KeyBindings, merge_key_bindings

from ghoshell_moss.core.blueprint.host import MossHost, GhostRuntime
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.host.tui import TUIState, MossHostTUI, Renderable
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.repl.inspector_ghost import GhostInspector
from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector
from ghoshell_moss.host.repl.inspector_manifests import ManifestsInspector

__all__ = ["GhostLogosState", "GhostOutputState", "GhostTUI"]


class _GhostStateBase(REPLState):
    """Shared base: session access + ghost inspectors for both logos and output states."""

    def __init__(self, ghost_runtime: GhostRuntime, name: str):
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
                moss.project.matrix_manifests(),
                mode.manifests() if mode else None,
            ),
        }

    async def _on_text_input(self, console_input: str) -> None:
        self._session.add_input_signal(
            console_input,
            description="from ghost tui",
        )
        self.console.hint(f"signal sent: {console_input[:60]}...")


class GhostLogosState(_GhostStateBase):
    """Logos streaming display — text input drives signals, logos renders line by line."""

    def __init__(self, ghost_runtime: GhostRuntime, name: str = "echo"):
        self._logos_task: asyncio.Task | None = None
        super().__init__(ghost_runtime, name)

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


class GhostOutputState(_GhostStateBase):
    """Output item display — structured messages from ghost session."""

    def __init__(self, ghost_runtime: GhostRuntime, name: str = "messages"):
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


class GhostTUI(MossHostTUI[GhostRuntime]):
    """Ghost TUI — logos stream, output items, and shell debug.

    Usage: GhostTUI().run()
    Start with ``moss-run-ghost <name>`` or configure via Environment.
    """

    def __init__(self, host: MossHost | None = None):
        super().__init__(host=host or MossHost.discover())
        self._safe_mode_wired: bool = False

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
        # SafeMode 只在开启后才实例化, 早期检查 hasattr 避免触发懒加载副作用.
        # 直接调 safe_mode() 也 OK — 首次调用只是分配对象, 不改变行为.
        if self.runtime.safe_mode().is_enabled():
            parts.append(("fg:yellow bold", "[SAFE] "))
        return parts

    # ── SafeMode ──────────────────────────────────

    def _toggle_safe_mode(self) -> None:
        sm = self.runtime.safe_mode()
        target = not sm.is_enabled()
        if sm.set_enabled(target):
            # 首次开启时挂 pending 回调.
            if target and not self._safe_mode_wired:
                sm.on_pending_changed(self._on_safe_pending_changed)
                self._safe_mode_wired = True
            self._invalidate()

    def _on_safe_pending_changed(self) -> None:
        """SafeMode pending 进/出都回调此函数 — 刷 toolbar + invalidate."""
        p = self.runtime.safe_mode().pending()
        if p is None:
            self.set_bottom_toolbar("")
        else:
            preview = p['logos'][:80].replace("\n", " ")
            self.set_bottom_toolbar(
                f"[SAFE pending {p['uuid'][:8]}] c-y approve · c-d reject | {preview}"
            )
        self._invalidate()

    def _safe_approve(self) -> None:
        p = self.runtime.safe_mode().pending()
        if p is not None:
            self.runtime.safe_mode().approve(p['uuid'])

    def _safe_reject(self) -> None:
        p = self.runtime.safe_mode().pending()
        if p is not None:
            self.runtime.safe_mode().reject(p['uuid'], "rejected by user via c-d")

    def _invalidate(self) -> None:
        if self._prompt_session and self._prompt_session.app:
            self._prompt_session.app.invalidate()

    def default_commands(self) -> dict[str, tuple[str, Callable[[], None]]]:
        cmds = super().default_commands()
        cmds["safe"] = ("toggle SafeMode gate on articulator→action", self._toggle_safe_mode)
        return cmds

    def default_key_bindings(self) -> KeyBindings:
        base = super().default_key_bindings()
        # c-y / c-d 仅在有 pending 时激活, 无 pending 时保留默认行为 (yank / delete-char).
        gate_kb = KeyBindings()

        @gate_kb.add('c-y')
        def _approve(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._safe_approve)

        @gate_kb.add('c-d')
        def _reject(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._safe_reject)

        conditional = ConditionalKeyBindings(
            gate_kb,
            Condition(lambda: self.runtime.safe_mode().pending() is not None),
        )
        return merge_key_bindings([base, conditional])

    def _get_custom_intro(self) -> Renderable:
        from rich.text import Text
        return Text(
            f"\nGhost: {self.host.env.ghost_name}\n"
            f"Type anything to talk to the ghost. Ctrl+T to switch views.\n"
            f"/safe to toggle approval gate · c-y approve · c-d reject",
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
