"""Ghost TUI — logos stream + output items split into separate states, debug via REPL inspector."""

import asyncio
from typing import Iterable

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

    def _get_custom_intro(self) -> Renderable:
        from rich.text import Text
        return Text(
            f"\nGhost: {self.host.env.ghost_name}\n"
            f"Type anything to talk to the ghost. Ctrl+T to switch views.",
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
