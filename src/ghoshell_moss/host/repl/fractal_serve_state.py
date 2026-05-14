"""
FractalServeState — REPL 内分形 Hub 调试状态。

从 IoC 容器发现 FractalHub，提供调试交互命令。
Hub 的生命周期和 channel 集成由 Matrix 管理，此状态仅做观测和控制。
"""

from ghoshell_moss.core.blueprint.host import MossHost, MossRuntime, FractalHub
from ghoshell_moss.host.repl.repl_state import REPLState

__all__ = ["FractalServeState"]


class _FractalOps:
    """暴露给 REPL 命令的 fractal 操作接口（`/fractal.xxx()`）。"""

    def __init__(self, state: "FractalServeState"):
        self._state = state

    def explain(self) -> str:
        hub = self._state.hub
        if hub is None:
            return "No FractalHub configured in this environment."
        return hub.usage()

    def status(self) -> str:
        hub = self._state.hub
        if hub is None:
            return "No FractalHub configured in this environment.\nUse `moss manifests contracts` to check if FractalHub is registered."
        if not hub.is_running():
            return f"FractalHub '{hub.name}' is not running.\nUse `/fractal.start()` to start listening."
        return hub.status()

    async def start(self) -> str:
        hub = self._state.hub
        if hub is None:
            return "No FractalHub configured. Check workspace configs/zenoh_config_fractal_hub.json5."
        if hub.is_running():
            return f"FractalHub '{hub.name}' already running."
        await hub.__aenter__()
        return f"FractalHub '{hub.name}' started."

    async def stop(self) -> str:
        hub = self._state.hub
        if hub is None:
            return "No FractalHub configured."
        if not hub.is_running():
            return "FractalHub not running."
        await hub.__aexit__(None, None, None)
        return "FractalHub stopped."

    def connected(self) -> str:
        hub = self._state.hub
        if hub is None:
            return "No FractalHub configured."
        if not hub.is_running():
            return "FractalHub not running."
        nodes = hub.get_connected()
        if not nodes:
            return "No connected fractal nodes."
        lines = [f"Connected ({len(nodes)}):"]
        for c in nodes:
            lines.append(f"  - {c.name}")
        return "\n".join(lines)


class FractalServeState(REPLState):
    """
    Fractal Hub 调试状态。

    从 IoC 容器发现 FractalHub（由 Matrix 自动集成），
    提供 `/fractal.start|stop|status|connected|explain` 命令。
    """

    def __init__(self, host: MossHost, runtime: MossRuntime):
        self._host = host
        self._runtime = runtime
        self._hub = runtime.get_fractal_hub()
        super().__init__("Fractal Serve")

    # ---- REPLState interface ----

    def _create_repl_inspectors(self) -> dict[str, object]:
        return {"fractal": _FractalOps(self)}

    def output_on_switch(self, enter_or_leave: bool) -> None:
        if enter_or_leave:
            if self._hub is None:
                self.console.info(
                    "Fractal Serve — 分形 Hub 调试\n"
                    "\n"
                    "No FractalHub configured in this environment.\n"
                    "To enable: add zenoh_config_fractal_hub.json5 to workspace configs/\n"
                    "and register ZenohFractalHubProvider in manifests/providers.py."
                )
            else:
                self.console.info(
                    "Fractal Serve — 分形 Hub 调试\n"
                    f"Hub: {self._hub.name}\n"
                    f"Running: {self._hub.is_running()}\n"
                    "\n"
                    "Use `/fractal.start()` to begin listening.\n"
                    "Use `/fractal.stop()` to stop listening.\n"
                    "Use `/fractal.status()` to check running state.\n"
                    "Use `/fractal.connected()` to list discovered nodes.\n"
                    "Use `/fractal.explain()` for transport info."
                )
        else:
            self.console.info("Leaving Fractal Serve.")

    async def _on_text_input(self, console_input: str) -> None:
        pass  # all interaction via REPL commands

    # ---- async lifecycle ----

    async def __aenter__(self):
        await super().__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)

    # ----

    @property
    def hub(self) -> FractalHub | None:
        return self._hub
