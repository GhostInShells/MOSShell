"""
FractalServeState — REPL 内分形 Hub 调试状态。

从 IoC 容器发现 FractalHub，提供调试交互命令。
Hub 的生命周期和 channel 集成由 Matrix 管理，此状态仅做观测和控制。
"""

from ghoshell_moss.core.blueprint.host import FractalHub

__all__ = ['FractalInspector']


class FractalInspector:
    """暴露给 REPL 命令的 fractal 操作接口（`/fractal.xxx()`）。"""

    def __init__(self, hub: FractalHub | None):
        self._hub = hub

    def explain(self) -> str:
        hub = self._hub
        if hub is None:
            return "No FractalHub configured in this environment."
        return hub.self_explain()

    def status(self) -> str:
        hub = self._hub
        if hub is None:
            return "No FractalHub configured in this environment.\nUse `moss manifests contracts` to check if FractalHub is registered."
        if not hub.is_running():
            return f"FractalHub is not running.\nUse `/fractal.start()` to start listening."
        return hub.status()

    def accept(self, cell_name: str) -> str:
        """accept connected cell"""
        hub = self._hub
        if hub is None:
            return "No FractalHub configured in this environment."
        try:
            hub.accept(cell_name)
            return "FractalHub accepted {}.".format(cell_name)
        except KeyError:
            return f"FractalHub has no cell named {cell_name}."

    def ignore(self, cell_name: str) -> str:
        """ignore connected cell"""
        hub = self._hub
        if hub is None:
            return "No FractalHub configured in this environment."
        try:
            hub.ignore(cell_name)
            return "FractalHub ignored {}.".format(cell_name)
        except KeyError:
            return f"FractalHub has no cell named {cell_name}."

    def connected(self) -> str:
        hub = self._hub
        if hub is None:
            return "No FractalHub configured."
        if not hub.is_running():
            return "FractalHub not running."
        nodes = hub.get_connected()
        if not nodes:
            return "No connected fractal nodes."
        lines = [f"Connected ({len(nodes)}):"]
        for c in nodes:
            lines.append(f"  - {c.name} alive={c.is_alive()}")
        return "\n".join(lines)
