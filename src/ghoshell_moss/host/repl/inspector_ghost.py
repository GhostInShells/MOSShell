"""Ghost 调试 Inspector — 在 REPL 中暴露 pause / health / state / faculties."""

from ghoshell_moss.core.blueprint.host import IGhostRuntime, LoopHealth
from ghoshell_moss.core.blueprint.ghost import Ghost
from ghoshell_moss.core.blueprint.mindflow import Mindflow
from ghoshell_moss.core.concepts.shell import MOSShell

__all__ = ["GhostInspector"]


class GhostInspector:
    """Ghost 运行时调试工具集。每个 public 方法自动成为 REPL 命令。"""

    def __init__(
            self,
            ghost_runtime: IGhostRuntime,
            ghost: Ghost,
            mindflow: Mindflow,
            shell: MOSShell,
    ):
        self._gr = ghost_runtime
        self._ghost = ghost
        self._mindflow = mindflow
        self._shell = shell

    def health(self) -> LoopHealth:
        """三循环状态: main / articulate / action 各自 running/stopped/not_started."""
        return self._gr.inspect_loop_health()

    def ghost_state(self) -> dict:
        """Ghost 内部快照（无固定 schema，每个原型自决）。"""
        return self._ghost.inspect_state()

    def faculties(self) -> dict:
        """Mindflow 中注册的全部 Nucleus — 类型、监听信号、运行状态、是否有 pending impulse."""
        result = {}
        for name, nucleus in self._mindflow.nuclei().items():
            result[name] = {
                "type": type(nucleus).__name__,
                "signals": list(nucleus.signals()),
                "status": nucleus.status() or "—",
                "running": nucleus.is_running(),
                "has_impulse": nucleus.peek() is not None,
            }
        return result

    def signal_routes(self) -> dict:
        """Signal → Nucleus 路由表. 从各 nucleus 的 signals() 反向聚合."""
        routes: dict[str, list[str]] = {}
        for nucleus_name, nucleus in self._mindflow.nuclei().items():
            for signal_name in nucleus.signals():
                if signal_name not in routes:
                    routes[signal_name] = []
                routes[signal_name].append(nucleus_name)
        return routes

    def pause(self) -> str:
        """暂停 mindflow 和 shell（拒答）— 走 GhostRuntime 统一入口."""
        self._gr.pause(True)
        return "paused — mindflow + shell"

    def resume(self) -> str:
        """恢复 mindflow 和 shell — 走 GhostRuntime 统一入口."""
        self._gr.pause(False)
        return "resumed — mindflow + shell"

    def mindflow_info(self) -> dict:
        """Mindflow 基本信息：类型、运行状态、pause 状态。"""
        mf = self._mindflow
        return {
            "type": type(mf).__name__,
            "is_running": mf.is_running(),
            "is_paused": self._gr.is_paused(),
        }
