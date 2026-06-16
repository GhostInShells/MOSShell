"""急停级联控制器 — GhostRuntime 的 pause 可测试等价物.

持有 mindflow + shell, 负责级联协调. 不含回调, 回调由 GhostRuntime 层管理.
"""

from ghoshell_moss.core.blueprint.mindflow import Mindflow
from ghoshell_moss.core.concepts.shell import MOSShell


class PauseController:
    """急停单一真源 — 幂等, 级联 mindflow + shell."""

    def __init__(
            self,
            mindflow: Mindflow | None = None,
            shell: MOSShell | None = None,
    ):
        self._paused = False
        self._mindflow = mindflow
        self._shell = shell

    def bind(self, mindflow: Mindflow, shell: MOSShell) -> None:
        """GhostRuntime 生命周期中延迟绑定."""
        self._mindflow = mindflow
        self._shell = shell

    def is_paused(self) -> bool:
        return self._paused

    def pause(self, toggle: bool = True) -> bool:
        """设值. 返回 True 表示状态变更 (调用方应通知外部).

        幂等: ``pause(True)`` 多次调用返回 False, 不重复级联."""
        if self._paused == toggle:
            return False
        self._paused = toggle
        self._cascade(toggle)
        return True

    def _cascade(self, paused: bool) -> None:
        """级联到 mindflow 和 shell. 回调由 GhostRuntime 通过 shell.pause 的 callback 管理."""
        if paused:
            if self._mindflow is not None:
                self._mindflow.pause(True)
            if self._shell is not None:
                self._shell.pause(True)
        else:
            if self._mindflow is not None:
                self._mindflow.pause(False)
            if self._shell is not None:
                self._shell.pause(False)