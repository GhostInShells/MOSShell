"""Desktop — Ghost 的文件系统工作桌面.

公开契约 (ABC / Protocol / dataclass): ``ghoshell_moss.contracts.desktop``
进程内默认实现: ``DefaultDesktop`` (本模块).

12+1 原语三层 (导航 / 发现 / 读写 / 执行 / 后台 / Pin) + 两条元规则
(read-before-write 经 ReadHistory 协议, 统一输出截断). 详细契约见
``.design/2026-06-28_desktop_in_4d_cross_section.md``.
"""

from ghoshell_moss.contracts.desktop import (
    Desktop,
    ReadHistory,
    ReflectionHint,
    FileContent,
    ExecResult,
    Match,
    Task,
    PinInfo,
    DirectoryTree,
    DesktopError,
    ReadBeforeWriteError,
    PathOutsideRootError,
    PinBudgetExceeded,
)
from ghoshell_moss.core.desktop.desktop import (
    DefaultDesktop,
    DEFAULT_INSTRUCTION,
    DEFAULT_REFLECTION_PATHS,
)
from ghoshell_moss.core.desktop.models import (
    InProcessReadHistory,
    PinRecord,
)

__all__ = [
    # 契约 (re-export from contracts.desktop)
    "Desktop",
    "ReadHistory",
    "ReflectionHint",
    "FileContent",
    "ExecResult",
    "Match",
    "Task",
    "PinInfo",
    "DirectoryTree",
    "DesktopError",
    "ReadBeforeWriteError",
    "PathOutsideRootError",
    "PinBudgetExceeded",
    # 实现
    "DefaultDesktop",
    "DEFAULT_INSTRUCTION",
    "DEFAULT_REFLECTION_PATHS",
    "InProcessReadHistory",
    "PinRecord",
]
