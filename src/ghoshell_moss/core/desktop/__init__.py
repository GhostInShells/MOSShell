"""Desktop — Ghost 的文件系统认知桌面 (concrete 层).

公开契约: ``ghoshell_moss.contracts.desktop``.
进程内默认实现: ``DefaultGrounds`` (owner-scoped) + ``DefaultGround``.

内部模块 (下划线前缀, 不承诺稳定):
- _addr.py         地址解析 (path / range / glob)
- _hash.py         对账观察 (mtime + hash)
- _l0.py           L0 (DESKTOP.md) frontmatter + pin 段 IO
- _instruction.py  法链向上收集 (CLAUDE.md 等)
- _render.py       context 帧渲染
- _ground.py       DefaultGround
- _grounds.py      DefaultGrounds
"""

from ghoshell_moss.contracts.desktop import (
    ContextBudgetExceeded,
    DesktopError,
    Ground,
    GroundConvention,
    Grounds,
    PathOutsideRootError,
    Pin,
    UpdateResult,
)
from ghoshell_moss.core.desktop._grounds import DefaultGrounds
from ghoshell_moss.core.desktop._ground import DefaultGround
from ghoshell_moss.core.desktop._l0 import (
    DEFAULT_L0_FILENAME,
    PIN_SECTION_HEADING,
)

__all__ = [
    # 契约 re-export (便利导入)
    "Ground",
    "Grounds",
    "Pin",
    "GroundConvention",
    "UpdateResult",
    "DesktopError",
    "PathOutsideRootError",
    "ContextBudgetExceeded",
    # 实现
    "DefaultGrounds",
    "DefaultGround",
    # L0 常量
    "DEFAULT_L0_FILENAME",
    "PIN_SECTION_HEADING",
]
