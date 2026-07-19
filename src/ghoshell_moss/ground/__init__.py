"""Ground — Ghost 的文件系统认知场 (concrete 层).

公开契约: ``ghoshell_moss.ground.contract``.
进程内默认实现: ``DefaultGrounds`` (owner-scoped) + ``DefaultGround``.

内部模块 (下划线前缀, 不承诺稳定):
- _addr.py         地址解析 (path / range / glob)
- _hash.py         对账观察 (mtime + hash)
- _l0.py           L0 (GROUND.md) frontmatter + pin 段 IO
- _instruction.py  法链向上收集 (CLAUDE.md 等)
- _render.py       context 帧渲染
- _ground.py       DefaultGround
- _grounds.py      DefaultGrounds
"""

from ghoshell_moss.ground.contract import (
    ContextBudgetExceeded,
    GroundBaseError,
    Ground,
    GroundConvention,
    Grounds,
    PathOutsideRootError,
    Pin,
    UpdateResult,
)
from ghoshell_moss.ground._grounds import DefaultGrounds
from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import (
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
    "GroundBaseError",
    "PathOutsideRootError",
    "ContextBudgetExceeded",
    # 实现
    "DefaultGrounds",
    "DefaultGround",
    # L0 常量
    "DEFAULT_L0_FILENAME",
    "PIN_SECTION_HEADING",
]
