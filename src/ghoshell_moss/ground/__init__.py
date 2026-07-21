"""Ground — Ghost 的文件系统认知场.

公开契约: ``ghoshell_moss.ground.contract``.
进程内默认实现: ``DefaultGroundSet`` + ``DefaultGround``.

内部模块:
- contract.py      契约 (ABC + 数据模型)
- _addr.py         路径锚点解析 ($GROUND / $CWD / $HOME)
- _hash.py         pin 观察 (per-class 多态)
- _l0.py           GROU.md 读写
- _chain.py        法链 (祖先 GROU.md body 收集)
- _render.py       frame 渲染 (SPEC §6 布局)
- _ground.py       DefaultGround
- _grounds.py      DefaultGroundSet
"""

from ghoshell_moss.ground.contract import (
    AT_BUDGET,
    AT_MAX_DEPTH,
    PIN_LABEL_MAX_LEN,
    FilePin,
    FrontmatterPin,
    GlobPin,
    Ground,
    GroundConvention,
    GroundError,
    GroundSet,
    LsPin,
    PathOutsideRootError,
    Pin,
    UpdateResult,
)
from ghoshell_moss.ground._grounds import DefaultGroundSet
from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, PIN_SECTION_HEADING

__all__ = [
    # contract
    "GroundSet",
    "Ground",
    "Pin",
    "FilePin",
    "GlobPin",
    "FrontmatterPin",
    "LsPin",
    "GroundConvention",
    "UpdateResult",
    "GroundError",
    "PathOutsideRootError",
    # constants
    "AT_BUDGET",
    "AT_MAX_DEPTH",
    "PIN_LABEL_MAX_LEN",
    # concrete
    "DefaultGroundSet",
    "DefaultGround",
    # L0
    "DEFAULT_L0_FILENAME",
    "PIN_SECTION_HEADING",
]
