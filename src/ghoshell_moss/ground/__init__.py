"""Ground — Ghost 的文件系统认知场.

公开契约: ``ghoshell_moss.ground.contract``.
进程内默认实现: ``DefaultGroundSet`` (根 + 边界, 按位置渲染).

内部模块:
- contract.py      契约 (数据模型 + GroundSet ABC)
- _addr.py         路径锚点解析 ($GROUND / $CWD / $HOME)
- _hash.py         pin 观察 (per-class 多态)
- _l0.py           GROUND.md 读写
- _chain.py        内容文件链 (content pin 向上收集)
- _edit.py         GROUND.md 的 pin 增删与校验 (纯文件操作)
- _render.py       渲染: 位置 → 自解释 XML 面
- _grounds.py      DefaultGroundSet
"""

from ghoshell_moss.ground.contract import (
    PIN_LABEL_MAX_LEN,
    ContentArguments,
    ContentPin,
    ExecArguments,
    ExecPin,
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    GroundConvention,
    GroundError,
    GroundSet,
    LsArguments,
    LsPin,
    PathOutsideRootError,
    Pin,
)
from ghoshell_moss.ground._grounds import DefaultGroundSet
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME
from ghoshell_moss.ground._render import RenderLimits, render_position

__all__ = [
    # contract
    "GroundSet",
    "Pin",
    "FilePin",
    "FileArguments",
    "GlobPin",
    "GlobArguments",
    "FrontmatterPin",
    "FrontmatterArguments",
    "LsPin",
    "LsArguments",
    "ExecPin",
    "ExecArguments",
    "ContentPin",
    "ContentArguments",
    "GroundConvention",
    "GroundError",
    "PathOutsideRootError",
    # constants
    "PIN_LABEL_MAX_LEN",
    "DEFAULT_L0_FILENAME",
    # concrete
    "DefaultGroundSet",
    "RenderLimits",
    "render_position",
]
