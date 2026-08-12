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
    PIN_LABEL_MAX_LEN,
    ExecArguments,
    ExecPin,
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    Ground,
    GroundConvention,
    GroundError,
    GroundSet,
    LawArguments,
    LawPin,
    LsArguments,
    LsPin,
    PathOutsideRootError,
    Pin,
    RenderedView,
    TemplateInfo,
    ViewBlock,
    ViewHeader,
)
from ghoshell_moss.ground._grounds import DefaultGroundSet
from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME

__all__ = [
    # contract
    "GroundSet",
    "Ground",
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
    "LawPin",
    "LawArguments",
    "GroundConvention",
    "TemplateInfo",
    "ViewHeader",
    "ViewBlock",
    "RenderedView",
    "GroundError",
    "PathOutsideRootError",
    # constants
    "PIN_LABEL_MAX_LEN",
    # concrete
    "DefaultGroundSet",
    "DefaultGround",
    # L0
    "DEFAULT_L0_FILENAME",
]
