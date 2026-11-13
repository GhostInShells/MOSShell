"""Ground contract — Ghost 的认知场.

Ground 一句话承诺: "在 context 表面钉住一组注视目标, 每帧重绘."

结构:
1. GroundSet — 容器. open/close 多个 Ground, CTML 接触面
2. Ground   — 一个打开的场. 绑定目录 root, 持有 pin 集, 承担 frame 渲染
3. Pin      — 一枚注视声明. 具体子类携带 verb + typed arguments

不承担: 子进程执行 / 周期性 fold / 持久记忆 — 各自由独立 contract 负责.

SPEC: ``ghoshell_moss.ground.SPECIFICATION.md`` (draft 0.2.0).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "GroundSet",
    "Pin",
    "FilePin",
    "GlobPin",
    "FrontmatterPin",
    "LsPin",
    "ExecPin",
    "ContentPin",
    "FileArguments",
    "GlobArguments",
    "FrontmatterArguments",
    "LsArguments",
    "ExecArguments",
    "ContentArguments",
    "GroundConvention",
    "GroundError",
    "PathOutsideRootError",
]

# -- constants ----------------------------------------------------------------

# 1-byte length prefix + 64-char namespace convention.
PIN_LABEL_MAX_LEN = 63
_PIN_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,%d}$" % PIN_LABEL_MAX_LEN)


# -- frontmatter -------------------------------------------------------------


class GroundConvention(BaseModel):
    """L0 frontmatter — 场的身份声明 + pins 清单 + 场级 ignore 规则.

    frontmatter 是 MOSS 唯一的机器发明域. pins 作为机器声明的注视
    列表驻留在 frontmatter 中, body 保持纯粹的人/模型叙事域.
    未知 key 保留不拒 (extra="allow").

    ignore / ignore_file 是场级规则 — 所有发现型 pin (glob, frontmatter
    pattern, ls) 自动受约束, 无需 pin 级 opt-in. inline list 与文件引用
    合并为最终规则集.
    """

    id: str | None = Field(default=None, alias="$id")
    name: str | None = None
    description: str | None = None
    pins: list[dict] = Field(default_factory=list)
    inline: list[str] | None = Field(
        default=None,
        description="要展开的子场 — 相对场根的目录路径清单, 每个目录自带 GROUND.md. "
        "由 GroundSet 物化 (一层, 不递归), Ground 自身不消费.",
    )
    ignore: list[str] | None = Field(
        default=None,
        description="场级 ignore 规则清单 — .gitignore 语义, 相对场根.",
    )
    ignore_file: str | None = Field(
        default=None,
        description="场根下的 ignore 规则文件路径 (.gitignore / .groundignore).",
    )

    model_config = {"extra": "allow"}


# -- pin: verb arguments models -----------------------------------------------


class FileArguments(BaseModel):
    """file verb 的 arguments."""
    path: str = Field(description="文件路径, 锚点语法允许.")
    range: str | None = Field(
        default=None,
        pattern=r"^\d+(-\d+)?$",
        description="行区间: 'N' (单行) 或 'N-M' (1-indexed).",
    )
    budget: int | None = Field(
        default=None,
        ge=1,
        description="内容字符数上限, 超限 truncate.",
    )

    model_config = {"extra": "allow"}


class GlobArguments(BaseModel):
    """glob verb 的 arguments."""
    path: str = Field(description="glob 路径 (*, **, ? 标准语义).")
    limit: int | None = Field(default=None, ge=1, description="命中路径数上限.")
    max_depth: int | None = Field(default=None, ge=1, description="** 递归深度上限.")

    model_config = {"extra": "allow"}


class FrontmatterArguments(BaseModel):
    """frontmatter verb 的 arguments.

    Single-file mode (path 不含 glob 字符): 提取该文件 frontmatter.
    Pattern mode (path 含 *?[): 匹配多文件, 每个文件的 frontmatter
    作为独立结果块. 这是渐进式披露的核心 — 一个 pin 看全部子场身份.
    """
    path: str = Field(description="文件路径, 或 glob pattern 匹配多文件.")
    keys: list[str] | None = Field(
        default=None,
        description="只提取指定 frontmatter key. None = 全块.",
    )
    budget: int | None = Field(default=None, ge=1, description="内容字符数上限.")
    limit: int | None = Field(default=None, ge=1, description="多文件模式命中数上限.")
    max_depth: int | None = Field(default=None, ge=1, description="递归发现深度上限.")

    model_config = {"extra": "allow"}


class LsArguments(BaseModel):
    """ls verb 的 arguments."""
    path: str = Field(description="目录路径.")
    depth: int = Field(default=2, ge=1, le=8, description="遍历深度. 默认 2.")
    limit: int | None = Field(default=None, ge=1, description="输出条目数上限.")
    max_depth: int | None = Field(
        default=None,
        ge=1,
        description="递归深度上限. 与 depth 取较小者.",
    )

    model_config = {"extra": "allow"}


class ExecArguments(BaseModel):
    """exec verb 的 arguments.

    授权模型 = Makefile 级信任: ref 指向场根子树内可执行文件, 场作者背书.
    协议禁止内联 shell 字符串 (授权泄漏), 禁止跨场引用 (../, 绝对路径).

    mode 决定解释器: shebang(默认, 需 +x) / python(sys.executable) /
    shell(sh). 非 shebang 模式用解释器显式执行, 不要求脚本 +x.
    """
    ref: str = Field(
        description="场根子树内的可执行文件相对路径. 不允许 ../, 不允许绝对路径.",
    )
    mode: str = Field(
        default="shebang",
        description="解释器模式: shebang(默认, 需 +x) | python(sys.executable) | shell(sh). 非 shebang 不要求 +x.",
    )
    timeout: float = Field(
        default=10.0, gt=0, le=60,
        description="秒. 超时渲染 [timeout] 标记, 不静默.",
    )
    budget: int | None = Field(default=None, ge=1, description="stdout 字符数上限.")

    model_config = {"extra": "allow"}


class ContentArguments(BaseModel):
    """content verb 的 arguments — 约定文件内容链.

    参数是文件名而非路径: 从 cwd 向上逐层收集该文件 (CLAUDE.md /
    AGENT.md 等约定文件), 到场根为止. 收集到的是每个祖先目录里的
    body 内容, 父级向下展示.
    """
    filename: str = Field(
        description="约定文件名 (CLAUDE.md, AGENT.md...). 从 cwd 向上逐层收集.",
    )
    budget: int | None = Field(
        default=None,
        ge=1,
        description="总字符数上限, 超限 truncate.",
    )
    lines: int | None = Field(
        default=None,
        ge=1,
        description="总行数上限, 超限 truncate.",
    )

    model_config = {"extra": "allow"}


# -- pin verb registry -------------------------------------------------------

_VERB_CLASSES: dict[str, type[Pin]] = {}
"""verb → Pin subclass dispatch.  Populated below after class definitions."""


def _register(verb: str):
    def _deco(cls: type[Pin]) -> type[Pin]:
        _VERB_CLASSES[verb] = cls
        return cls
    return _deco


# -- pins --------------------------------------------------------------------


class Pin(BaseModel, ABC):
    """pin 基类 — 场里的一枚注视声明.

    固定 envelope: {label, verb, arguments, description}.  具体子类携带
    typed arguments — verb 是 Literal discriminator, arguments 是多态载体.
    """

    verb: str = Field(
        default='',
        description="verb of the pin"
    )
    label: str = Field(
        min_length=1,
        max_length=PIN_LABEL_MAX_LEN,
        pattern=_PIN_LABEL_RE.pattern,
        description="ground 内唯一标识, 承担 unpin 定位.",
    )
    description: str = Field(
        default="",
        max_length=280,
        description="短评注 — 一行 '为什么盯这个'. 长解说走 body.",
    )
    always_show: bool = Field(
        default=False,
        description="walk / --template 模式下也不折叠 — 永远展开内容. 默认折叠.",
    )

    model_config = {"extra": "ignore"}

    @property
    def is_cwd_anchored(self) -> bool:
        """是否随 $CWD 移动 — walk 时展开, 场根时也渲染. 默认 False."""
        return False


@_register("file")
class FilePin(Pin):
    """单文件注视, 可选行区间."""

    verb: Literal["file"] = "file"
    arguments: FileArguments


@_register("glob")
class GlobPin(Pin):
    """glob 注视 — 命中路径清单, 不出内容."""

    verb: Literal["glob"] = "glob"
    arguments: GlobArguments


@_register("frontmatter")
class FrontmatterPin(Pin):
    """单文件 frontmatter 注视 — 只出 YAML frontmatter, 不出 body."""

    verb: Literal["frontmatter"] = "frontmatter"
    arguments: FrontmatterArguments


@_register("ls")
class LsPin(Pin):
    """目录列表注视 — 结构视图, 不出内容."""

    verb: Literal["ls"] = "ls"
    arguments: LsArguments


@_register("exec")
class ExecPin(Pin):
    """可执行文件注视 — observe 即执行, 对账 = hash(stdout).

    授权模型: ref 指向场作者背书的场内可执行文件, 类比 Makefile target.
    不允许内联 shell 字符串, 不允许跨场引用.
    """

    verb: Literal["exec"] = "exec"
    arguments: ExecArguments


@_register("content")
class ContentPin(Pin):
    """约定文件内容链注视 — 兼容外部项目约定 (CLAUDE.md / AGENT.md).

    拉的是文档, 参数是文件名而非路径: 从 cwd 向上逐层收集该文件,
    到场根为止 (边界 = ground root). 父级向下展示, 最多一层 @ 解析,
    有 budget/lines 截断语义. 位置依赖 cwd — walk 时随站立位置变化.
    """

    verb: Literal["content"] = "content"
    arguments: ContentArguments

    @property
    def is_cwd_anchored(self) -> bool:
        return True


# -- errors ------------------------------------------------------------------


class GroundError(Exception):
    """Ground 契约层所有异常的基类."""


class PathOutsideRootError(GroundError):
    """路径逃逸出锚点子树. SPEC §8 per-anchor confinement."""


# -- GroundSet --------------------------------------------------------------


class GroundSet(ABC):
    """一棵场树: 一个根目录 + 边界, 渲染按位置解析 (SPEC §6.1).

    不再持有打开的 Ground 对象: 渲染是冷读, 每次从盘上解析. 保留这个类是因为
    边界需要一个身份 —— channel / runtime / CLI 各持一个, 不相交.
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """锚点场根 — 也是向上探索的边界, 永不越出."""

    @property
    @abstractmethod
    def boundary(self) -> Path:
        """向上探索的边界. 默认等于 root."""

    @abstractmethod
    async def render(
        self,
        position: str | Path | None = None,
        *,
        template: str | None = None,
        limits: object | None = None,
    ) -> str:
        """渲染一个位置 → 自解释 XML. None = 场根."""
