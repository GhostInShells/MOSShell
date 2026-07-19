"""Pin address parsing — 三种语法的解析与 root-bound 解析.

Pin.addr 契约里是 str, 三种语法:

- ``path``            file: 整份文件 (相对 Ground.root)
- ``path:80-140``     range: 行区间 (1-based, 含端点)
- ``**/*.py``         glob: 匹配集合

impl 层一次性解析成 ParsedAddr, 之后 observe / render 直接消费.

命名: 只暴露 ``parse_addr`` + ``resolve_*``, 不做 class 语法糖. K18 收缩
纪律 — 单纯的字符串 → dataclass 变换不值得包一层 Parser 类.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ghoshell_moss.ground.contract import PathOutsideRootError

__all__ = [
    "AddrKind",
    "ParsedAddr",
    "parse_addr",
    "resolve_file_addr",
    "resolve_glob_addr",
]

AddrKind = Literal["file", "range", "glob"]

# 严格匹配 `path:start-end` 后缀. start/end 都是纯数字, 都必须存在.
# path 部分允许含 ':' (unix 文件名合法), 匹配的是最后一段 `:N-N` 语法后缀.
_RANGE_SUFFIX_RE = re.compile(r"^(?P<path>.+?):(?P<start>\d+)-(?P<end>\d+)$")

# glob 特征字符. shell glob 也是这几种 (*, ?, [...]).
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class ParsedAddr:
    """一次性解析的 Pin.addr 结构.

    Frozen: 解析后不可变. 相同 raw 字符串多次解析结果等价, 可作 dict key.
    """

    kind: AddrKind
    """三分类: file / range / glob."""

    raw: str
    """原始 addr 字符串, 用于错误消息与显示."""

    path: str
    """
    file / range: 相对 Ground.root 的路径.
    glob: 相对 Ground.root 的模式 (原样, 与 raw 相同).
    """

    start: int | None = None
    """range only: 行号起点 (1-based, 含)."""

    end: int | None = None
    """range only: 行号终点 (1-based, 含)."""

    @property
    def is_glob(self) -> bool:
        return self.kind == "glob"


def parse_addr(addr: str) -> ParsedAddr:
    """把一个 addr 字符串解析成 ParsedAddr.

    优先级 (按 K16 骑先验的顺序):
    1. 含 glob 特征字符 (``* ? [``) → glob
    2. 命中 ``path:N-M`` 后缀模式 → range
    3. 其余 → file

    Raises:
        ValueError: 空字符串; range 端点 < 1; range start > end.
    """
    if not addr:
        raise ValueError("empty addr")

    if any(c in _GLOB_CHARS for c in addr):
        return ParsedAddr(kind="glob", raw=addr, path=addr)

    m = _RANGE_SUFFIX_RE.match(addr)
    if m is not None:
        start = int(m.group("start"))
        end = int(m.group("end"))
        if start < 1:
            raise ValueError(f"range start must be >= 1: {addr!r}")
        if end < 1:
            raise ValueError(f"range end must be >= 1: {addr!r}")
        if start > end:
            raise ValueError(f"range start > end: {addr!r}")
        return ParsedAddr(
            kind="range",
            raw=addr,
            path=m.group("path"),
            start=start,
            end=end,
        )

    return ParsedAddr(kind="file", raw=addr, path=addr)


def resolve_file_addr(parsed: ParsedAddr, root: Path) -> Path:
    """把 file / range 类型的 addr 解析为绝对路径, 校验在 root 子树内.

    不检查文件是否存在 — 那是 observe 的职责.

    Raises:
        ValueError: 传入 glob 类型的 ParsedAddr.
        PathOutsideRootError: 解析后的路径逃逸出 root (K12 空间边界).
    """
    if parsed.kind == "glob":
        raise ValueError(f"resolve_file_addr called on glob: {parsed.raw!r}")

    root_abs = root.resolve()
    # 允许 parsed.path 为绝对或相对; (root / abs) 保留 abs 语义, 之后
    # relative_to 会拒绝越界. 这样两种传法都能被统一验边界.
    target = (root_abs / parsed.path).resolve()
    try:
        target.relative_to(root_abs)
    except ValueError:
        raise PathOutsideRootError(
            f"addr {parsed.raw!r} escapes root {root_abs}"
        )
    return target


def resolve_glob_addr(parsed: ParsedAddr, root: Path) -> list[Path]:
    """把 glob 类型的 addr 展开为命中的文件绝对路径列表.

    返回按路径字符串排序 (为 hash 稳定性), 且已过滤:
    - 目录 (只留文件, glob pin 观察内容集合而非结构)
    - 逃逸出 root 的命中 (防御性, 正常 Path.glob 不产生)

    空命中集是合法状态, 返回空列表.

    Raises:
        ValueError: 传入非 glob 类型的 ParsedAddr.
    """
    if parsed.kind != "glob":
        raise ValueError(
            f"resolve_glob_addr called on {parsed.kind}: {parsed.raw!r}"
        )

    root_abs = root.resolve()
    matches: list[Path] = []
    for hit in root_abs.glob(parsed.path):
        try:
            hit.relative_to(root_abs)
        except ValueError:
            continue
        if hit.is_file():
            matches.append(hit)
    matches.sort()
    return matches
