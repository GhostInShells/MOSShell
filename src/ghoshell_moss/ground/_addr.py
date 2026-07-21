"""Path resolution with anchor syntax (SPEC §8).

三锚点: $GROUND (法锚) / $CWD (属地锚) / $HOME (机器逃生口).
裸相对路径默认 = $GROUND. 解析后校验 per-anchor subtree confinement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ghoshell_moss.ground.contract import PathOutsideRootError

__all__ = [
    "Anchor",
    "resolve_path",
    "is_glob_pattern",
]

_ANCHOR_PREFIXES = {"$GROUND", "$CWD", "$HOME"}


@dataclass(frozen=True)
class Anchor:
    """一次 open 的锚点上下文.

    不可变: 一次 open 锚点不变, frame 间共享.
    """

    ground: Path
    """$GROUND — 被加载的 GROUND.md 所在目录 (法锚)."""

    cwd: Path
    """$CWD — open(dir=...) 传入的目录 (属地锚)."""


def resolve_path(
    raw: str,
    anchor: Anchor,
) -> Path:
    """将锚点语法路径解析为绝对路径, 校验 per-anchor confinement.

    规则 (SPEC §8):
    - $GROUND/path → 以 ground anchor 为锚
    - $CWD/path    → 以 cwd anchor 为锚
    - $HOME/path   → 以 $HOME 为锚
    - 裸相对路径  → 默认 $GROUND
    - 转义 ``\\$`` → 字面量 ``$`` (罕见: 文件名校 $)
    - 越界 ``..``  → PathOutsideRootError

    不检查文件是否存在 — caller 职责.
    """
    # 转义: \$ → 字面量 $
    unescaped = raw.replace("\\$", "$")

    # 选择锚点
    resolved_anchor: Path
    if unescaped.startswith("$GROUND/"):
        resolved_anchor = anchor.ground
        rel = unescaped[len("$GROUND/") :]
    elif unescaped.startswith("$CWD/"):
        resolved_anchor = anchor.cwd
        rel = unescaped[len("$CWD/") :]
    elif unescaped.startswith("$HOME/"):
        resolved_anchor = _home()
        rel = unescaped[len("$HOME/") :]
    elif unescaped.startswith("$GROUND") or unescaped.startswith("$CWD") or unescaped.startswith("$HOME"):
        # 锚点后没有 '/' — 裸锚点, 非法
        raise PathOutsideRootError(f"anchor missing path separator: {raw!r}")
    elif unescaped.startswith("/"):
        raise PathOutsideRootError(
            f"bare absolute path not allowed: {raw!r}. Use $HOME prefix."
        )
    else:
        # 裸相对路径 → $GROUND
        resolved_anchor = anchor.ground
        rel = unescaped

    # 解析: anchor (resolve) / rel → resolve() → 校验 subtree
    resolved_anchor = resolved_anchor.resolve()
    target = (resolved_anchor / rel).resolve()
    try:
        target.relative_to(resolved_anchor)
    except ValueError:
        raise PathOutsideRootError(
            f"{raw!r} escapes anchor subtree {resolved_anchor}"
        )

    return target


def is_glob_pattern(raw: str) -> bool:
    """判定路径是否含 glob 特征字符."""
    unescaped = raw.replace("\\$", "$")
    return any(c in unescaped for c in "*?[")


def _home() -> Path:
    h = os.environ.get("HOME")
    return Path(h).resolve() if h else Path("/").resolve()
