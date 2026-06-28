"""Desktop 实现层内部模型.

公开数据模型 (FileContent / ExecResult / Match / Task / PinInfo /
DirectoryTree / ReflectionHint) 都在 ``ghoshell_moss.contracts.desktop``.
本模块只放实现内部专用的运行时记录, 不对外导出.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

__all__ = ["PinRecord", "InProcessReadHistory"]


@dataclass
class PinRecord:
    """Desktop 内部持有的 pin 运行时记录.

    LRU 顺序由 ``dict`` 插入顺序维护 — 每次命中重新 pop + 插入即可移到末尾.
    """

    id: str
    """pin 唯一 id, 由方法名 + 参数签名哈希得到."""

    command_name: str
    """被 pin 的原语名 — 用户可读."""

    args_preview: str
    """参数的人类可读摘要."""

    method_name: str
    """重执行用的方法名."""

    method_args: tuple
    """重执行用的位置参数 — 已剔除 ``_pin``."""

    method_kwargs: dict
    """重执行用的关键字参数 — 已剔除 ``_pin``."""

    is_async: bool = False
    """方法是否为 async."""

    last_output: str = ""
    """最近一次执行的截断预览."""

    error: str = ""
    """最近一次执行的错误信息. 空串 = 上次成功."""


class InProcessReadHistory:
    """缺省 ``ReadHistory`` 实现 — 进程内 ``set[Path]``.

    单测和单实例场景使用. Phase 4 由 Memento branch state 后置接管 —
    本类在那之后仍保留为单测和不依赖 Memento 的最小场景的默认值.
    """

    def __init__(self) -> None:
        self._paths: set = set()

    def has_read(self, path) -> bool:
        return path in self._paths

    def mark_read(self, path) -> None:
        self._paths.add(path)
