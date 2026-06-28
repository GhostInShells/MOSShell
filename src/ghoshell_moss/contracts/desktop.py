"""Desktop contract — Ghost 在文件系统上的工作桌面.

Desktop 是 Ghost 在 4 剪影拓扑中的"空间剪影 / 未来脏器" — 与 Memento (过去),
Matrix runtime (当下), Git worktree (结构版本) 共构反身性基建.

12+1 原语三层 (导航/发现/读写/执行/后台/Pin) + 两条元规则:
  read-before-write 守卫, 统一输出截断.

ABC 不依赖 Matrix / Memento / Session 任何具体实现. ReadHistory 通过
protocol 注入, 让 Ghost 的 epistemic state 可被 Memento branch 后置接管.
ReflectionHint 是 Desktop 给上层的反思信号, Desktop 自身不直接调 Memento.

Phase 1 设计稿: .design/2026-06-28_desktop_in_4d_cross_section.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "Desktop",
    "ReadHistory",
    "ReflectionHint",
    "FileContent",
    "ExecResult",
    "Match",
    "Task",
    "PinInfo",
    "DirectoryTree",
    "ReflectionSeverity",
    "DesktopError",
    "ReadBeforeWriteError",
    "PinBudgetExceeded",
    "PathOutsideRootError",
]

# -- 反思严重程度字面量 --

ReflectionSeverity = str
"""反思路径的严重程度. 取值: 'config' | 'instruction' | 'vcs'.

- ``config``:      .moss/, pyproject.toml, MOSS.md — 改了可能影响 ghost 装配
- ``instruction``: CLAUDE.md, DESKTOP.md          — 改了下一帧 ghost 行为受影响
- ``vcs``:         .git/                          — 改了仓库结构, 重风险
"""


# -- 协议: Ghost epistemic state 的注入点 --


@runtime_checkable
class ReadHistory(Protocol):
    """Ghost 在当前认知轨迹上读过哪些文件.

    Desktop 的 read-before-write 守卫语义是: "Ghost 在当前认知轨迹上至少
    看过这个文件". 这是 Ghost 的 epistemic state, 不是 Desktop instance
    的工具状态. 一个 Ghost session 内 Desktop 可能被多次实例化 (探不同
    子目录), read 历史必须穿透实例边界.

    缺省实现 (process-local set) 用于单测和单实例场景. Phase 4 由 Memento
    branch state 后置 — commit 时进快照, fork 时跟着 base pointer 继承,
    切换 branch 时切换 read history 上下文.
    """

    def has_read(self, path: Path) -> bool:
        """是否在当前 Ghost 认知轨迹上读过这个文件."""
        ...

    def mark_read(self, path: Path) -> None:
        """登记一次读取. 幂等."""
        ...


# -- 反思信号: write/edit 命中高影响路径时附带 --


@dataclass
class ReflectionHint:
    """高影响路径写入后, Desktop 给上层的反思信号.

    Desktop 不直接调 Memento. channel 封装层把 recommend_commit=True
    翻译为建议 emit ``<memento:commit summary=.../>``. 这是 reflection
    (事后给信号) + memento (提供锚点) 的最小协作单元, 为 Phase 5+ 的
    sandbox + keyframe 提供 pre-write anchor 机制.
    """

    path: str
    """命中白名单的文件路径 (相对 desktop root)."""

    diff_preview: str
    """变更的人类可读摘要. Phase 1 简化为 'create' / 'replace N chars'
    之类的短描述, 不强制是 unified diff."""

    severity: ReflectionSeverity
    """命中路径的严重程度. 用于上层决定是否真的推 memento commit."""

    recommend_commit: bool = True
    """是否建议立即在 Memento 上 commit. 缺省 True; 反向场景 (例如批量
    write 中间态) 上层可选择 false 收敛."""


# -- 数据模型: 命令返回值 --


@dataclass
class FileContent:
    """``read()`` 的返回值."""

    path: str
    """相对于 desktop root 的文件路径."""

    lines: list[tuple[int, str]]
    """带行号的内容 ``(line_number, text)``. 已应用 offset/limit."""

    total_lines: int
    """文件总行数."""

    start_line: int
    """返回内容的起始行号 (1-based, 继承 offset)."""

    truncated: bool = False
    """内容是否被截断写入 tmp."""

    tmp_path: str | None = None
    """完整内容的 tmp 路径. 仅 truncated=True 时有值. 用 ``read(tmp_path)``
    再次读取时不会再次截断."""


@dataclass
class ExecResult:
    """``exec()`` 的返回值. ``_bg=True`` 时退化为只有 task 字段."""

    stdout: str
    """标准输出, 可能截断."""

    stderr: str
    """标准错误, 可能截断."""

    exit_code: int
    """进程退出码. -1 表示 killed."""

    killed: bool = False
    """是否因超时被 kill."""

    truncated: bool = False
    """stdout 是否被截断."""

    stdout_tmp_path: str | None = None
    """stdout 完整内容的 tmp 路径. 仅 truncated=True 时有值."""

    stderr_tmp_path: str | None = None
    """stderr 完整内容的 tmp 路径. 仅 stderr 超阈值时有值."""

    task_id: int | None = None
    """``_bg=True`` 启动时返回的后台任务 id. 此时 stdout/stderr 为空."""


@dataclass
class Match:
    """``grep()`` 的单条匹配结果."""

    file: str
    """相对于 root 的文件路径."""

    line: int
    """1-based 行号."""

    text: str
    """匹配行的完整文本 (去尾换行)."""


@dataclass
class DirectoryTree:
    """``tree()`` 的返回值."""

    name: str
    """当前层级名称."""

    path: str
    """相对于 root 的路径."""

    type: str
    """``'dir'`` | ``'file'`` | ``'symlink'``."""

    children: list[DirectoryTree] | None = None
    """子项. ``None`` 表示文件 (无子节点); 空列表表示空目录或达到 depth 边界."""


@dataclass
class PinInfo:
    """``pinned()`` 返回的一条 pin 状态."""

    id: str
    """pin 唯一标识, 由方法名 + 参数签名哈希得到."""

    command_name: str
    """被 pin 的原语名 (``exec`` / ``tree`` / ``glob`` / ...)."""

    args_preview: str
    """参数的人类可读摘要 — 用于模型识别."""

    last_preview: str
    """最近一次执行输出的截断预览."""

    error: str = ""
    """最近一次执行的错误信息. 空串 = 上一次成功."""

    pin_budget_warning: bool = False
    """命中 ``max_pins`` 上限时为 True — 提示模型注意取舍 (LRU 淘汰仍在生效)."""


@dataclass
class Task:
    """``tasks()`` 返回的一条后台任务句柄.

    Task 本身持 ``read()`` / ``cancel()`` 接口, 收掉独立的 ``read_task``
    / ``cancel`` 顶层原语 — 一个对象一组动作.
    """

    id: int
    """任务唯一标识."""

    command: str
    """执行的 shell 命令."""

    loop: int
    """总循环次数. 0 = 无限."""

    executed: int
    """已完成次数."""

    alive: bool
    """进程是否仍在运行."""

    return_code: int | None = None
    """最近一次执行的退出码. None 表示尚未完成或仍在运行."""

    stdout_preview: str = ""
    """最近一次 stdout 的截断预览."""

    # 行为契约 — Desktop 实现侧填充以下回调, 模型直接调用
    _read: object = field(default=None, repr=False)
    """``async (offset: int, limit: int) -> str`` — 读取该任务输出窗口."""

    _cancel: object = field(default=None, repr=False)
    """``async () -> None`` — 取消该任务."""

    async def read(self, *, offset: int = 0, limit: int = 100) -> str:
        """读取后台任务输出窗口."""
        if self._read is None:
            raise RuntimeError(f"Task {self.id}: read callback not bound")
        return await self._read(offset, limit)

    async def cancel(self) -> None:
        """取消后台任务."""
        if self._cancel is None:
            raise RuntimeError(f"Task {self.id}: cancel callback not bound")
        await self._cancel()


# -- 异常 --


class DesktopError(Exception):
    """Desktop 基础异常."""


class ReadBeforeWriteError(DesktopError, PermissionError):
    """write / edit 前未 read 目标文件触发. 同时是 PermissionError 子类以
    兼容上层捕获."""


class PathOutsideRootError(DesktopError, ValueError):
    """路径越出 desktop root 子树 (含 cd 和绝对路径访问)."""


class PinBudgetExceeded(DesktopError):
    """显式 pin 命中预算上限的尝试 (LRU 淘汰路径不抛异常, 只在 PinInfo 上
    带 warning 标记)."""


# -- 主契约 --


class Desktop(ABC):
    """Ghost 在文件系统上的工作桌面.

    Desktop 不关心 root 的"身份" — 可以是 project root, ghost home,
    mode home, 或任意目录. 使用方通过 ``DESKTOP.md`` 或 channel 层的
    instruction 告知模型此 Desktop 的用途.

    所有路径相对 ``pwd`` 解析. 绝对路径必须在 ``root`` 或 ``tmp_root``
    子树内 — Desktop 提供空间边界保证.

    元规则:
    - **read-before-write**: write / edit 之前必须先在 ReadHistory 上
      登记过对应文件
    - **统一输出截断**: 任何超阈值的命令输出都自动落 tmp, 返回截断预览
      + tmp_path; ``read(tmp_path)`` 不再截断

    详细契约见 ``.design/2026-06-28_desktop_in_4d_cross_section.md``.
    """

    # -- 拓扑属性 --

    @property
    @abstractmethod
    def root(self) -> Path:
        """Desktop 的空间边界. 构造期固定."""
        ...

    @property
    @abstractmethod
    def tmp_root(self) -> Path:
        """截断输出的回收目录. 构造参数, 缺省 ``root/tmp/desktop/``.

        Phase 4 可指向 Memento storage 提供的目录."""
        ...

    @abstractmethod
    def instruction(self) -> str:
        """生成给模型的入口指令. 若 ``root/DESKTOP.md`` 存在则用它覆盖
        默认模板, 否则返回内置模板 (含 root / pwd / 原语列表 / 规则摘要)."""
        ...

    # -- 导航层 --

    @abstractmethod
    def cd(self, path: str) -> str:
        """切换工作目录. 返回切换后绝对路径. 越界抛 PathOutsideRootError."""
        ...

    @abstractmethod
    def pwd(self) -> str:
        """当前工作目录的绝对路径."""
        ...

    # -- 发现层 --

    @abstractmethod
    def tree(
        self,
        depth: int = 2,
        *,
        path: str = ".",
        _pin: bool = False,
    ) -> DirectoryTree:
        """目录结构. 子项标注类型 (file / dir / symlink). 隐藏文件忽略."""
        ...

    @abstractmethod
    def glob(self, pattern: str, *, _pin: bool = False) -> list[str]:
        """匹配文件路径, 返回相对 root 的路径列表. 支持 ``**`` 递归."""
        ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        _pin: bool = False,
    ) -> list[Match]:
        """搜索文件内容. 正则匹配. 尊重 ``.gitignore`` (Phase 1 简化:
        跳过 ``.`` 前缀目录)."""
        ...

    # -- 读取层 --

    @abstractmethod
    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 200,
        _pin: bool = False,
    ) -> FileContent:
        """读文件. 超阈值落 tmp. ``path`` 为 tmp_root 内的路径时不再截断.

        读取成功后 ReadHistory 上登记一次, 解锁 write / edit."""
        ...

    @abstractmethod
    def frontmatter(self, path: str, *keys: str) -> dict | None:
        """提取 markdown YAML frontmatter. ``keys`` 指定时只返回这些键.

        Desktop 不硬编码 ``CLAUDE.md`` / ``SKILL.md`` 等约定 — 约定由
        使用方 (ghost system prompt / DESKTOP.md / channel instruction)
        下达. 这是提取原语, 不是约定. L1 试用后可决定去留."""
        ...

    # -- 写入层 --

    @abstractmethod
    def write(self, path: str, content: str) -> ReflectionHint | None:
        """创建或覆盖文件. 必须 ReadHistory 上已登记 (新文件路径除外).

        命中反思路径白名单时返回 ReflectionHint, 否则返回 None."""
        ...

    @abstractmethod
    def edit(self, path: str, old: str, new: str) -> tuple[int, ReflectionHint | None]:
        """替换文件中的字符串. ``old`` 必须精确匹配一次. 返回 (替换处行号,
        ReflectionHint | None)."""
        ...

    # -- 执行层 (12+1 收缩: exec 吃 _bg, tasks 返回带方法的 Task) --

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        timeout: float = 60.0,
        _bg: bool = False,
        loop: int = 1,
        _pin: bool = False,
    ) -> ExecResult:
        """执行 shell 命令.

        - ``_bg=False`` (默认): 阻塞到完成或超时. 超时 kill 进程组,
          返回 ``killed=True``.
        - ``_bg=True``:  立即返回 ExecResult(task_id=N, stdout='',...).
          ``loop=0`` 无限循环, ``loop=N`` 执行 N 次. 通过 ``tasks()``
          查询和管理."""
        ...

    @abstractmethod
    def tasks(self, *, _pin: bool = False) -> list[Task]:
        """活跃后台任务列表. 每个 Task 持 ``read()`` / ``cancel()`` 方法."""
        ...

    # -- Pin 管理 --

    @abstractmethod
    def pinned(self) -> list[PinInfo]:
        """所有活跃 pin 的状态快照. LRU 淘汰已发生时仅出现在剩余条目上."""
        ...

    @abstractmethod
    def unpin(self, pin_id: str) -> None:
        """移除一个 pin. 不存在抛 ``KeyError``."""
        ...

    @abstractmethod
    async def refresh(self) -> None:
        """重执行所有活跃 pin. 由 channel ``refresh_meta`` 回调驱动;
        独立调用 (单测 / 手动) 也可以."""
        ...

    # -- 生命周期 --

    @abstractmethod
    async def shutdown(self) -> None:
        """清理所有后台进程. 幂等."""
        ...
