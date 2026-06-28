"""Desktop 实现 — Ghost 文件系统工作桌面的具体实现.

契约见 ``ghoshell_moss.contracts.desktop``. 本模块实现:
- 12+1 原语 (head 删除, exec_bg 合入 exec(_bg=True), tasks() 返回带方法的 Task)
- 两条元规则 (read-before-write 经 ReadHistory protocol, 统一输出截断)
- LRU pin 预算 (max_pins 构造参数, 溢出时淘汰 + 标记 warning)
- 反思路径白名单 → ReflectionHint
- ProcessManager 可选注入; 未注入时退化为裸 asyncio subprocess

实现纪律:
- 不 import Matrix / Memento / Session 任何具体实现
- ReadHistory 通过构造参数注入, 默认走 InProcessReadHistory
- tmp_root 通过构造参数注入, 默认 root/tmp/desktop/
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import signal
import time
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import frontmatter as _frontmatter_lib

from ghoshell_moss.contracts.desktop import (
    Desktop as DesktopABC,
    ReadHistory,
    ReflectionHint,
    FileContent,
    ExecResult,
    Match,
    Task,
    PinInfo,
    DirectoryTree,
    ReadBeforeWriteError,
    PathOutsideRootError,
    PinBudgetExceeded,
)
from ghoshell_moss.core.desktop.models import PinRecord, InProcessReadHistory

if TYPE_CHECKING:
    from ghoshell_moss.contracts.process_manager import ProcessManager

__all__ = ["DefaultDesktop", "DEFAULT_INSTRUCTION", "DEFAULT_REFLECTION_PATHS"]


# -- 截断阈值 --
_LINE_THRESHOLD = 200
_BYTE_THRESHOLD = 32 * 1024  # 32KB

# -- Pin 预算缺省 --
_DEFAULT_MAX_PINS = 16

# -- 反思路径白名单缺省值. 命中触发 ReflectionHint --
DEFAULT_REFLECTION_PATHS: dict[str, str] = {
    ".moss": "config",
    ".moss/": "config",
    ".git": "vcs",
    ".git/": "vcs",
    "DESKTOP.md": "instruction",
    "CLAUDE.md": "instruction",
    "MOSS.md": "config",
    "pyproject.toml": "config",
}

# -- grep 默认扫描的文本后缀 --
_GREP_TEXT_SUFFIXES = (
    ".py", ".md", ".yml", ".yaml", ".json", ".toml", ".txt", ".sh",
    ".cfg", ".ini", ".xml", ".html", ".css", ".js", ".ts", ".rs",
    ".go", ".java", ".c", ".h", ".cpp", ".hpp",
)


DEFAULT_INSTRUCTION = """\
Your Desktop at {root}. Current: {pwd}.

NAVIGATE
  cd / pwd

DISCOVER
  tree / glob / grep

READ
  read  — supports offset/limit, large outputs auto-spill to tmp
  frontmatter — extract YAML metadata from markdown (optional primitive)

WRITE
  write / edit  — must have read the target in this session first
  hits on .moss/ / .git/ / DESKTOP.md / CLAUDE.md / MOSS.md / pyproject.toml
  return a ReflectionHint (consider committing in Memento before proceeding)

EXECUTE
  exec(command, ..., _bg=False, loop=1)
    _bg=False  — block until complete or timeout
    _bg=True   — return immediately; manage via tasks()
  tasks()  — list active background tasks; each Task has .read() / .cancel()

PERSISTENT VIEWS (pins)
  Add _pin=True to any discover/read/exec call to register a periodic
  re-execution. pinned() lists them, unpin(id) removes one.
  Budget: max {max_pins} concurrent pins; LRU eviction with warning.

RULES
  - cd / read / write / exec are confined to root subtree
  - write / edit require prior read of target (read-before-write guard)
  - output > 200 lines or 32KB → truncated preview + tmp_path; read(tmp_path) is exempt
"""


class DefaultDesktop(DesktopABC):
    """Desktop 的进程内默认实现.

    构造参数全部为关键字, 让 Desktop instance 完全由调用方决定形状:
    - ``root``:           空间边界 (必需)
    - ``tmp_root``:       截断回收目录 (可选, 默认 root/tmp/desktop/)
    - ``read_history``:   ReadHistory 协议实现 (可选, 默认 InProcessReadHistory)
    - ``process_manager``:ProcessManager 注入 (可选, 默认走裸 asyncio)
    - ``max_pins``:       Pin LRU 预算 (可选, 默认 16)
    - ``reflection_paths``: 反思路径白名单 {pattern: severity} (可选, 默认见上)
    """

    def __init__(
        self,
        root: Path,
        *,
        tmp_root: Path | None = None,
        read_history: ReadHistory | None = None,
        process_manager: "ProcessManager | None" = None,
        max_pins: int = _DEFAULT_MAX_PINS,
        reflection_paths: dict[str, str] | None = None,
        name: str = "desktop",
        description: str = "",
    ):
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Desktop root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Desktop root is not a directory: {root}")

        self._root = root
        self._pwd = self._root
        self._name = name
        self._description = description
        self._pm = process_manager
        self._max_pins = max(1, max_pins)
        self._reflection_paths = dict(
            reflection_paths if reflection_paths is not None else DEFAULT_REFLECTION_PATHS
        )

        self._read_history: ReadHistory = read_history or InProcessReadHistory()

        if tmp_root is None:
            tmp_root = self._root / "tmp" / "desktop"
        self._tmp_root = tmp_root.resolve()
        self._tmp_root.mkdir(parents=True, exist_ok=True)

        # LRU 顺序: 以 OrderedDict 维护; 最近访问的在末尾
        self._pins: "OrderedDict[str, PinRecord]" = OrderedDict()

        # 裸 subprocess 兜底用的后台任务索引
        self._bg_procs: dict[int, asyncio.subprocess.Process] = {}
        self._bg_meta: dict[int, dict] = {}  # command/loop/executed/last_stdout
        self._bg_counter = 0
        self._bg_tasks_handles: dict[int, asyncio.Task] = {}

        # shutdown 时统一 kill 的所有进程
        self._procs: set[asyncio.subprocess.Process] = set()

    # ================================================================
    # 拓扑属性
    # ================================================================

    @property
    def root(self) -> Path:
        return self._root

    @property
    def tmp_root(self) -> Path:
        return self._tmp_root

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def instruction(self) -> str:
        desktop_md = self._root / "DESKTOP.md"
        if desktop_md.is_file():
            return desktop_md.read_text(encoding="utf-8")
        return DEFAULT_INSTRUCTION.format(
            root=str(self._root), pwd=str(self._pwd), max_pins=self._max_pins,
        )

    # ================================================================
    # 导航层
    # ================================================================

    def cd(self, path: str) -> str:
        p = Path(path)
        target = p.resolve() if p.is_absolute() else (self._pwd / p).resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise PathOutsideRootError(
                f"cd: '{path}' is outside desktop root {self._root}"
            )
        if not target.is_dir():
            raise FileNotFoundError(f"cd: '{path}' is not a directory")
        self._pwd = target
        return str(self._pwd)

    def pwd(self) -> str:
        return str(self._pwd)

    # ================================================================
    # 发现层
    # ================================================================

    def tree(
        self,
        depth: int = 2,
        *,
        path: str = ".",
        _pin: bool = False,
    ) -> DirectoryTree:
        target = self._resolve(path)
        result = self._build_tree(target, depth)
        if _pin:
            self._record_pin(
                "tree", (), {"depth": depth, "path": path},
                self._truncate_preview(self._format_tree(result)),
            )
        return result

    def _build_tree(self, target: Path, depth: int) -> DirectoryTree:
        name = target.name or str(target)
        relative = self._relative(target)

        if target.is_symlink():
            return DirectoryTree(name=name, path=relative, type="symlink")
        if not target.is_dir():
            return DirectoryTree(name=name, path=relative, type="file")

        children: list[DirectoryTree] = []
        if depth > 0:
            try:
                entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                entries = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    children.append(self._build_tree(entry, depth - 1))
                else:
                    children.append(DirectoryTree(
                        name=entry.name,
                        path=self._relative(entry),
                        type="symlink" if entry.is_symlink() else "file",
                    ))
        return DirectoryTree(name=name, path=relative, type="dir", children=children)

    def glob(self, pattern: str, *, _pin: bool = False) -> list[str]:
        matches = sorted(
            self._relative(p)
            for p in self._pwd.glob(pattern)
            if not any(part.startswith(".") for part in p.relative_to(self._root).parts)
        )
        if _pin:
            self._record_pin(
                "glob", (pattern,), {},
                self._truncate_preview("\n".join(matches)),
            )
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        _pin: bool = False,
    ) -> list[Match]:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"grep: invalid regex '{pattern}': {e}") from e

        search_dir = self._resolve(path)
        results: list[Match] = []
        for f in search_dir.rglob("*"):
            try:
                rel_parts = f.relative_to(self._root).parts
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel_parts):
                continue
            if not f.is_file():
                continue
            if f.suffix not in _GREP_TEXT_SUFFIXES:
                continue
            try:
                content = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(Match(
                        file=self._relative(f),
                        line=i,
                        text=line.rstrip(),
                    ))
        if _pin:
            preview = "\n".join(f"{m.file}:{m.line}: {m.text}" for m in results[:20])
            self._record_pin(
                "grep", (pattern,), {"path": path},
                self._truncate_preview(preview),
            )
        return results

    # ================================================================
    # 读取层
    # ================================================================

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 200,
        _pin: bool = False,
    ) -> FileContent:
        file_path = self._resolve(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"read: '{path}' does not exist or is not a file")

        is_tmp = self._is_tmp_path(file_path)

        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = file_path.read_bytes().decode("utf-8", errors="replace")
        all_lines = raw.splitlines()
        total = len(all_lines)

        start = max(0, offset)
        end = min(total, start + limit)
        selected = all_lines[start:end]
        lines = [(start + i + 1, selected[i]) for i in range(len(selected))]

        relative = self._relative(file_path)
        content = "\n".join(selected)

        # tmp_root 内的文件不再触发截断 — 元规则要求
        if is_tmp:
            truncated, tmp_path = False, None
        else:
            truncated, tmp_path = self._maybe_truncate(content, file_path.name)

        result = FileContent(
            path=relative,
            lines=lines,
            total_lines=total,
            start_line=start + 1,
            truncated=truncated,
            tmp_path=tmp_path,
        )
        # 只有 root 内的文件登记 read history; tmp 文件本身是 desktop 内部产物,
        # 登记没有语义价值
        if not is_tmp:
            self._read_history.mark_read(file_path)
        if _pin:
            preview = self._format_file_content(result)
            self._record_pin(
                "read", (path,), {"offset": offset, "limit": limit}, preview,
            )
        return result

    def frontmatter(self, path: str, *keys: str) -> dict | None:
        file_path = self._resolve(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"frontmatter: '{path}' does not exist")
        try:
            post = _frontmatter_lib.load(str(file_path))
        except Exception:
            return None
        metadata = post.metadata
        if not metadata:
            return None
        if keys:
            return {k: metadata.get(k) for k in keys if k in metadata}
        return dict(metadata)

    # ================================================================
    # 写入层
    # ================================================================

    def write(self, path: str, content: str) -> ReflectionHint | None:
        file_path = self._resolve(path)
        existed = file_path.exists()
        if existed:
            self._check_writable(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        # 新建文件视为已读 (写入即是当前 epistemic 锚点)
        self._read_history.mark_read(file_path)

        diff_preview = (
            f"create ({len(content)} chars)" if not existed
            else f"replace ({len(content)} chars)"
        )
        return self._reflection_for(file_path, diff_preview)

    def edit(self, path: str, old: str, new: str) -> tuple[int, ReflectionHint | None]:
        file_path = self._resolve(path)
        self._check_writable(file_path)
        current = file_path.read_text(encoding="utf-8")
        count = current.count(old)
        if count == 0:
            raise ValueError(f"edit: '{old[:80]}' not found in '{path}'")
        if count > 1:
            raise ValueError(
                f"edit: '{old[:80]}' matches {count} times in '{path}' — "
                f"old must match exactly once"
            )
        updated = current.replace(old, new, 1)
        file_path.write_text(updated, encoding="utf-8")
        self._read_history.mark_read(file_path)

        before = current[: current.index(old)]
        line_no = before.count("\n") + 1
        diff_preview = f"edit @line {line_no}: -{len(old)}/+{len(new)} chars"
        hint = self._reflection_for(file_path, diff_preview)
        return line_no, hint

    # ================================================================
    # 执行层
    # ================================================================

    async def exec(
        self,
        command: str,
        *,
        timeout: float = 60.0,
        _bg: bool = False,
        loop: int = 1,
        _pin: bool = False,
    ) -> ExecResult:
        if _bg:
            task_id = await self._spawn_bg(command, loop=loop)
            result = ExecResult(stdout="", stderr="", exit_code=0, task_id=task_id)
            if _pin:
                self._record_pin(
                    "exec", (command,), {"timeout": timeout, "_bg": True, "loop": loop},
                    f"[bg task={task_id}] {command}", is_async=True,
                )
            return result

        if self._pm is not None:
            result = await self._exec_via_pm(command, timeout=timeout)
        else:
            result = await self._exec_raw(command, timeout=timeout)

        if _pin:
            self._record_pin(
                "exec", (command,), {"timeout": timeout},
                self._truncate_preview(result.stdout), is_async=True,
            )
        return result

    def tasks(self, *, _pin: bool = False) -> list[Task]:
        result: list[Task] = []
        if self._pm is not None:
            for bg in self._pm.background_tasks():
                tid = hash(bg.id) & 0x7FFFFFFF
                last_stdout = bg.last_stdout() if hasattr(bg, "last_stdout") else ""
                task = Task(
                    id=tid,
                    command=bg.name,
                    loop=bg.loop,
                    executed=bg.executed,
                    alive=bg.is_running,
                    return_code=(bg.last.return_code if bg.last else None),
                    stdout_preview=self._truncate_preview(last_stdout or ""),
                )
                task._read = self._pm_task_reader(bg)
                task._cancel = self._pm_task_canceller(bg)
                result.append(task)
        else:
            for tid, proc in list(self._bg_procs.items()):
                meta = self._bg_meta.get(tid, {})
                rc = proc.returncode
                task = Task(
                    id=tid,
                    command=meta.get("command", f"bg:{tid}"),
                    loop=meta.get("loop", 1),
                    executed=meta.get("executed", 0),
                    alive=rc is None,
                    return_code=rc,
                    stdout_preview=self._truncate_preview(meta.get("last_stdout", "")),
                )
                task._read = self._raw_task_reader(tid)
                task._cancel = self._raw_task_canceller(tid)
                result.append(task)

        if _pin:
            preview = "\n".join(
                f"[{t.id}] alive={t.alive} rc={t.return_code} {t.command}" for t in result
            )
            self._record_pin(
                "tasks", (), {}, self._truncate_preview(preview),
            )
        return result

    # ================================================================
    # Pin 管理
    # ================================================================

    def pinned(self) -> list[PinInfo]:
        over_budget = len(self._pins) >= self._max_pins
        return [
            PinInfo(
                id=r.id,
                command_name=r.command_name,
                args_preview=r.args_preview,
                last_preview=self._truncate_preview(r.last_output),
                error=r.error,
                pin_budget_warning=over_budget,
            )
            for r in self._pins.values()
        ]

    def unpin(self, pin_id: str) -> None:
        if pin_id not in self._pins:
            raise KeyError(f"pin '{pin_id}' not found")
        del self._pins[pin_id]

    async def refresh(self) -> None:
        for record in list(self._pins.values()):
            try:
                method = getattr(self, record.method_name)
                if record.is_async:
                    result = await method(*record.method_args, **record.method_kwargs)
                else:
                    result = method(*record.method_args, **record.method_kwargs)
                record.last_output = self._extract_preview(result, record.command_name)
                record.error = ""
            except Exception as e:
                record.error = f"{type(e).__name__}: {e}"

    # ================================================================
    # 生命周期
    # ================================================================

    async def shutdown(self) -> None:
        # 取消所有 bg task handles
        for handle in list(self._bg_tasks_handles.values()):
            handle.cancel()
        # 优雅信号
        for proc in list(self._procs):
            if proc.returncode is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        # 等待 + 强杀
        for proc in list(self._procs):
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
        self._procs.clear()
        self._bg_procs.clear()
        self._bg_meta.clear()
        self._bg_tasks_handles.clear()

    # ================================================================
    # 内部: 路径解析与安全
    # ================================================================

    def _resolve(self, path: str) -> Path:
        """解析路径. 相对 → 相对 pwd; 绝对 → 必须在 root 或 tmp_root 子树内."""
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
            if self._is_within(resolved, self._root) or self._is_within(resolved, self._tmp_root):
                return resolved
            raise PathOutsideRootError(
                f"path '{path}' is outside desktop root {self._root} and tmp_root {self._tmp_root}"
            )
        return (self._pwd / p).resolve()

    def _relative(self, p: Path) -> str:
        """统一以 root 为基, root 外的(如 tmp)使用绝对路径."""
        try:
            rel = p.relative_to(self._root)
            return str(rel) if str(rel) != "." else "."
        except ValueError:
            return str(p)

    @staticmethod
    def _is_within(target: Path, base: Path) -> bool:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False

    def _check_writable(self, file_path: Path) -> None:
        resolved = file_path.resolve()
        if not self._read_history.has_read(resolved):
            try:
                hint_path = file_path.relative_to(self._root)
            except ValueError:
                hint_path = file_path.name
            raise ReadBeforeWriteError(
                f"Cannot write to '{file_path.name}' — read it first in this session. "
                f"Use read('{hint_path}') before write/edit."
            )

    # ================================================================
    # 内部: 输出截断
    # ================================================================

    def _is_tmp_path(self, p: Path) -> bool:
        return self._is_within(p.resolve(), self._tmp_root.resolve())

    def _maybe_truncate(self, content: str, filename: str = "") -> tuple[bool, str | None]:
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        n_bytes = len(content.encode("utf-8", errors="replace"))
        if n_lines <= _LINE_THRESHOLD and n_bytes <= _BYTE_THRESHOLD:
            return False, None
        suffix = hashlib.sha256(f"{filename}{time.time_ns()}".encode()).hexdigest()[:8]
        tmp_file = self._tmp_root / f"{filename or 'out'}_{suffix}"
        tmp_file.write_text(content, encoding="utf-8")
        return True, str(tmp_file)

    @staticmethod
    def _truncate_preview(text: str, max_chars: int = 200) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _truncate_for_return(self, content: str, truncated: bool) -> str:
        if not truncated:
            return content
        lines = content.splitlines()
        preview = "\n".join(lines[:_LINE_THRESHOLD])
        preview += f"\n[...] {len(lines)} lines total"
        return preview

    # ================================================================
    # 内部: 反思路径
    # ================================================================

    def _reflection_for(self, file_path: Path, diff_preview: str) -> ReflectionHint | None:
        try:
            rel = str(file_path.relative_to(self._root))
        except ValueError:
            return None  # tmp 内文件不触发反思
        for pattern, severity in self._reflection_paths.items():
            if self._reflection_match(rel, pattern):
                return ReflectionHint(
                    path=rel,
                    diff_preview=diff_preview,
                    severity=severity,
                    recommend_commit=True,
                )
        return None

    @staticmethod
    def _reflection_match(rel: str, pattern: str) -> bool:
        # 目录前缀: 以 '/' 结尾或纯目录名
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            return rel == prefix or rel.startswith(prefix + "/")
        # glob 风格匹配
        if any(ch in pattern for ch in "*?["):
            return fnmatch(rel, pattern)
        # 精确名: 可能是顶层文件名 (CLAUDE.md) 或目录名 (.git, .moss)
        if rel == pattern:
            return True
        if rel.startswith(pattern + "/"):
            return True
        return False

    # ================================================================
    # 内部: Pin 记录 + LRU
    # ================================================================

    def _record_pin(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        preview: str,
        is_async: bool = False,
    ) -> None:
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "_pin"}
        pin_id = hashlib.sha256(
            f"{method_name}:{args}:{sorted(kwargs_clean.items())}".encode()
        ).hexdigest()[:12]
        args_preview = self._format_args_preview(method_name, args, kwargs_clean)

        # 命中已有: 覆盖 + 移到末尾 (LRU 最近)
        if pin_id in self._pins:
            self._pins.pop(pin_id)
        else:
            # 新 pin: 必要时 LRU 淘汰
            while len(self._pins) >= self._max_pins:
                self._pins.popitem(last=False)
        self._pins[pin_id] = PinRecord(
            id=pin_id,
            command_name=method_name,
            args_preview=args_preview,
            method_name=method_name,
            method_args=args,
            method_kwargs=kwargs_clean,
            is_async=is_async,
            last_output=preview,
        )

    @staticmethod
    def _format_args_preview(method_name: str, args: tuple, kwargs: dict) -> str:
        parts = [repr(a) for a in args]
        parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{method_name}({', '.join(parts)})"

    def _extract_preview(self, result, command_name: str) -> str:
        if result is None:
            return "(none)"
        if isinstance(result, str):
            return self._truncate_preview(result)
        if isinstance(result, list):
            if command_name == "grep":
                items = [f"{m.file}:{m.line}: {m.text}" for m in result[:20]]
            else:
                items = [str(item) for item in result[:20]]
            return self._truncate_preview("\n".join(items))
        if isinstance(result, FileContent):
            return self._truncate_preview(self._format_file_content(result))
        if isinstance(result, ExecResult):
            return self._truncate_preview(result.stdout or f"[bg task={result.task_id}]")
        if isinstance(result, DirectoryTree):
            return self._truncate_preview(self._format_tree(result))
        return self._truncate_preview(str(result))

    @staticmethod
    def _format_file_content(fc: FileContent) -> str:
        preview = "\n".join(f"{num:6d}  {text}" for num, text in fc.lines[:10])
        if fc.truncated:
            preview += f"\n... ({fc.total_lines} lines total, full at {fc.tmp_path})"
        return preview

    @staticmethod
    def _format_tree(node: DirectoryTree, indent: int = 0) -> str:
        prefix = "  " * indent + ("- " if indent > 0 else "")
        line = f"{prefix}{node.name}/" if node.type == "dir" else f"{prefix}{node.name}"
        lines = [line]
        if node.children:
            for child in node.children:
                lines.append(DefaultDesktop._format_tree(child, indent + 1))
        return "\n".join(lines)

    # ================================================================
    # 内部: exec 实现
    # ================================================================

    async def _exec_via_pm(self, command: str, *, timeout: float) -> ExecResult:
        task = await self._pm.execute_task(
            "sh", "-c", command, cwd=self._pwd, name="desktop.exec",
        )
        killed = False
        try:
            await asyncio.wait_for(task.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            task.kill()
            try:
                await task.wait()
            except Exception:
                pass
            killed = True

        stdout = task.stdout_buffer()
        stderr = task.stderr_buffer()
        truncated_out, tmp_out = self._maybe_truncate(stdout, "stdout")
        truncated_err, tmp_err = self._maybe_truncate(stderr, "stderr")
        exit_code = task.return_code if task.return_code is not None else (-1 if killed else 0)
        return ExecResult(
            stdout=self._truncate_for_return(stdout, truncated_out),
            stderr=self._truncate_for_return(stderr, truncated_err),
            exit_code=exit_code,
            killed=killed,
            truncated=truncated_out,
            stdout_tmp_path=tmp_out,
            stderr_tmp_path=tmp_err,
        )

    async def _exec_raw(self, command: str, *, timeout: float) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._pwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._procs.add(proc)
        killed = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            killed = True
            try:
                if proc.returncode is None:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""
        finally:
            self._procs.discard(proc)

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        truncated_out, tmp_out = self._maybe_truncate(stdout, "stdout")
        truncated_err, tmp_err = self._maybe_truncate(stderr, "stderr")
        exit_code = proc.returncode if proc.returncode is not None else (-1 if killed else 0)
        return ExecResult(
            stdout=self._truncate_for_return(stdout, truncated_out),
            stderr=self._truncate_for_return(stderr, truncated_err),
            exit_code=exit_code,
            killed=killed,
            truncated=truncated_out,
            stdout_tmp_path=tmp_out,
            stderr_tmp_path=tmp_err,
        )

    # ================================================================
    # 内部: 后台任务
    # ================================================================

    async def _spawn_bg(self, command: str, *, loop: int) -> int:
        if self._pm is not None:
            return await self._spawn_bg_via_pm(command, loop=loop)
        return await self._spawn_bg_raw(command, loop=loop)

    async def _spawn_bg_via_pm(self, command: str, *, loop: int) -> int:
        task = await self._pm.execute_task(
            "sh", "-c", command,
            cwd=self._pwd,
            name=f"desktop.bg",
            background_run=("loop", loop),
        )
        return hash(task.id) & 0x7FFFFFFF

    async def _spawn_bg_raw(self, command: str, *, loop: int) -> int:
        self._bg_counter += 1
        task_id = self._bg_counter
        self._bg_meta[task_id] = {
            "command": command, "loop": loop, "executed": 0, "last_stdout": "",
        }

        async def _runner():
            iteration = 0
            while loop == 0 or iteration < loop:
                iteration += 1
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(self._pwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                self._bg_procs[task_id] = proc
                self._procs.add(proc)
                try:
                    out, _ = await proc.communicate()
                except Exception:
                    out = b""
                finally:
                    self._procs.discard(proc)
                self._bg_meta[task_id]["executed"] = iteration
                self._bg_meta[task_id]["last_stdout"] = out.decode("utf-8", errors="replace")
                if proc.returncode != 0 and loop > 0:
                    break

        self._bg_tasks_handles[task_id] = asyncio.create_task(_runner())
        return task_id

    def _raw_task_reader(self, task_id: int) -> Callable[[int, int], Awaitable[str]]:
        async def _read(offset: int, limit: int) -> str:
            meta = self._bg_meta.get(task_id)
            if meta is None:
                raise LookupError(f"task {task_id} not found")
            text = meta.get("last_stdout", "")
            lines = text.splitlines()
            window = lines[offset: offset + limit] if limit > 0 else lines[offset:]
            return "\n".join(window)
        return _read

    def _raw_task_canceller(self, task_id: int) -> Callable[[], Awaitable[None]]:
        async def _cancel() -> None:
            proc = self._bg_procs.pop(task_id, None)
            handle = self._bg_tasks_handles.pop(task_id, None)
            self._bg_meta.pop(task_id, None)
            if proc is not None and proc.returncode is None:
                try:
                    proc.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                except ProcessLookupError:
                    pass
            if handle is not None and not handle.done():
                handle.cancel()
        return _cancel

    def _pm_task_reader(self, bg) -> Callable[[int, int], Awaitable[str]]:
        async def _read(offset: int, limit: int) -> str:
            return bg.last_stdout(offset=offset, limit=limit)
        return _read

    def _pm_task_canceller(self, bg) -> Callable[[], Awaitable[None]]:
        pm = self._pm
        async def _cancel() -> None:
            await pm.stop_background_task(bg.id)
        return _cancel
