---
title: Project Manager
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-06-10
depends: [matrix-cell-governance, interactive-shell-channel]
milestone:
description: >-
  Project Manager — Ghost 的文件系统领地。以 cwd 为根作用域，统一 terminal、file_editor
  与 cell 管理。认知体系由两个动态脚本驱动：meta.sh（instruction）+ memory.sh（memory）。
  模型通过 terminal / file_editor 编辑脚本即修改自身认知架构。
---

# Project Manager

## Motivation

当前 Ghost 在 MOSS 运行时中没有"自己的领地"。

- 文件操作分散在 notebook_channel（笔记 CRUD）、terminal_channel（bash 读写）、shell_channel（pexpect 会话）三个 channel 中
- 没有统一的目录作用域概念 — 每次 bash.exec 都要指定路径
- 模型上下文没有可控的认知结构 — 系统推什么就看什么，无法自主构建注意力
- 行业方案（CLAUDE.md、MEMORY.md、rules、skills）是碎片化的子系统，各自有独立的格式、生命周期和加载逻辑

Project Manager 把它们统一为三个在文件系统上生长的概念：

1. **Scope** — cwd 即安全边界。边界内零审批。所有子 channel 继承。
2. **两个众所周知的脚本** — `.moss_ws/meta.sh` 生成 instruction，`.moss_ws/memory.sh` 生成 memory。脚本是动态的，可引用 `bin/` 下的其他脚本做组合。
3. **Cell 入网** — 模型通过 terminal 执行 `moss nodes run`，OS 管理进程生命周期，Matrix 提供入网发现，context message 反射运行状态。

与行业方案的本质差异: pin 的不是静态文件，是**可执行脚本**。脚本的输出是上下文。脚本本身可被模型通过 terminal / file_editor 修改。模型修改自身认知架构 = 编辑文件。

## Design Index

- GhostOS 前身实现: `GhostOS/libs/ghostos/ghostos/libraries/project/` — 2025-03 验证了 Directory/File/PyDevCtx/ProjectManager 抽象
- 本轮会话讨论: file_editor channel → scope-switching → pin-as-command → 两个脚本驱动认知体系

## Key Decisions

### 1. 入口 = cwd，约定 = .moss_ws

Project Manager 不接收 `root_dir` 参数。入口就是 `cwd`。Ghost 在哪里启动，哪里就是领地。

认知脚本路径:
```
.moss_ws/meta.sh          # instruction 生成脚本，每轮执行
.moss_ws/memory.sh        # memory 生成脚本，每轮执行
bin/                      # 可复用脚本仓库，被 meta.sh / memory.sh 引用
```

`.moss_ws` 是已有的 MOSS workspace 约定（`.env`、app 定义等），认知脚本自然归入其中。

### 2. Pin = 两个脚本，不是一套 API

不需要 `pin_instruction()` / `pin_memory()` 命令。Project Manager 每轮做的事情:

```bash
# 生成 instruction 上下文 — 插入 context 顶部
bash .moss_ws/meta.sh 2>/dev/null

# 生成 memory 上下文 — 插入 instruction 之后
bash .moss_ws/memory.sh 2>/dev/null
```

脚本可以动态组合:

```bash
#!/bin/bash
# meta.sh — 根据当前状态动态生成 instruction
cat CLAUDE.md
echo "--- 项目状态 ---"
git log --oneline -5 2>/dev/null
echo "--- 活跃特性 ---"
moss --ai features list 2>/dev/null
echo "--- 工作目录 ---"
ls -la
```

```bash
#!/bin/bash
# memory.sh — 动态生成记忆上下文
cat .moss_ws/MEMORY.md 2>/dev/null
echo "--- 近期决策 ---"
git log --oneline --since="14 days ago" 2>/dev/null | head -10
echo "--- 待办 ---"
cat .moss_ws/TODO.md 2>/dev/null
```

**模型修改这些脚本 = 修改自己的认知架构。** 用 terminal 的 `bash.write` 或 file_editor — 编辑一个文件就是重新配置自己的注意力。

### 3. 动态脚本 > 静态 md

CLAUDE.md 是数据源，不是上下文。把静态 md 直接怼进上下文，每轮消耗同样的 token 不管当前任务是什么。

脚本可以根据当前状态决定输出: 修 bug 时输出 bug 相关记忆和指引，做架构时输出设计文档索引和关键决策，空闲时输出极简摘要。

### 4. Cell 入网: OS 管进程，MOSS 做反射

模型通过 terminal 启动 cell:

```bash
moss nodes run watcher.py --watch src/ --on-change "moss event file-changed"
```

底层:
- `moss nodes run` → OS fork 进程
- 进程连上 Matrix（Zenoh pub-sub）
- Matrix 上出现新 channel node
- Project Manager 的 context 自动浮现该 channel 的 perspective

MOSS 不重新实现进程管理。OS 已经有了。MOSS 提供的是**语法糖 + 入网发现 + context 反射**。模型不感知 pid、IPC 协议 — 它只知道启动了一个东西，视角里有了它的输出。

### 5. 上下文结构: instruction → memory → conversations → perspectives

```
[instruction]     ← bash .moss_ws/meta.sh 的输出
[memory]          ← bash .moss_ws/memory.sh 的输出
[conversations]   ← 对话历史（平台管理）
[perspectives]    ← 所有子 channel 的 context_messages（实时视口）
```

Project Manager 的 context message 渲染这个结构。它自己几乎什么都不做 — 运行两个脚本、聚合子 channel 视角、拼成上下文。

### 6. 安全: OS 目录权限，不做审批

cwd 是作用域。所有子 channel（terminal、file_editor、cells）继承这个边界。

原理: 全双工系统不能靠时间中断（审批框堵住流）。安全由空间边界保证。边界是一次性设置的（Ghost 启动时确定 cwd），边界内的操作不需要再证明安全。

### 7. Terminal 是一切的基础，bash 是通用后退

任何不能用高效 channel 完成的事，退到 terminal。这保证 Project Manager 的能力完备性 — 没有"做不到"，只有"不够高效"。

以低效为起点，构建高效。高效 channel（file_editor 等）是对常见操作的优化，不是对能力集的扩展。

### 8. Channel 设计: scope-switching，先不做 class-based

当前 CTML 反射没有做同类型 channel 的实例化反射。多个同类型 channel 会暴露 N 份完全相同的 interface。这个问题不应阻塞文件编辑能力。

方案: 单个 FileEditor channel，`focus()` 切换作用域。未来 CTML 支持实例反射后，可从内部 dict 拆为独立 channel 实例，外部接口不变。

### 9. 分层: tools → channel → project

```
ghoshell_moss/tools/file_editor.py            # 纯逻辑，零 MOSS 依赖
ghoshell_moss/channels/file_editor_channel.py  # Channel 胶水层
ghoshell_moss/channels/project_manager.py      # Root channel，整合所有子能力
```

与现有 `SubprocessTerminal → terminal_channel` 的分层模式完全对齐。tools 层可独立测试，可被 GhostOS 前身复用。

## Implementation Plan

### Phase 1: FileEditor (tools 层) — 预计 1 天

文件: `src/ghoshell_moss/tools/file_editor.py`

纯 Python 类，零 MOSS 依赖。核心接口:

```python
class FileEditor:
    """单文件有状态编辑器 — 滑动窗口 + patch 累积 + 原子提交。"""

    def __init__(self, filepath: pathlib.Path, window_size: int = 50)
    def close(self) -> None

    # 导航
    def show(self, start: int, lines: int | None = None) -> FileWindow
    def slide(self, n: int) -> FileWindow
    def goto(self, line: int) -> FileWindow

    # 编辑 (累积 patch)
    def replace(self, old: str, new: str) -> int
    def replace_lines(self, start: int, end: int, text: str) -> None
    def insert(self, after_line: int, text: str) -> None
    def delete(self, start: int, end: int) -> None

    # 审查与提交
    def diff(self) -> str
    def patch_count(self) -> int
    def write(self) -> None        # 原子应用所有 patch，mtime 冲突时 raise
    def undo(self) -> bool

    @property
    def window(self) -> FileWindow
    @property
    def line_count(self) -> int

class FileWindow:
    path: pathlib.Path
    start_line: int
    lines: list[tuple[int, str]]
    total_lines: int
    has_pending: bool
```

关键实现:
- 打开时构建行偏移索引（一次扫描），之后窗口跳转 O(1)
- patch 用 `list[PatchOp]` 累积，`write()` 时从后往前应用到行数组，避免行号漂移
- `replace()` 在当前窗口内做字符串精确匹配
- `write()` 前检查 mtime，磁盘冲突时 raise
- 纯标准库，无新依赖

### Phase 2: FileEditor Channel — 预计 1 天

文件: `src/ghoshell_moss/channels/file_editor_channel.py`

```python
def new_file_editor_channel(
    *,
    name: str = "editor",
    window_size: int = 50,
) -> MutableChannel:
```

Scope 从 Project Manager 的 cwd 继承，不由 channel 自己管理。

Channel commands:
- `open(path)` — 打开文件
- `focus(path)` — 切换活跃文件
- `show(start, lines)` / `slide(n)` / `goto(line)` — 导航
- `replace(old, new)` / `replace_lines(start, end, text__)` / `insert(after_line, text__)` / `delete(start, end)` — 编辑
- `diff()` / `write()` / `undo()` — 审查与提交
- `close(path)` / `tabs()` — 文件管理

Context message: 活跃文件窗口 + tabs 栏 + pending diff 摘要。`focus()` 时自动更新。

### Phase 3: Project Manager Bootstrap — 预计 1 天

文件: `src/ghoshell_moss/channels/project_manager.py`

```python
def new_project_manager(
    *,
    name: str = "project",
) -> MutableChannel:
    # cwd 即根作用域
```

子 channel 树:
```
project (root, cwd)
  ├── terminal (terminal_channel, workspace_root=cwd)
  ├── editor (file_editor_channel)
  └── (未来) cells, skills...
```

Channel commands:
- `tree(prefix="", recursion=2)` — 目录树
- `describe(path, desc)` — 为文件/目录添加描述
- `work_on(subdir)` — 切换 working directory，更新 context

Context message 结构:
```
[instruction]     ← 运行 .moss_ws/meta.sh 的输出（如果脚本存在）
[memory]          ← 运行 .moss_ws/memory.sh 的输出（如果脚本存在）
[directory]       ← 目录树 + working directory
[perspectives]    ← 子 channel 的 context_messages（editor tabs, terminal state...）
```

Project Manager 本身极简 — 运行两个脚本、聚合子 channel 视角、拼成上下文。

## Deliverables

| # | 文件 | 状态 |
|---|------|------|
| 1 | `src/ghoshell_moss/tools/__init__.py` | 新建 |
| 2 | `src/ghoshell_moss/tools/file_editor.py` | 新建 |
| 3 | `tests/test_file_editor.py` | 新建 |
| 4 | `src/ghoshell_moss/channels/file_editor_channel.py` | 新建，注册 channeltype |
| 5 | `src/ghoshell_moss/channels/project_manager.py` | 新建，注册 channeltype |

## Implementation Notes

- `tools/file_editor.py` 参考 GhostOS `FileImpl`（`directory_impl.py:98-185`）但做本质升级: 从全量读写变为滑动窗口 + patch 累积
- Channel 层参考 `mcp_hub.py` 的 stateful channel 模式
- `focus()` 的 scope-switching 语义参考 GhostOS `DirectoryImpl.focus()`（`directory_impl.py:426-432`）
- 认知脚本路径 `.moss_ws/meta.sh` / `.moss_ws/memory.sh` — 利用已有 `.moss_ws` 约定
- Project Manager 不管理 pin 的完整生命周期 — 两个脚本的编辑是 terminal / file_editor 的事
- Cell 入网依赖 `matrix-cell-governance` feature（本 FEATURE 的 depends 之一）
- 不引入新依赖 — FileEditor 纯标准库，channel 层复用现有 `new_channel` builder
