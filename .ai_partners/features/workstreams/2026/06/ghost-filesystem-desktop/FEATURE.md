---
title: Ghost Filesystem Desktop — Ghost 的文件系统工作桌面
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-06-28
renamed_from: Project Manager
depends:
  - matrix-cell-governance
  - interactive-shell-channel
milestone: 0.1.0
description: >-
  Desktop 是 Ghost 在文件系统上的工作桌面 — 17 个原语 (发现/读写/执行) + 两条元规则。
  _pin 通用化收缩 pin/unpin API。所有命令输出统一截断 + tmp 路径。
  Desktop 是 Project 级公共 API，由 ProcessManager (入 Matrix) 支撑。
  frontmatter 作为信息提取原语，约定不写在 Desktop 实现里 — 使用方 (Ghost/Mode) 自己定义。
status_note: >-
  2026-06-28 人类架构师 + deepseek-v4-pro 完整设计收敛:
  (1) 命名从 ProjectManager → Desktop, 避开 ProcessManager 歧义.
  (2) _pin 通用标记 — 所有信息型命令支持 _pin=True, 消除独立 pin/unpin API.
  (3) 统一输出截断 — 所有命令超阈值写 tmp, read(tmp_path) 不截断.
  (4) 两条元规则: read-before-write + 输出截断.
  (5) frontmatter 原语 — 不硬编码 SKILL.md/__doc__ 约定.
  (6) Desktop 放 Project 上 (project.desktop()), 不进 Matrix.
  (7) DESKTOP.md 可选覆盖默认 instruction.
  (8) CTML pin 洞察记录, 当前不做 — pin 本质是 schedule(ctml, refresh).
---

# Desktop — Ghost 的文件系统工作桌面

## Motivation

当前 Ghost 在 MOSS 运行时中没有"自己的领地"。

- 文件操作分散在 notebook_channel、terminal_channel、shell_channel 三个 channel 中
- 没有统一的目录作用域概念 — 每次 bash.exec 都要指定路径
- 模型上下文没有可控的认知结构 — 系统推什么就看什么，无法自主构建注意力
- 行业方案 (CLAUDE.md, MEMORY.md, rules, skills) 是碎片化的子系统

Desktop 把它们统一为 Ghost 在文件系统上的**工作桌面** — 发现、读写、执行三类原语 +
两条元规则。模型通过组合这些原语自己构建对项目的认知。

与行业方案的本质差异：没有硬编码"什么文件有认知价值"。`frontmatter()` 是提取原语，
不是约定。约定由使用方 (Ghost 的 system prompt / mode 的 HOST.md) 自己下。

## Design Index

- 原设计: `src/ghoshell_moss/contracts/project_manager.py` — 旧 ProjectManager ABC
- ProcessManager: `src/ghoshell_moss/contracts/process_manager.py` + `core/process_manager/_impl.py`
- Channel 构建: `moss codex blueprint channel_builder`
- CTML 模型视角: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- 相关讨论: `matrix-cell-governance` FEATURE.md — Matrix 重构中 ProcessManager 入 Matrix

## Key Decisions

### 1. 定位: Project 级公共 API，不进 Matrix

Desktop 是 `project.desktop(root=...)` 构造的。不放 Matrix 上:

- Matrix = 通讯基础设施 (cells, session, network, IoC)
- Desktop = 文件系统工作表面
- Desktop 依赖 ProcessManager (集成在 Matrix) 做命令执行，但依赖不等于归属

```
Matrix
  ├── session, network, cells, IoC
  ├── spawn() — ProcessManager 驱动的子进程管理
  └── project → project.desktop(root=path)
```

### 2. 构造: 以任意目录为 root

Desktop 不关心目录的"身份" — 可以是 project root、ghost home、mode home:

```python
class Project(ABC):
    def desktop(self, root: Path | None = None) -> Desktop:
        """创建 Ghost 在文件系统上的工作桌面。root 默认 project.root。"""
```

### 3. 命令集 — 17 个原语三层

```
发现层 (Glob / Grep / Tree)
────────────────────────────────────
glob(pattern: str, *, _pin: bool = False) -> list[str]
  """匹配文件路径。返回相对于 root 的路径列表。"""

grep(pattern: str, *, path: str = ".", _pin: bool = False) -> list[Match]
  """搜索文件内容。返回 {file, line, text}。支持正则。"""

tree(depth: int = 2, *, path: str = ".", _pin: bool = False) -> DirectoryTree
  """目录结构。子项标注类型 (file/dir/symlink)。"""

cd(path: str) -> str
  """切换工作目录。返回绝对路径。限制在 root 子树内。"""

pwd() -> str
  """当前工作目录的绝对路径。"""

读取层 (Read / Head / Frontmatter)
────────────────────────────────────
read(path: str, *, offset: int = 0, limit: int = 200, _pin: bool = False) -> str
  """读文件内容。超 threshold 自动截断 + tmp。返回带行号文本或 {content, truncated, tmp_path}。"""

head(path: str, *, lines: int = 20) -> str
  """文件前 N 行 — 快速扫描，不触发截断逻辑。"""

frontmatter(path: str, *keys: str) -> dict | None
  """提取 markdown 文件的 YAML frontmatter。keys 过滤字段。无 frontmatter 返回 None。"""

写入层 (Write / Edit)
────────────────────────────────────
write(path: str, content: str) -> None
  """创建或覆盖文件。必须本 session 内先 read 过目标文件。"""

edit(path: str, old: str, new: str) -> int
  """替换文件中的字符串。必须本 session 内先 read 过。old 必须精确匹配一次。返回替换行号。"""

执行层 (Exec / ExecBg / Tasks)
────────────────────────────────────
exec(command: str, *, timeout: float = 60.0, _pin: bool = False) -> ExecResult
  """执行 shell 命令。返回 {stdout, stderr, exit_code, killed}。超时 kill 进程组。"""

exec_bg(command: str, *, loop: int = 1) -> int
  """后台执行命令。loop=0 无限循环。返回 task_id。"""

tasks(*, _pin: bool = False) -> list[TaskInfo]
  """列出活跃后台任务。"""

read_task(task_id: int, *, offset: int = 0, limit: int = 100) -> str
  """读取后台任务的输出窗口。"""

cancel(task_id: int) -> None
  """取消后台任务。"""

Pin 管理
────────────────────────────────────
pinned() -> list[PinInfo]
  """列出所有活跃 pin。返回 {id, command_name, args, last_preview}。"""

unpin(pin_id: str) -> None
  """移除一个 pin。"""
```

### 4. _pin 通用化收缩

不是 `pin(command)`，而是每个信息型命令带 `_pin: bool = False`:

```python
desktop:exec("git status", _pin=True)
desktop:tree(depth=2, _pin=True)
desktop:glob("*.py", _pin=True)
```

语义: 执行命令并注册为周期性执行。每帧 refresh 时自动重跑，输出注入 context。
`_pin=False` 是一次性调用。

_pin 只对信息型命令有效: `tree`, `glob`, `grep`, `read`, `head`, `frontmatter`, `exec`, `exec_bg`, `tasks`。
`cd`, `write`, `edit`, `cancel`, `unpin` 无效 — 传了忽略。

_pin 下划线前缀 = CTML 元参数，不是业务参数。模型看到 interface 自然区分。

### 5. 统一输出截断

不是 `read` 独享 — 所有命令输出共享一条规则:

```
output > threshold (200 行 或 32KB)?
  → 完整内容写入 tmp/desktop/{command}/{hash}
  → 返回值 = 截断预览 + 完整路径标记
  → 模型用 read(tmp_path) 获取完整内容

read(tmp_path)
  → tmp 路径不截断，直接返回完整内容
```

Desktop 维护 `_tmp_roots` 集合 (`tmp/desktop/`)。任何路径以 `_tmp_roots` 开头不触发截断。

### 6. read-before-write 元规则

`write` 和 `edit` 检查调用方是否在本 session 内 `read` 过目标文件。
Desktop 维护 `_read_set: set[Path]`。未 read 抛 `ObserveError`。

这是 Claude Code 最核心的 guard — 防止幻觉写入。

### 7. frontmatter 原语，不硬编码约定

Desktop **不**定义 `__doc__`、`SKILL.md` 等命名约定。`frontmatter()` 是提取原语 —
任何一个 markdown 的 YAML 头都可以被提取。约定由使用方通过 instruction 下:

> 当你进入一个项目:
> 1. glob('**/CLAUDE.md') 找到认知入口
> 2. glob('.skills/**/SKILL.md') 找到所有 skill
> 3. frontmatter(path, 'description') 提取元信息
> 4. 用这些信息构建你对项目的认知

Desktop 的默认 instruction 建议初始探测方向，不强制。

### 8. DESKTOP.md — 可选覆盖默认 instruction

`root/DESKTOP.md` 如果存在，其内容覆盖 Desktop channel 的默认 instruction。
这让项目/gohst/mode 可以定制 Ghost 在特定 Desktop 上看到的第一帧。

如果不存在，Desktop 使用内置的默认 instruction (含命令列表 + 规则摘要 + root + pwd)。

### 9. CTML Pin — 记录，当前不做

pin 本质不是"钉一块 bash 输出屏幕" — 是"定义一个周期性执行的 CTML 子程序":

```python
# 当前 (bash pin)
desktop:exec("git status", _pin=True)

# 未来 (CTML pin)
desktop:pin(ctml='<desktop:tree depth="1"/><desktop:tasks _pin="true"/>', refresh="on_prompt")
```

CTML pin 的输出是 Desktop 自己的命令结果 — `tree` 返回 `DirectoryTree` 结构，
可以被 context 按规则渲染。这和 bash stdout 不在同一语义空间。

当前不实现。模型用熟原语后，自然浮现"哪些组合值得 pin"。届时 pin 从 `_pin` 参数
升级为 `schedule(ctml, refresh)` — 一个用户态 cron。

### 10. context_messages: 多元信息有机组合

第一版不做硬性的自动注入。context_messages 由 Ghost / mode / DESKTOP.md 指令驱动，
是 `tree` + `pinned` + `tasks` 等结果的有机组合。后续迭代定义默认组合逻辑。

### 11. ProcessManager 作为底层

Desktop 的 `exec` / `exec_bg` 底层走 ProcessManager (集成在 Matrix)。

- `exec` → ProcessManager.execute_task (阻塞 + 输出捕获)
- `exec_bg` → ProcessManager.execute_task(background_run=...) (调度 + buffer 复用)
- `cancel` → ProcessManager.stop_background_task
- `tasks` → ProcessManager.background_tasks

Desktop 不自己管理子进程生命周期 — ProcessManager 提供 start_new_session +
pipe fencing + polling 三件套。

### 12. 安全模型: 空间边界，零审批

`cd` 限制在 root 子树内。边界内所有操作 (`exec`, `write`, `edit`) 零审批。
Claude Code 的"边界内不确认"原则:

> 全双工系统不能靠时间中断 (审批框堵住流)。安全由空间边界保证。
> 边界是一次性设置的，边界内的操作不需要再证明安全。

### 13. Channel 架构

```
desktop (MutableChannel, root)
  │  commands: cd / pwd / tree / glob / grep / read / head / frontmatter
  │            write / edit / exec / exec_bg / tasks / read_task / cancel
  │            pinned / unpin
  │  instruction: DESKTOP.md 或默认模板
  │
  ├── terminal (terminal_channel, scope=cwd)  ← cwd 随 Desktop 同步
  ├── editor (file_editor_channel)            ← read_set 共享
  └── tasks                                   ← exec_bg 产生的子任务视图
```

## 与关联基建的交叉

| 基建 | 关系 | 状态 |
|------|------|------|
| `ProcessManager` | Desktop 的执行底层 | 已有 impl + 42 tests。需集成入 Matrix (matrix-cell-governance 推进中) |
| `Matrix.spawn()` | Desktop 不直接用 — 走 ProcessManager | 待 ProcessManager 入 Matrix 后统一 |
| `CellRegistry` | `cells` 子 channel 未来集成 | Desktop 第一版不含 cells 管理 |
| `FileEditor` | Desktop 的子 channel | 待实现 (project-manager FEATURE Phase 1-2) |
| `Terminal` | Desktop 的子 channel | 已有 interactive-shell-channel |
| `CTMLShell` | CTML pin 的运行时 (未来) | 记录，当前不做 |

## Implemented Notes

- 旧 `contracts/project_manager.py` 废弃重写 — 改为 `contracts/desktop.py`
- `tools/file_editor.py` 先不写 — Desktop 的 `read`/`write`/`edit` 覆盖了核心读写需求
- 第一版不集成 file_editor_channel 为子 channel — `read` + `write` + `edit` 已够
- `exec` / `exec_bg` 底层适配 ProcessManager，等 ProcessManager 入 Matrix 后对接
