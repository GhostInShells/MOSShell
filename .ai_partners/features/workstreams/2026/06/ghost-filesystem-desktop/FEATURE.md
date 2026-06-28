---
title: Ghost Filesystem Desktop — Ghost 的文件系统工作桌面
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-06-29
renamed_from: Project Manager
depends:
  - matrix-cell-governance
  - interactive-shell-channel
  - momento-mori
milestone: 0.1.0
description: >-
  Desktop 是 Ghost 在 4 剪影拓扑里的空间脏器/未来剪影 — 配合 Memento (过去)、
  Matrix runtime (当下)、Git worktree (结构版本) 共同构成 Ghost 反身性基建。
  12+1 原语三层 (发现/读写/执行) + 两条元规则 (read-before-write + 输出截断)。
  Desktop 是 OS 层抽象, 不耦合特定 ghost; 通过 ReadHistory protocol 与 Memento
  对接, 通过 ReflectionHint 在高影响路径写入时建议 commit 锚点。
status_note: >-
  2026-06-29 Claude Opus 4.7 Stage 1 完成: 12+1 原语 L0 独立闭环跑通.
  contracts/desktop.py (ABC + ReadHistory protocol + ReflectionHint),
  core/desktop/{desktop.py, models.py} (DefaultDesktop + InProcessReadHistory),
  tests/ghoshell_moss/core/desktop/test_desktop.py 53 个单测全绿.
  旧 contracts/project_manager.py 已删除 (无外部引用, 设计被 Desktop 完全覆盖).
  Stage 2/3 待启动. 详见正文 Stage 1 完成记录.
  --
  2026-06-28 Claude Opus 4.7 L2 收敛 (基于早前 deepseek-v4-pro 设计稿):
  (1) 加入 4 剪影拓扑视角 — Desktop 是未来剪影, 与 Memento/Matrix/Git 完备性对齐.
  (2) 17 原语 → 12+1 — head 删除 (bypass read-before-write), exec_bg 收成 exec(_bg=True),
      read_task/cancel 收进 tasks() 返回结构, frontmatter 列为可选.
  (3) _pin 语义校正 — pin 必须进 moss_dynamic, 落 cache breakpoint 之后, 加 LRU 预算.
  (4) _read_set → ReadHistory protocol 注入 — Phase 4 由 Memento branch state 后置.
  (5) tmp 目录改构造参数 — 为 Phase 4 接 Memento storage 留窗口.
  (6) 反思机制升级 — 高影响路径写入返回 ReflectionHint(diff, recommend_commit).
  (7) 对称 fork 概念 — Phase 6 反身性要求 memento branch + desktop worktree 同步 fork.
  (8) Ghost/OS 分层纪律 — Desktop 完全在 OS 层, 不为特定 ghost 妥协.
  (9) 6-Phase 切分 — Phase 1 (L0 独立闭环) 可在 matrix 重构同期穿插推进.
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

- **L2 收敛 (2026-06-28)**: `.design/2026-06-28_desktop_in_4d_cross_section.md`
  — 4 剪影拓扑、完备性判据、接口契约、Phase 切分、acceptance 边界
- **L2 涌现轨迹**: `.discuss/2026-06-28_desktop_l2_emergence.md` — 物理 vs 化学
  方法论、L2 协作的真实坐标、OS 命题的浮现过程
- 配对基建: `momento-mori` FEATURE.md — Memento 是过去剪影, 与 Desktop 共构反身性基建
- 原设计: `src/ghoshell_moss/contracts/project_manager.py` — 旧 ProjectManager ABC (待废弃)
- ProcessManager: `src/ghoshell_moss/contracts/process_manager.py` + `core/process_manager/_impl.py`
- Channel 构建: `moss codex blueprint channel_builder`
- CTML 模型视角: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- 相关讨论: `matrix-cell-governance` FEATURE.md — Matrix 重构中 ProcessManager 入 Matrix

## 迭代方法论 — 模型自迭代的三阶递进

Desktop 的开发本身遵循 MOSS 的 channel 构建梯度 (channels/CLAUDE.md §4):
**L0 纯模块 → L1 module_eval (MCP 试用) → L2 正式 channel (moss-as-mcp 自迭代)**。
每个阶段使用不同的自迭代机制, 模型在自己的迭代窗口里完成 interface 设计、实现、
loop 调整、验证。

### Stage 1 — L2 对齐后的模型重建 (当前阶段)

**范围严格控制**: 只动两个路径, 不扩展任何外部体系。

- `src/ghoshell_moss/core/blueprint/desktop.py` — Desktop ABC + ReadHistory protocol
- `src/ghoshell_moss/core/desktop/` — 实现 (desktop.py, models.py, 内部 helpers)
- `tests/ghoshell_moss/core/desktop/` — 单测

**不动**: project.py / Matrix / channels/ / atom / contracts/project_manager.py
(旧 ABC 暂留, 由人类工程师后续 IDE 整体回迁)。

**自迭代闭环**:

```
1. 读 .design/2026-06-28_desktop_in_4d_cross_section.md 锚定 L2 契约
2. 设计 ABC (interface) → 写 blueprint/desktop.py
3. 实现 (concrete) → 写 core/desktop/desktop.py + models.py
4. 写单测 → 跑 → 失败/通过
5. 单测发现 interface 形状不顺手 → 回 step 2 修 ABC → 重走
6. 走完 acceptance 边界 → Stage 1 结束
```

**Acceptance 边界** (来自 .design §10):
- 12 原语 (+frontmatter 可选) 的契约用 ABC 表达
- ReadHistory protocol + 进程内缺省实现
- read-before-write 守卫在 write/edit 上正确触发
- 统一输出截断 + tmp_path 路径不重复截断
- 反思路径白名单触发 ReflectionHint
- Pin 注册、查询、移除、LRU 淘汰
- ProcessManager 注入 vs 裸 subprocess 两条路径行为等价 (cwd 一致)
- 全部覆盖单测, read-before-write / 截断 / pin LRU / reflection 边界各有专门单测

**模型纪律** (L2 漂移防御):
- interface 改一次, 实现和单测同步改一次 — 不允许 interface 改了实现没跟
- 任何"再加一层 abstraction / 再加一个参数 / 再加一个 hint"的提议都视为漂移信号,
  停下来问"能不能反过来收紧而不是扩展"
- 实现里不出现对 Matrix / Memento / Session 的任何 import
- ReflectionHint / ReadHistory 这类对外接口不预设具体下游 (反推: 任何对它们的
  实现应该可以在不动 desktop 源码的前提下提供)

### Stage 2 — Eval Channel 试用 (Stage 1 acceptance 后)

把 `core/desktop/desktop.py` 包装成 `module_eval` channel 形态, 让开发者模型
通过 MCP 直接 exec 它, 在使用中暴露形状问题。

**机制**: 类比 .moss_ws/apps/browsers/playwright/main.py 的形态——
desktop 源码作为 instruction (Code as Prompt), 模型在持久化 namespace 里
exec `desktop.tree(...)` / `desktop.read(...)` / `desktop.write(...)`, 立刻看到
返回值, 别扭就改 desktop 源码再 exec。

**产出**: 不是新功能, 是 Stage 1 接口的形状校验。Stage 2 结束时, desktop
的 interface 应该没有任何"用起来别扭但能凑合"的点——所有这些点要么修了,
要么明确记录到本 FEATURE.md 的"已知不便"段, 等 Stage 3 解决。

**不进 channels/**: Stage 2 的 module_eval 包装是一次性试用工具, 不沉淀。

### Stage 3 — 正式 Channel + moss-as-mcp 自迭代

实现 `src/ghoshell_moss/channels/desktop_channel.py` 作为正式 channeltype,
进入 channels/ CLAUDE.md 的 status=alpha → beta → active 流转。

**机制**: 启动 moss-as-mcp, 模型一边定义 app/cell 使用 desktop channel,
一边体验它, 一边修改 channel 实现。这是 MOSS 自身宣称的"模型 native 开发
模式"在 desktop 本身上的应用。

**自迭代闭环**:

```
1. 写 desktop_channel.py (基于 Stage 2 校准过的 core/desktop)
2. 启动 moss-as-mcp, 模型连入
3. 模型定义一个 small app/cell 使用 desktop channel (例如: 项目认知探针 / 
   日记生成器 / 文件结构总结)
4. 跑, 体验
5. 发现 channel 层 (pin 注入位置、moss_dynamic 刷新时机、ReflectionHint
   路由、ReadHistory 接 Memento) 不对劲 → 改 channel 实现 → 重跑
6. 走完 channels/CLAUDE.md §7 的测试模式 → 进入 status=beta
```

**acceptance**:
- pin 输出正确落在 moss_dynamic 的 staging 段 (cache breakpoint 后)
- ReflectionHint 正确路由为 memento commit 建议
- ReadHistory 切到 Memento branch state 后端 (需 Memento Phase 4)
- 完成 Memento ↔ Desktop 的接口契约联调 (见 .design §6 对称 fork 不在此阶段)

**与正式集成的关系**: Stage 3 完成后, `project.desktop(root=...)` 工厂 +
default mode 的 desktop channel 注入可以由人类工程师拍板, 模型不主动推进
这一步——OS / Ghost 分层纪律要求 Desktop 进入哪些 mode 是 Ghost 层决策。

### 阶段间的产物归档纪律

- Stage 1 结束: 更新本 FEATURE.md status_note 标记 "Stage 1 complete",
  列出 acceptance 通过情况和发现的 L2 偏差 (若有)
- Stage 2 结束: 在本 FEATURE.md 增加 "Stage 2 校准记录" 段, 列出修了什么、
  保留了什么、为什么
- Stage 3 结束: channel 进入 channels/CLAUDE.md 索引, status=alpha,
  本 FEATURE.md set-status completed, .design 文件按需补充 channel-layer
  契约更新

### 模型协作的 ground rule

- 任何阶段, 模型每完成一个 acceptance 子项, 立即更新 FEATURE.md
  (不批量, 不延后) — 滚动可恢复
- 阶段中若发现 L2 偏差 (.design 文件结论需要修正), **先停下来更新 .design,
  再继续实现** — 不允许实现绕过设计
- 模型自感漂移时 (开始加机制而非收紧), 主动停下来 ground 回 .design 文件,
  不寻求人类工程师确认

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

## Stage 1 完成记录 (2026-06-29, Claude Opus 4.7)

### 落地清单

| 文件 | 内容 |
|------|------|
| `src/ghoshell_moss/contracts/desktop.py` | Desktop ABC + ReadHistory Protocol + ReflectionHint dataclass + 全部公开数据模型 (FileContent / ExecResult / Match / Task / PinInfo / DirectoryTree) + 异常 (ReadBeforeWriteError / PathOutsideRootError / PinBudgetExceeded) |
| `src/ghoshell_moss/core/desktop/desktop.py` | DefaultDesktop 实现 — 12+1 原语 + 两条元规则 + LRU pin 预算 + 反思路径白名单 + ProcessManager 可选注入 + 裸 asyncio 兜底 |
| `src/ghoshell_moss/core/desktop/models.py` | PinRecord (实现内部) + InProcessReadHistory (缺省协议实现) |
| `src/ghoshell_moss/core/desktop/__init__.py` | 重导出契约 + 实现 |
| `tests/ghoshell_moss/core/desktop/test_desktop.py` | 53 个 acceptance 单测, 全绿 |

旧 `src/ghoshell_moss/contracts/project_manager.py` 已删除 — 与人类工程师对齐后整体废弃, Desktop 完全覆盖, 无外部 import 引用 (grep 验证).

### Acceptance 边界覆盖情况

- ✅ 12 原语 (+frontmatter 可选) 的契约用 ABC 表达 — 见 `contracts/desktop.py`
- ✅ ReadHistory protocol + 进程内缺省实现 — InProcessReadHistory, 单测注入第三方实现验证可替换
- ✅ read-before-write 守卫在 write/edit 上正确触发 — `test_write_existing_requires_read`, `test_edit_requires_read` 等
- ✅ 统一输出截断 + tmp_path 路径不重复截断 — `test_read_truncation_writes_tmp`, `test_tmp_path_read_does_not_truncate`
- ✅ 反思路径白名单触发 ReflectionHint — 覆盖顶层文件 / 目录前缀 / 自定义白名单 / 命中 vs 不命中
- ✅ Pin 注册 / 查询 / 移除 / LRU 淘汰 — `test_pin_lru_eviction`, `test_pin_lru_refresh_on_repin`, budget warning 标记
- ✅ ProcessManager 注入 vs 裸 subprocess 两条路径行为等价 — `test_exec_via_process_manager_cwd` 对照两路 cwd/exit_code
- ✅ 12 原语全部覆盖单测; read-before-write / 截断 / pin LRU / reflection 边界各有专门单测

### L2 偏差记录 (实现过程中相对 .design 的微调)

1. **`write` 返回类型从 `None` 改为 `ReflectionHint | None`** — .design §5 只描述了 hint 概念, 没有明确返回路径. 选择走返回值而非回调, 符合 §3.2 "Desktop 通过返回值发信号, 上层路由" 的纪律. `edit` 同理返回 `tuple[int, ReflectionHint | None]`.

2. **`Task` 把 `read()` / `cancel()` 做成方法** — .design §7 说 "tasks 返回结构持 `read()` / `cancel()` 方法". 实现侧用 dataclass + bound async callable (`_read`, `_cancel`) 让顶层不再需要 `read_task` / `cancel` 原语, 收口符合 12+1 数. 顶层 ABC 上只剩 `tasks()`.

3. **新建文件不触发 read-before-write** — .design 没明说. 选择: 路径不存在的 write 直接放行 (创建本身就是初始 epistemic 锚点), 路径存在的 write 强制 ReadHistory 命中. 这符合 Claude Code 的行为, 也避免 "为了写新文件先要 read 一个不存在的文件" 的死锁.

4. **`tmp_path` 读取不登记 ReadHistory** — tmp 文件是 Desktop 自己的截断产物, 不是 Ghost 主动观察的代码/配置. 登记 read history 没有反身性语义, 反而污染 Memento branch state.

5. **`reflection_paths` 改为 `dict[str, severity]` 构造参数** — .design §5 给了 5 个默认项 + 单 severity 概念, 实现把它统一成 `{pattern: severity}` 表, 默认值导出为 `DEFAULT_REFLECTION_PATHS`, 让上层可以覆盖. 支持目录前缀 (`.moss/`)、精确名 (`CLAUDE.md`)、glob (`*.toml`).

6. **`Task` 异常用 `KeyError`, 不是 `LookupError`** — `unpin` 不存在的 id 抛 KeyError 符合 dict 语义; `Task.read` / `Task.cancel` 等回调未绑定时抛 `RuntimeError`. 异常分层尽量贴近 Python 内建.

### 已知未决 / 待后续阶段

- **`frontmatter` 去留**: 当前保留. L1 (Stage 2 module_eval 试用) 后定. 倾向于删 — `read(limit=20)` + 模型自解析 YAML 可替代, 没必要做内置依赖 `python-frontmatter` 库.
- **shutdown 幂等性的强保证**: 当前实现已是幂等 (set 清空), 但没有专门单测. Stage 2 试用时如果发现 shutdown 重入有问题再加 race condition 单测.
- **跨 worktree 的 Pin fork 行为**: Phase 6 处理.
- **DESKTOP.md 写守卫两步确认**: Phase 2 决策, 当前 reflection 只给 hint 不阻止写入.

### 模型纪律自评

- ✅ interface 改一次, 实现和单测同步改一次 — 期间多次往返 (e.g. `Task` 从独立 `read_task`/`cancel` 收成方法, 三个文件一起改)
- ✅ 实现里不出现对 Matrix / Memento / Session 的任何 import — `grep` 验证
- ✅ ReflectionHint / ReadHistory 这类对外接口不预设具体下游 — `_RecordingHistory` 单测证明可外部实现 ReadHistory 而不动 Desktop 源码
- ✅ 没漂移加机制 — 反而把 17 原语缩到 12+1, 把数据模型全部上推到 contracts 让 core 只承担实现

### 下一步

进入 **Stage 2 (eval channel 试用)** 之前等人类工程师评审 Stage 1 的接口形状.
评审通过后:
- 包一个 `module_eval` 形态让模型在 MCP 里 exec `desktop.tree(...)` / `desktop.read(...)` 等
- 暴露 interface "用起来别扭" 的真实痛点
- 痛点要么修, 要么记录到本文件 "已知不便" 段
