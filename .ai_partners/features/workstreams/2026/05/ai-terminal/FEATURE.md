---
title: AI Terminal — 带审批链的智能模型命令行工具（GUI / TUI 可选）
status: draft
priority: P2
created: 2026-05-29
updated: 2026-06-03
depends:
  - session-communication-bus
  - tui-render-governance  # 如果第 3 层选择 TUI 审批而非 PyQt6 GUI
milestone:
description: >-
  为 Ghost 提供一套可插拔的 Terminal Channel，通过审批链让人类安心授权 AI 操作命令行。
  三层解耦：contracts 下的 Terminal 模块（纯逻辑）、Channel 体系（MOSS 命令反射）、
  审批交互层（可选 PyQt6 GUI App Cell 或 TUI ApprovalState，依赖 tui-render-governance）。
---

# AI Terminal

## Motivation

MOSS 的 Ghost 需要一个可以直接操作命令行的能力集合（对标 Claude Code 的 Bash/Read/Write/Glob）。
但直接给 AI 开 shell 权限是危险的——所以需要审批链 + GUI。

相对优势不在"bash 执行得更好"，而在执行模型不同：
- 审批前置（白名单 / 黑名单 / 人工确认 可切换）
- context messages 做"AI 的仪表盘"（后台任务状态一目了然）
- 运行时 GUI 让人类可见、可控
- CTML 并行调度下不打断交流（语音 + 命令同时跑）
- 通过 apps 体系对 Ghost 可插拔

## Architecture

第 3 层审批交互有两种可选实现，见 Key Decision 9。

```
审批交互层 (二选一):
  ┌─ TUI ApprovalState (MOSS TUI 内)        ← Phase 1 推荐, 依赖 tui-render-governance
  │   审批队列 + completion 交互 + bottom_toolbar 通知
  │
  └─ GUI App Cell (独立进程, PyQt6 主线程)   ← Phase 2, 跨进程审批控制台
      审批面板 / 执行历史 / 输出查看
      通过 Terminal 模块的线程安全 API 驱动
  ▼
Terminal 模块 (contracts, 纯逻辑, 无 MOSS/GUI/TUI 耦合)
  │ 命令执行引擎 (subprocess)
  │ 审批策略 (Phase 1: 硬编码 whitelist / ask-human)
  │ ApprovalModule — 审批队列管理 (submit / pending / decide / on_change)
  │ ThreadSafeFuture 桥接同步审批和异步 Channel
  │ 输出 buffer (Phase 2)
  ▼
Terminal Channel (MOSS Channel 树)
  │ bash.exec / file.read / file.write
  │ 每个 command 经审批链 → subprocess → 结果返回
  │ CTML 调用 → AI 可见
```

## Design Index

- Terminal 抽象接口: `src/ghoshell_moss/contracts/terminal.py` (待创建)
- Terminal 开箱实现: `src/ghoshell_moss/<non-core-layer>/terminal/` (待创建)
  - 非内核抽象的实现层，目录名待定，与 `core/` 同级
  - `core/` 管内核抽象，这一层管非内核的开箱实现
- Terminal Channel (可组装 channel): `src/ghoshell_moss/channels/terminal/` (待创建)
- GUI 实现 (PyQt6): `src/ghoshell_moss_contrib/terminal_gui/` (待创建)
  - 注意: PyQt6 相关实现在项目中是 alpha 版，beta 版需重新治理 GUI 体系
  - 原则: contracts 做抽象，channels 做可组装 channel，ghoshell_moss_contrib 做 GUI
- App Cell: `workspace/apps/terminal/` (待创建)
- ThreadSafeFuture: `src/ghoshell_moss/core/helpers/asyncio_utils.py:ThreadSafeFuture` (已可用)
- Session Comm Bus (Phase 2 依赖): `session-communication-bus` feature
- Session tmp storage (Phase 1 依赖 file.write 大文件): `session-communication-bus` feature 的 Cabinet 部分

## Key Decisions

### 1. 三层解耦: GUI → Terminal → Channel

Terminal 模块是纯 Python，不依赖 MOSS 也不依赖 GUI 框架。GUI 和 Channel 各自以自己的方式消费 Terminal。
这意味着 Terminal 模块可以独立 import 和测试，Channel 只是它的一个 MOSS 适配层。

### 2. Phase 1 审批硬编码，不做策略链抽象

Phase 1 只做两种模式：`allow-all`（环境变量切换）和 `ask-human`（默认）。
通过 ThreadSafeFuture 桥接：命令到达 → 创建 Future (pending) → GUI 展示 → 人类 approve/reject → set_result/set_exception → Channel 继续。
Phase 2 再抽象策略链 (WhitelistPolicy → BlacklistPolicy → AskHumanPolicy)。

### 3. Phase 1 串行执行，无 buffer，无并行

单 App Cell，GUI + Terminal 模块 + Channel。命令串行执行，不引入流式 buffer。
并行和流式是 Phase 2，到时 context messages 做仪表盘的优势才能充分体现。

### 4. Phase 1 三个命令

| 命令 | 说明 | 审批敏感度 |
|------|------|-----------|
| `bash.exec` | 执行任意命令，返回 stdout/stderr/exit_code | 高 |
| `file.read` | 读文件，带行号，大文件优化 | 低 |
| `file.write` | 写/覆盖文件，支持 update 化逐行修改 | 高 |

glob/grep 通过 bash.exec 间接实现。大文件先写临时区再分段读（依赖 Session tmp storage）。

### 5. Phase 2 演进方向

- Session Comm Bus 的 FutureManager 落地后，GUI 可独立于 Terminal 进程运行
- 支持多个 GUI 同时管理同一个 Terminal
- Context messages 仪表盘（后台命令状态一行一个，简洁不占 token）
- Skills 文件夹（AI 可写 markdown，context messages 按需反射目录结构）
- 流式输出 buffer

### 6. ThreadSafeFuture 直接用于 Phase 1

`ThreadSafeFuture` 已在 ROS2 控制器中生产使用（`Move` 和 `TrajectoryAction` 继承它）。
Phase 1 单进程内直接用，不需要等 Session Comm Bus 的 FutureManager。
Phase 2 迁移到 FutureManager 实现跨进程。

### 7. "可插拔"的语义

指对 Ghost 可插拔——Ghost 可以管理是否打开这个功能。打开时能打开 GUI。
不是指 Terminal 模块自身的解耦（那是三层架构保证的）。

### 8. 2024 年已有雏形可继承

GhostOS 项目中的 `terminal/abcd.py` (Terminal 抽象 + CommandResult) 和 `project/abcd.py` (Directory/File 抽象: read with line numbers, insert with range, continuous_write)。
设计思路直接继承，用 MOSS 当前架构重写实现。

### 9. 第 3 层（审批交互）优先 TUI，GUI 延后

审批交互有两种可选实现：

| 方案 | 适用阶段 | 特点 |
|------|---------|------|
| **TUI ApprovalState** | Phase 1 | 依赖 tui-render-governance。ApprovalModule 维护审批队列，TUI 内 completion 交互，bottom_toolbar 全局通知。不引入新进程、新依赖 |
| **PyQt6 GUI App Cell** | Phase 2+ | 独立进程，跨进程审批控制台。依赖 Session Comm Bus 的 FutureManager |

Phase 1 选 TUI 的原因：
- 全双工场景下 TUI 审批不抢占人类输入焦点，比独立 GUI 弹窗更符合 Ghost 控制哲学
- 不需要等 PyQt6 GUI 体系治理（项目当前 alpha），也不需要 Session Comm Bus 跨进程 Future
- ApprovalModule 纯逻辑 + TUI 审批交互可以独立跑通，再倒灌回 Terminal Channel

实现顺序：tui-render-governance（TUI 基础设施） → ApprovalState（TUI 审批界面） → Terminal 审批模块 + Channel。

## Implementation Notes

- Phase 1 目标：2-3 次对话内跑出原型。不追求设计完备，追求可运行。
- 非内核抽象实现层目录名待定（与 `core/` 同级），Terminal 的开箱实现放此处。
- PyQt6 GUI 放 `ghoshell_moss_contrib`，alpha 版先跑通，beta 版再治理。
- 审批队列用 `collections.deque` + maxsize，超出直接 reject，避免内存膨胀。
- 历史记录比待审批队列长，方便 GUI 回溯。
- `bash.exec` 不做命令清理/转义（审批链本身就是安全机制），Phase 1 用 `subprocess.Popen(shell=True)`。
