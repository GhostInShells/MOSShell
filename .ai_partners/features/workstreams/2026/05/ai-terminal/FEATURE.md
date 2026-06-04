---
title: AI Terminal — Ghost 的操作系统双手（bash + 文件读写）
status: in-progress
priority: P0
created: 2026-05-29
updated: 2026-06-04
depends: []
milestone:
description: >-
  Ghost 最基础的操作系统工具链：bash.exec / file.read / file.write。
  Phase 1 纯逻辑 Terminal 模块 + subprocess 实现 + allow-all 模式，零外部依赖，立即可做。
  Phase 2 加审批链（TUI ApprovalState）、pexpect 会话持久性、Skills 库封装。
  三层架构保留：Terminal 纯逻辑 → Channel 适配 → 审批交互（延后）。
---

# AI Terminal

## Motivation

MOSS 的 Ghost 需要一个可以直接操作命令行的能力集合（对标 Claude Code 的 Bash/Read/Write/Glob）。
当前代码库没有任何 bash/shell channel——Ghost 没有操作系统的"手"。
这是最基础的能力缺口，优先级高于 audio-capture、feishu-channel 等感官扩展。

相对优势不在"bash 执行得更好"，而在执行模型不同：
- **Code as Prompt**：Python 函数签名即接口，不走 JSON Schema
- **CTML 并行调度**：bash + 文件操作 + 语音同时跑，不打断交流
- **context messages** 做 Ghost 的仪表盘（进程状态、文件变更一目了然）
- **审批链可插拔**：allow-all → whitelist → ask-human，按场景切换
- **通过 apps 体系对 Ghost 可插拔**

### 与 pexpect (interactive-shell-channel) 的关系

两个 feature 互补，不是竞争：

| | ai-terminal (本 feature) | interactive-shell-channel |
|---|---|---|
| 定位 | 工具型：一次性命令 + 文件操作 | 会话型：持久 PTY 终端 |
| 后端 | Phase 1: subprocess | pexpect |
| 典型场景 | `npm test`, `cat file.py`, `git status` | REPL, SSH, DB CLI, dev server |
| 状态 | 无状态 | 有状态（venv/cwd/env 跨命令存活）|

Phase 2 统一：Terminal 模块定义抽象协议，subprocess 和 pexpect 是两个实现。
interactive-shell-channel 成为 bash.exec 的会话持久性 backend 选项。

## Architecture

三层解耦，Phase 1 只做下面两层：

```
Phase 2+: 审批交互层 (TUI ApprovalState 或 GUI App Cell)
  │ 审批队列 + 人工确认 + 执行历史
  ▼
Terminal Channel (MOSS Channel 树)          ← Phase 1
  │ bash.exec / file.read / file.write
  │ Code as Prompt: Python 函数签名即接口
  │ CTML 调用 → Ghost 可见
  ▼
Terminal 模块 (contracts, 纯逻辑, 无 MOSS 耦合)  ← Phase 1 先做
  │ 命令执行引擎 (subprocess)
  │ 审批策略：Phase 1 仅 allow-all
  │ 后续扩展：whitelist / blacklist / ask-human
```

Phase 1 零外部依赖，不阻塞于任何其他 feature。

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

### 2. Phase 1 仅 allow-all，审批链延后

Phase 1 不引入审批逻辑。所有命令直接执行，通过环境变量 `MOSS_TERMINAL_MODE=allow-all` 控制。
这消除了对 tui-render-governance 和 session-communication-bus 的依赖，Terminal 模块可以立即开工。

Phase 2 引入审批策略链 (WhitelistPolicy → BlacklistPolicy → AskHumanPolicy)，
通过 ThreadSafeFuture 桥接同步审批和异步 Channel。
审批交互优先走 TUI ApprovalState（tui-render-governance 就绪后），备选独立 GUI App Cell。

### 3. Phase 1 串行执行，无 buffer，无并行

Terminal 模块 + Channel，命令串行执行。不引入流式 buffer，不引入 GUI。
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

### 6. ThreadSafeFuture 延后到 Phase 2 审批链

`ThreadSafeFuture` 已在 ROS2 控制器中生产使用（`Move` 和 `TrajectoryAction` 继承它）。
Phase 1 allow-all 模式不需要——命令直接同步返回。Phase 2 引入审批链后，
通过 ThreadSafeFuture 桥接同步审批和异步 Channel。后续可迁移到 Session Comm Bus 的 FutureManager 实现跨进程。

### 7. "可插拔"的语义

指对 Ghost 可插拔——Ghost 可以管理是否打开这个功能。打开时能打开 GUI。
不是指 Terminal 模块自身的解耦（那是三层架构保证的）。

### 8. 2024 年已有雏形可继承

GhostOS 项目中的 `terminal/abcd.py` (Terminal 抽象 + CommandResult) 和 `project/abcd.py` (Directory/File 抽象: read with line numbers, insert with range, continuous_write)。
设计思路直接继承，用 MOSS 当前架构重写实现。

### 9. 审批交互延后到 Phase 2，Phase 1 不做

Phase 1 通过环境变量 `MOSS_TERMINAL_MODE=allow-all` 跳过所有审批。
Phase 2 审批交互有两种可选实现，届时根据 tui-render-governance 进展选择：

| 方案 | 适用阶段 | 特点 |
|------|---------|------|
| **TUI ApprovalState** | Phase 2 推荐 | 依赖 tui-render-governance。审批队列 + completion 交互 + bottom_toolbar 通知。不引入新进程、新依赖 |
| **PyQt6 GUI App Cell** | Phase 2+ | 独立进程，跨进程审批控制台。依赖 Session Comm Bus 的 FutureManager |

### 10. pexpect 是 Phase 2 的会话持久性 backend

Phase 1 的 bash.exec 使用 subprocess。pexpect (来自 interactive-shell-channel feature)
在 Phase 2 作为可选 backend：Terminal 模块定义抽象协议，subprocess 和 pexpect 是两个实现。
pexpect 带来会话持久性（venv/cwd/env 跨命令存活）和交互式程序支持（REPL/SSH/DB CLI），
但其线程管理、PTY buffer、expect 模式匹配增加复杂度，Phase 1 不需要。

### 11. Skills 库是 Terminal 的上层封装

Terminal 模块 + Channel 就绪后，可以在其上构建初始 skills 库：
常用 shell 操作（git 工作流、包管理、进程管理）封装为预置的 command 组合和 context_messages 模板。
Terminal 提供原子操作，Skills 库提供惯用模式。这是 Phase 2/3 的方向，
但 Terminal 抽象的设计应预留此扩展空间。

## Implementation Notes

### Phase 1 实现顺序

1. **Terminal 模块** (`src/ghoshell_moss/contracts/terminal.py` 或同级非内核实现层)
   - `CommandResult`: exit_code, stdout, stderr
   - `Terminal`: 抽象接口 — `exec(cmd)`, `read_file(path)`, `write_file(path, content)`
   - `SubprocessTerminal`: subprocess 实现
   - 零 MOSS 依赖，纯 Python，可独立 import 和测试

2. **Terminal Channel** (`src/ghoshell_moss/channels/terminal/`)
   - 三个 command: `bash.exec`, `file.read`, `file.write`
   - Code as Prompt：Python 函数签名即接口
   - 参考 `app_store_channel` 的 StatefulChannel 模式
   - 注册为 App，通过 MCP 暴露给 Ghost

3. **验证路径**: moss-as-mcp → Claude Code 连接 → Ghost 调 bash.exec("echo hello") → 返回结果

### Phase 2 演进方向

- 审批链: allow-all → whitelist → blacklist → ask-human (TUI ApprovalState)
- pexpect backend: 会话持久性，来自 interactive-shell-channel
- 并行执行: 多个 bash.exec 在 CTML 不同 channel 中并行
- context messages 仪表盘: 后台命令状态
- Skills 库: 基于 Terminal 原子操作的惯用模式封装

### 设计参考

- `app_store_channel`: StatefulChannel + virtual_children + context_messages 模式
- `typer_channel`: CLI 反射为 Channel 的模式（moss-self-channel 可复用）
- `speech_channel`: Channel interface 风格，生命周期管理
- `notebook_channel`: 文件系统 CRUD 的 Channel 化参考
- GhostOS 项目 `terminal/abcd.py`: Terminal 抽象 + CommandResult 设计思路，用 MOSS 当前架构重写
