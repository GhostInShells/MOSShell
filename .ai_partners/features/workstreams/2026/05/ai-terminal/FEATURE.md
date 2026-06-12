---
title: AI Terminal — Ghost 的操作系统双手（bash + 文件读写）
status: completed
status_note: >-
  2026-06-12 Phase 1 prototype completed — SubprocessTerminal + terminal_channel + 11 tests + MCP verified.
  Phase 2 (Terminal ABC extraction, multi-channel split, approval chain, context messages) triggers when
  project-manager and matrix-cell-governance are ready. The abstraction is clean: three method signatures
  are the implicit protocol boundary.
priority: P0
created: 2026-05-29
updated: 2026-06-12
depends: []
milestone: prototype
description: >-
  Ghost 最基础的操作系统工具链：bash.exec/run + file.read/write。
  Phase 1 原型已完成：core/terminal 实现 + channel 适配 + 11 单测 + MCP 验证。
  下阶段滚动更新每个原子功能。
---

# AI Terminal

## Motivation

MOSS 的 Ghost 没有操作系统的"手"——不能执行命令，不能读写文件。
对标 Claude Code 的 Bash/Read/Write，这是最基础的能力缺口。

相对优势不在"bash 执行得更好"，而在 MOSS 的执行模型：
- **Code as Prompt**：Python 函数签名即接口，不走 JSON Schema
- **CTML 并行调度**：bash + 文件操作 + 语音同时跑
- **审批链可插拔**：allow-all → whitelist → ask-human（Phase 2+）

### 与 pexpect (interactive-shell-channel) 的关系

| | ai-terminal | interactive-shell-channel |
|---|---|---|
| 定位 | 工具型：一次性命令 + 文件操作 | 会话型：持久 PTY 终端 |
| 后端 | Phase 1: subprocess | pexpect |
| 典型场景 | `npm test`, `cat file.py`, `git status` | REPL, SSH, DB CLI |

Phase 2 不统一为一个协议。ai-terminal 和 interactive-shell-channel 是两个独立的
交互范式——一次性命令执行 vs 持久交互会话。它们在 core/terminal/ 下各自持有
自己的抽象协议（Terminal ABC 和 InteractiveSession ABC），不互相实现。

## Architecture

### 分层与路径（实际实现）

参考 `core/concepts/topic.py` → `core/topic/` 的分层模式：

```
core/terminal/                  ← 零 MOSS 依赖的实现
  │ subprocess_terminal.py     CommandResult + SubprocessTerminal
  ▼
channels/terminal_channel.py    ← MOSS Channel 适配层 (L1 Builder)
  │ bash:exec  (blocking)      阻塞执行，需要结果时用
  │ bash:run   (@nonblocking)  不阻塞，后台任务
  │ bash:read  (always_observe) 读文件，带行号
  │ bash:write                 写文件，text__ 参数
```

Phase 1 不抽协议，但 **有协议思维**：`SubprocessTerminal` 的方法签名干净——
`exec()`, `read_file()`, `write_file()` 三个方法就是未来的协议边界。
Phase 2 提 ABC 时，接口不加不减，只加 `class Terminal(ABC)` 和 `@abstractmethod`。

contracts 层未创建——等 Phase 2 有第二个 backend（pexpect）时再提协议。

### Channel 命令设计

四个命令，`blocking` 和 `@nonblocking` 是 Builder API 的原生调度标记：

| 命令 | 调度 | always_observe | 说明 |
|------|------|----------------|------|
| `bash:exec` | blocking | True | 阻塞，需要结果时用 |
| `bash:run` | @nonblocking | False | 不阻塞，fire-and-forget |
| `bash:read` | blocking | True | 读文件，带行号 |
| `bash:write` | blocking | False | 写文件，text__ 传内容 |

Ghost 自行管理拓扑顺序：依赖前序结果的用 `exec`，可并行的用 `run`。
Phase 2 可拆为多 channel 实现真正的跨 channel 并行。

### Phase 1 边界

- allow-all 模式，无审批
- subprocess.run(shell=True)
- 零外部依赖
- cwd 默认为进程 cwd（workspace root）

## Prototype Verification (2026-06-08)

### 实现

| 文件 | 行数 | 职责 |
|------|------|------|
| `core/terminal/subprocess_terminal.py` | 116 | CommandResult dataclass + SubprocessTerminal (exec/read_file/write_file + _safe_path) |
| `channels/terminal_channel.py` | 79 | L1 Builder 适配，4 命令 (exec/run/read/write) |
| `tests/.../test_terminal_channel.py` | 113 | 11 单测全过 |

### MCP 验证

通过 `<bash:exec>` 成功调用 moss CLI 命令（`moss features list`, `moss codex architecture`），
实现递归自举——MOSS channel 内跑 moss 命令。

## Key Decisions

### 1. 不抽协议，但有协议思维

Phase 1 只有 `SubprocessTerminal` 一个实现，不创建 ABC。但方法签名即隐式契约。
Phase 2 有第二个 backend 时再提协议。

### 2. blocking / @nonblocking 标记并行调度

`bash:exec` 和 `bash:run` 的区别在于 `blocking` 参数——这是 Builder API 的原生语义，
不是我们在 channel 层自己实现的。模型通过选择不同命令来表达拓扑依赖。

### 3. CommandResult 是纯 dataclass

```python
@dataclass
class CommandResult:
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
```

当前和 `SubprocessTerminal` 同文件，Phase 2 提协议时分离。

### 4. file 操作走 Terminal 自身，不走 Storage 协议

Terminal 管"操作系统文件系统"，Storage 管"MOSS workspace 持久化"。两个边界。
路径安全：`_safe_path()` 拒绝穿透和绝对路径。

### 5. L1 Builder 模式，不注册 App

`new_channel()` + `chan.build.command()`，不用 StatefulChannel（无状态切换需求）。
不作为独立 App 暴露——Ghost 需要随时可调用的命令，不是进程生命周期。

## Next Phase

- 审批链：allow-all → whitelist → ask-human
- pexpect backend：会话持久性
- 多 channel 拆分：真正的跨 channel 并行
- context messages：进程状态仪表盘
- Terminal 抽象协议：提 ABC，抽 contracts

## 设计参考

- `notebook_channel.py` — L1 Builder + text__ + _safe_path
- `core/concepts/topic.py` → `core/topic/` — 概念层 → 实现层分层模式
- `module_eval_channel.py` — Sandbox 集成 + always_observe 模式
