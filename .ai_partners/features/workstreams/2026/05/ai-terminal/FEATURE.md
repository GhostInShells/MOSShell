---
title: AI Terminal — Ghost 的操作系统双手（Subprocesses rebase）
status: in-progress
status_note: >-
  2026-07-18 Phase 2 重启 — matrix 进程管理体系 (Subprocesses) 已落地，
  file_editor_channel 平级落地。terminal channel 瘦身为纯进程控制并
  rebase 到 Subprocesses 契约。设计已收敛，见 Phase 2 章节。
priority: P0
created: 2026-05-29
updated: 2026-07-18
depends: [cell-run-cycle, file-editor-contract]
milestone: prototype
description: >-
  Ghost 最基础的操作系统工具链。Phase 1 (bash.exec/run + file.read/write,
  subprocess.run 后端) 已完成并 MCP 验证。Phase 2: read/write 移交
  file_editor，exec/run rebase 到 Subprocesses（真后台进程 + 退出异步通知
  + 进程可感知可查），两层构建。
---

# AI Terminal

## Motivation

MOSS 的 Ghost 没有操作系统的"手"——不能执行命令。对标 Claude Code 的 Bash，
这是最基础的能力缺口。相对优势不在"bash 执行得更好"，而在 MOSS 的执行模型：
Code as Prompt（Python 签名即接口）+ CTML 并行调度。

## Phase 1（2026-06-12 完成，已压缩）

subprocess.run 后端的 bash.exec/run + file.read/write 原型，11 单测 + MCP
递归自举验证（channel 内跑 moss 命令）。当时的架构细节、五条 Key Decisions
见 git 历史：

```
git log --follow -- .ai_partners/features/workstreams/2026/05/ai-terminal/FEATURE.md
git log -- src/ghoshell_moss/channels/terminal_channel.py
```

仍然有效的判决：
- 与 interactive-shell-channel (pexpect) 是两个独立协议——一次性命令执行 vs
  持久交互会话，不互相实现。
- L1 Builder 模式（`new_channel()` + `chan.build.command()`），不注册 App。
- blocking / @nonblocking 是 Builder 原生调度语义，模型据此表达拓扑依赖。

已被 Phase 2 推翻的部分：假 non-blocking 的 `run`（同步跑完才返回）、
`bash.read/write` 文件动词、SubprocessTerminal 后端的 exec 路径。

## Phase 2 — Subprocesses Rebase（2026-07-18 设计收敛）

> 触发条件已满足：matrix 进程管理体系 (Subprocesses/JobSupervisor 契约 +
> MatrixSubprocessesProvider) 随 cell-run-cycle 落地；file_editor_channel
> 平级落地 (b5605077)。本节是 claude-fable-5 + 人类的设计收敛记录，
> 实装即按此执行，实装中发现问题回讨论修正（features 纪律，不 silent todo）。

### 职责三分（定案）

file_editor 管文件、shell_channel (pexpect) 管持久交互会话、**本 channel 管进程**：
一次性命令执行 + 真后台进程的 spawn/感知/停止。`bash.read/write` 移除。
Phase 1 的假 non-blocking `run`（同步跑完才返回）由 Subprocesses 真后台进程取代。

jobs（调度语义：interval/重启/持久化后台任务）**另开 workstream**，本 channel
零 JobSupervisor 依赖。边界：`bash:run` = owner 生命周期内的一次性后台进程；
jobs = 带调度语义的持久任务。未来 jobs channel 复用本 channel 的 context
行格式约定。

### 三种阻塞机制显式区分（人类判决）

MOSS 有三种阻塞机制，模型要能据此做时序规划，interface 必须显式区分：

1. **同步阻塞**（blocking）— 占据 channel FIFO，同 channel 后续命令等待。
2. **non-blocking** — 不占 channel，但解释器等其返回才进下一关键帧。
3. **全异步** — 命令 spawn 即返回，完成只能通过异步通知感知。

| 动词 | 机制 | always_observe | 说明 |
|---|---|---|---|
| `exec(cmd, cwd='', timeout=60)` | ① blocking | True | shell 模式 + capture，等退出回 stdout/stderr tail + exit code |
| `run(cmd, name='', cwd='', notify_priority=...)` | ③ 全异步 | True | spawn 即返薄回执 (index/pid)，进程结束异步通知 |
| `read_output(index, ...)` | ② nonblocking | True | 读 ProcessOutput 内存 tail 窗口，**默认限长** + offset/limit，附落盘文件路径提示 |
| `stop(index)` | ② nonblocking | False | ManagedProcess.stop() 优雅停止（SIGINT→SIGKILL）。是否需要同步阻塞版留实践verdict，先不过度设计 |

### run 的退出通知（人类判决）

- 退出必发 Signal，notify 模式。默认 `background_notice`（BACKGROUND +
  notify：不抢占注意力，buffer 留痕）。
- `run` 带优先级参数，模型对"死了要紧"的进程（dev server）可升 NOTICE/WARNING。
- 优先级通过 **ProcessMeta.additional (Addition)** 随 meta 走——on_exit 回调
  只收 ProcessMeta，优先级绑在 meta 上是最干净的通路（契约变更 1）。
- **docstring 零 signal 概念**：ghost 只需知道"创建成功，结束会异步通知"，
  不暴露 signal/mindflow 内部词汇。

### 两层构建（人类判决）

- **层 1** `new_terminal_channel(processes: Subprocesses, *, cwd, name='bash')`：
  传入实例。按 `processes.is_running()` 决定是否托管生命周期——已 running
  （如 matrix.processes 共享单例）则只用不管；未启动则 channel 在
  on_startup/on_close 托管 async with（契约变更 2：`Subprocesses.is_running()`）。
- **层 2** `build_terminal_channel(container)`：IoC 工厂。`container.get(Subprocesses)`
  （matrix 场景拿到 per-Matrix singleton）→ 拿不到自建 SubprocessesImpl。
- matrix channel 挂载本能力时，调用层 1 传 `matrix.processes`（函数归
  matrix-channel 实装，cell-run-cycle workstream）。
- **cwd 是构建前参数**：channel 级默认 cwd，exec/run 的 cwd 参数相对它解析。

### 所有权隔离（共享单例场景）

共享 matrix.processes 时，singleton 的 executing() 混入 run_node 的 cell
进程。channel **自持 spawned indices 集合**：context 只展示自己 spawn 的；
`stop(index)` 只允许停自己的（cell 停止归 matrix channel 的 stop(address)，
不开第二条 kill 路径）。

### 暴露的系统级讯息（人类判决）

- **instruction（固定参数）**：[System Context] 块——OS / user / 默认 cwd /
  TZ / lang / encoding（继承 GhostOS TerminalContext 血统），加三种机制的
  使用说明。
- **context_messages（每帧动态）**：后台任务简表——executing (own-only:
  index/name/pid/uptime) + 最近退出 (index/exit_code，非零附 stderr 内存
  tail)。后台任务**可感知 + 可查**（read_output 按 index 回溯）。
- 数据源纪律：全部来自 Subprocesses 内存视图（executing/executed +
  ProcessOutput 内存窗口），不落盘不读账本。

### 契约变更清单

1. `ProcessMeta` 加 `additional: Additional = None`（满足 HasAdditional，
   打开 Addition 生态；本次用于退出通知优先级）。
2. `Subprocesses` ABC 加 `is_running() -> bool`（两层构建的生命周期判据；
   SubprocessesImpl 已有 `_started` 内部态，公开为契约）。

### 实装文件

- `contracts/subprocesses.py` — 两处契约变更
- `core/subprocesses/_impl.py` — is_running 实现
- `channels/terminal_channel.py` — 重写（两层构建 + 四动词）
- `tests/ghoshell_moss/channels/test_terminal_channel.py` — 重写
- `channels/CLAUDE.md` — 顺手修：channel 开发前必读三件套
  (channel_builder / states_channel / ctml read) 前置到构建梯度之前
- `core/terminal/subprocess_terminal.py` — 不动（read_file/write_file 仍被
  引用处理时再清理；exec 路径由 Subprocesses 取代）
