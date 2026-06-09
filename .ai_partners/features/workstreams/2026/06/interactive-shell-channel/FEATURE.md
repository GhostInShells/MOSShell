---
title: Interactive Shell Channel — pexpect 持久化终端会话
status: in-progress
priority: P1
created: 2026-06-03
updated: 2026-06-09
depends: [ai-terminal]
milestone:
description: >-
  基于 pexpect/PTY 的持久化交互式 shell channel，让 Ghost 拥有持续存在的终端身体。
  一期：单 session channel，As Channel 封装，内存 buffer + 游标模型。
  二期：JSONL 审计轨迹 + Cell 共享终端。
---

# Interactive Shell Channel

## Motivation

bash:exec 和所有 subprocess-based channel 的共同局限：每次调用都是**一次性进程**，没有持久会话状态。

Ghost 无法做到三件事：
- 保持 shell 环境（venv 激活、工作目录、环境变量）跨命令存活
- 让进程在后台持续运行（dev server、训练任务），异步感知其输出
- 与交互式程序对话（REPL、数据库 CLI、SSH session）

pexpect 的 PTY + expect/send 模式填补了这个缺口。

**与 ai-terminal 的关系**：

| | ai-terminal | interactive-shell-channel |
|---|---|---|
| 定位 | 工具型：一次性命令 + 文件操作 | 会话型：持久 PTY 终端 |
| 后端 | subprocess.run() | pexpect |
| 核心交互 | `exec(cmd)` 阻塞等结果 | `sendline(text)` + 流式输出感知 |
| 状态 | 无状态 | 有状态，跨命令保持环境 |
| Channel 类型 | L1 Builder | As Channel（L0 反射 + 薄适配） |

两个不同的交互范式，不是同一个协议的两个实现。interactive-shell-channel 是**独立协议**（Session/Exchange 抽象），不是 Terminal 协议的 pexpect backend。

## Design: As Channel

PexpectSession 是纯粹的 Python 类，零 MOSS 依赖。可独立测试。

```python
class PexpectSession:
    def sendline(self, text: str, *, wait: float = 5.0) -> SegmentResult: ...
    def read_output(self, id: int, *, offset: int = 0, limit: int = 0) -> str: ...
    def sendcontrol(self, char: str) -> str: ...
    def close(self) -> str: ...
```

Channel 适配层用 `new_channel()` + `chan.build.command()` 反射 PexpectSession 的方法为 channel 命令。参考 `module_channel` 的 L0 模式：外部已有的事物包装为 Channel，不要求被包装者感知 Channel 的存在。

## Interaction Model

### 游标 + Segment 模型

终端输出是连续的流。输入和输出交替穿插。每次 sendline 在流上打一个切片标记——以"交互"为边界。

```
时间轴上的输出流:
...output... [游标N] ...output... [input:cmd] ...output... [exit] ...output(live)...
```

每次 sendline 执行完毕时，游标从 N 推进到 N+1。游标之前的输出已消费，游标之后的是"活的"。

### 两段式呈现

同一个输出 segment，在不同位置以不同粒度出现：

**command result（sendline 返回值，进对话历史）**：
```
[segment #12, 547 lines total, tail -200 shown]
... (200 lines of pytest output) ...
3 passed, 1 failed
... 347 lines folded. read_output(id=12) for full content.
```

直接放在 CTML 命令返回值中，**留在对话历史里**。模型三轮后回头找，仍然可见。

**context_messages（滑动窗口，每轮刷新）**：
```
[shell] zsh | cwd:/project | cursor: 21
  segments: 1..21
  live (tail -20):
    $ _
```

只展示瞬时状态——游标位置、历史 segment 列表、live 窗口（命令退出后当前终端画面的最新 20 行）。context_messages 是滑动窗口，下一轮替换。

### sendline 执行语义

- 写入 text + 换行到 PTY
- 如果 `wait > 0`：阻塞等待 shell prompt 重新出现，然后 pop 输出（上次游标 → 命令退出），推进游标
- 如果 `wait = 0` 或为空：写入即返回。输出积累在 buffer，后续 sendline 或 read_output 消费
- Channel 设 `blocking=True`，同 channel 内 sendline 自动排队，保证顺序执行
- 跨 channel：shell channel 的 sendline 阻塞期间，其他 channel 的命令照常并行

### 三段式截断（防信息爆炸）

| 层 | 粒度 | 作用 | 生命周期 |
|---|------|------|---------|
| context_messages | tail -20 行 | "终端现在什么状态" | 当前 keyframe |
| command result | tail -200 行 | "刚才发生了什么" | 对话历史 |
| read_output(id, offset, limit) | 完整 | "我要看全部" | 按需拉取 |

## Commands

```
sendline(text, *, wait: float = 5.0)  → SegmentResult
  主力交互。发送 text + 换行到终端。wait > 0 时阻塞等待命令退出并返回输出。
  wait = 0 时 fire-and-forget。输出中 ANSI 转义序列默认 strip。

read_output(id: int, *, offset: int = 0, limit: int = 0) → str
  按需拉取指定 segment 的完整或部分输出。limit = 0 表示全量。

sendcontrol(char: str) → str
  发送控制字符 (C-c, C-d, C-z)。返回 ack。

close() → str
  关闭 PTY session，返回 exit code。
```

### 降噪模式

read_output 默认 strip ANSI + 控制字符（对应 `moss --ai` 模式）。通过 channel 级配置可切换为 raw 模式（人类看图、彩色输出场景）。

## context_messages

每轮展示：

```
## shell — zsh session
  cursor: 21  |  cwd: /project  |  idle 3s
  segments: [1..21] (use read_output(id) for full content)
  --- live (last 20 lines) ---
  $ _
  -----------------------------
```

- 游标 + 完整 segment 列表：模型知道前面有几轮交互，每轮可通过 read_output 回溯
- live tail：当前终端画面，"人类盯着终端看到的东西"

## Storage

### 一期：零存储依赖

输出 buffer：进程内 `dict[int, str]` + 自增游标。生命周期等于 channel 实例。

### 二期（高优）：JSONL 审计轨迹

**不做审计就不能授权。** 所有交互必须可追溯。

```
session.tmp_storage/shell-sessions/{session_name}/
  audit.jsonl          ← 交互时间线（append-only, 人可读）
  segment_1.txt        ← 完整输出正文
  segment_2.txt
  ...
```

audit.jsonl 格式：
```jsonl
{"type":"session_start","session_name":"dev","shell":"zsh","cwd":"/project","ts":"..."}
{"type":"sendline","seg_id":1,"input":"source .venv/bin/activate","ts":"..."}
{"type":"segment","seg_id":1,"lines":3,"size_bytes":120,"ts":"..."}
{"type":"sendline","seg_id":2,"input":"pytest tests/ -x","ts":"..."}
{"type":"segment","seg_id":2,"lines":547,"size_bytes":13200,"ts":"..."}
{"type":"session_close","exit_code":0,"ts":"..."}
```

人类直接 `cat audit.jsonl` 就能看到完整交互时间线。审计、review、debug 全部可达。

## Cell + 共享终端（二期）

shell session 作为 Cell 运行，人类有自己的 TUI 渲染界面（prompt-toolkit 驱动）。模型和人类**共享同一个终端视野**：

- 人类操作时模型是 passenger（context_messages 里有 live view）
- 模型 sendline 时人类在 TUI 上看到命令执行
- 人类 Ctrl+C 关掉 cell，关闭通知到 channel

## Implementation Plan

### Phase 1（当前）

**文件结构**：
```
core/terminal/
  subprocess_terminal.py   # 已有，不动
  pexpect_session.py       # PexpectSession 类（零 MOSS 依赖）

channels/
  terminal_channel.py      # 已有，不动
  shell_channel.py         # new_shell_channel() — As Channel 薄适配
```

**PexpectSession**：
- pexpect.spawn + janus queue + 线程卸载
- PTY 输出持续写入 buffer
- 游标管理：sendline 时 pop 并推进
- segment 存储：`dict[int, str]`
- 降噪：默认 strip ANSI

**shell_channel**：
- `new_channel(name="shell")` + `chan.build.command()` 反射 PexpectSession 方法
- `chan.context_messages()` 注册 live window 生成函数
- `is_dynamic() = True`（预留 virtual children 插槽）
- instruction 描述：auto-spawn、命令用法、游标模型

**不做的**：
- 多 session / virtual children
- spawn() 显式命令（auto-spawn 在首次 sendline）
- Signal 异步通知
- 磁盘存储 / JSONL 审计
- Cell 封装

### Phase 2

1. JSONL 审计轨迹 — 最高优，权限基石
2. Cell + prompt-toolkit TUI — 人类共享终端
3. 多 session — manager channel + virtual children
4. SSH/REPL 特化变体

## Implementation References

- `module_channel.py` — As Channel (L0) 模式：反射外部对象为 channel 命令
- `app_store_channel.py` — StatefulChannel + context_messages + virtual_children + is_dynamic
- `terminal_channel.py` — L1 Builder 模式，instruction / context 写法
- `speech_channel.py` — Channel interface 风格，生命周期管理

## Design Record

关键设计决策的决策轨迹：

1. **As Channel 而非 Channel Interface** — PexpectSession 是干净的 Python 类，可独立测试。Channel 层只是薄反射。参考 module_channel。
2. **sendline 阻塞 + 游标模型** — 模型通过 sendline 的 command result 拿到输出，结果留在对话历史。context_messages 只做轻量仪表盘。
3. **游标以交互为边界切分** — 不是以时间或字节数。每个 sendline 是一次交互，产生一个 segment。
4. **三段式截断** — 同一个 segment 在 context_messages（-20）、command result（-200）、read_output（完整）三层呈现。防止信息爆炸。
5. **一期零存储** — 输出 buffer 纯内存。不做 JSONL 审计（但二期高优）。
6. **与 ai-terminal 互补不重叠** — ai-terminal 是一次性 subprocess 工具，shell channel 是持久交互会话。两个独立协议，不是同一个协议的两个实现。
