---
title: Interactive Shell Channel — pexpect 持久化终端会话
status: draft
priority: P1
created: 2026-06-03
updated: 2026-06-04
depends: []
milestone:
description: >-
  基于 pexpect/PTY 的持久化交互式 shell channel。P1 因为是"质变"——从一次性 tool call 到持续操作系统感知。
  Channel interface 仍在推演，方案收敛后进入实现。验证方式：moss-as-mcp 自体验迭代。
---

# Interactive Shell Channel

> `moss features set-status interactive-shell-channel <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Skills 体系和现有 subprocess-based channel（如 mac_channel）的共同局限：每次调用
都是**一次性进程**，没有持久会话状态。Ghost 无法：

- 保持 shell 环境（venv 激活、工作目录、环境变量）跨命令存活
- 让进程在后台持续运行（dev server、训练任务），异步感知其输出
- 与交互式程序对话（REPL、数据库 CLI、SSH session）
- 对操作系统做运行时级操控（信号、PTY 控制、gdb attach）

pexpect 的 PTY + expect/send 模式填补了这个缺口。它让 Ghost 拥有一个"活的终端身体"
——一个持续存在的、状态累积的、可异步感知的 shell 会话。

**P1 判断**：这不是"更好用的工具"，而是 MOSS 架构表里 Duplex（感知-思考-行动重叠）
和 Active（身体作为主动传感器）两行的直接落地。持久交互式会话 + 异步感知的组合，
是 Ghost 降临的基础器官。

## Design Index

- 通用交互协议设计: `design/`
- 关键讨论: `discuss/`

## 验证方法

通过 `moss-as-mcp` 连接 Claude Code，AI 一边使用 shell channel 一边改它的 interface。
以下所有选项都不是定案，需要在自体验中迭代确认。

---

## 已定案

### 抽象协议 + pexpect 实现

不只做 pexpect channel，而是设计一层通用的 **交互式进程会话协议**。
Channel interface 层定义 Session/Exchange 抽象，pexpect 是实现之一。
以后可替换为 paramiko（SSH）、serial（嵌入式串口）等。

### blocking=True，优先做阻塞 command

pexpect session 天然串行。Channel 设 `blocking=True`，多个 Command 自动排队。
这是 MOSS 已有原语，零额外成本。先做阻塞 command 跑通，再做异步 command（signal task 等）。

### 实现参考

- `app_store_channel`: StatefulChannel, `is_dynamic()`, `get_virtual_children()`, context_messages 模式
- `fractal_hub`: ChannelFactory 注册模式
- `speech_channel`: Channel interface 风格，生命周期管理

---

## 设计选项：单 Session

以下均为备选方案，通过实际使用迭代选择。

### context_messages — 被动感知

AI 在每个关键帧看到的终端状态。

**呈现模式**：

| 选项 | 行为 | 适用 |
|------|------|------|
| A) Tail | 展示输出 buffer 最后 N 行 | 快速恢复上下文 |
| B) Delta | 只展示上次 interaction 后新产生的输出 | 省 token，默认候选 |
| C) Full buffer | 展示全部 buffer | 短 session 可行，长 session 炸 |
| D) 可切换 | Command 参数控制，或 Channel 配置 | 最灵活 |

**Buffer 消费语义**：

| 选项 | 行为 |
|------|------|
| pop | 读走移除，不重复消费 |
| peek | 读不走，每次刷新都看到同样历史 |
| indexed pop | 维护 cursor，记录上次消费位置 |

在实际操作中迭代确认哪种组合最自然。

### command result — 主动响应

命令执行完毕后的返回。需要在实际使用中确认：
- 只返回结构化结果（exit code, pattern match）
- 附带本次交互产生的输出文本（全量 vs 仅匹配段）
- pexpect 的 `before` / `after` / `match` 如何映射到 return

一个备选方向：command result 吐 buffer 文本，context_messages 只标记 session 信息和状态。
两个出口各走各的，不互相污染。

### 异步通知 — Signal

**触发事件候选**：
- 进程意外退出
- 输出匹配到配置的 pattern（"ERROR"、"CRASH" 等）
- 命令执行完毕

**触发方式**：
- 默认行为 vs opt-in flag vs Channel 级配置
- 需要考虑高频日志场景下的噪音问题

Signal 不是必须的——先做阻塞 command，在实战中感受是否需要异步通知。

### Command 集（初步）

```
spawn(cmd, *, cwd, env)          → session info
sendline(text)                   → output / match info
expect(pattern, *, timeout)      → before, after, match
read_buffer(*, since)            → new output since cursor
sendcontrol(char)                → None
close()                          → exit code
signal(sig)                      → None
```

粒度选择待验证：底层 sendline/expect 组合 vs 高层 execute(cmd) 一键。

---

## 设计选项：多 Session

多 session 的核心价值是**跨 channel 并行控制**——多个终端 session 在 CTML 中同时执行。

### 问题：interface 去重

无论哪种方案，每个 session 的 interface 都是相同的（sendline/expect/close）。
如果每个 session 是独立 channel，`moss_dynamic` 会重复展示 N 份相同的 Python interface。
目前 Channel 体系没有 Class 语义（`channelname(Class)`），无法声明"这些 channel 共享同一个 interface 定义"。

以下方案在不同层面解决此问题，可组合使用。

### 方案 1：__content__ 做终端输入，instruction 替代 Python interface

Session channel 不暴露显式 command 签名。非标记文本通过 `__content__(chunks__)` 流入终端。
`__content__` 的 prompt/行为描述提到 instruction 中，不走 `moss_dynamic` 的 Python interface。
控制命令（spawn/close 等）留在父 channel，显式传 session id。

- 优点：interface 去重问题自然消失；模型从 instruction 理解"这是一个终端，打字就是输入"
- 缺点：`__content__` 语义是输入而非输出，终端输出需要走其他出口（context_messages / command result）

### 方案 2：命令全在父 channel，带 session id

```
shell:sendline id="1">ls</shell:sendline>
shell:expect id="1" pattern="$"/>
shell:close id="1"/>
```

父 channel 持有所有命令，session id 用数字索引（`1`, `2`, ...）。context_messages 排列所有 session
的快照（序号、状态、一行摘要）。

- 优点：无 interface 去重问题；实现最简单
- 缺点：失去多 channel 并行——所有 session 命令排队在父 channel

### 方案 3：virtual_sub_children + Class 语义（需新机制）

每个 session 是动态注册的 virtual child channel。长期方向是引入 Class 语义：
`moss_dynamic` 对同 Class 的 channel 只展示一次 interface，实例引用它。

- 优点：CTML 寻址天然（`shell/session:1:sendline`）；每个 session 的 context_messages 自治；生命周期独立
- 缺点：依赖尚未实现的 Class 语义

### 方案 4：CTML scope 做 session 分组

```ctml
<shell:session id="1"><shell:sendline>ls</shell:sendline></shell:session>
```

使用 CTML scope 语法管理 session 上下文。但 scope 内所有 session 都阻塞。

### 组合方向

方案 3（多 channel 并行）是终极形态，但依赖 Class 语义。
方案 1（__content__）在 Class 语义就绪前可以作为轻量替代。
方案 2（父 channel 集中）可以作为最早的 MVP 验证 session 概念。
方案 1 + 3 可以并存：__content__ 处理输入流，Class 语义解决 interface 去重。

在实际使用中迭代选择。

---

## Implementation Notes

- pexpect 是同步库。命令卸载到单独线程，通过 janus queue + ThreadSafeResult 桥接
- PTY 的输出 buffer 需要在 thread 侧维护，context_messages 的生成需要线程安全
- 取消语义：asyncio CancelledError → ThreadSafeResult.cancel() → 线程侧清理
- 先做本地 shell，SSH/REPL/DB CLI 作为后续特化变体
- 实现顺序：单 session 阻塞 command → context/result 选型迭代 → 多 session

## 与现有模块的关系

- `mac_channel`: 互补。mac_channel 是 macOS 专用（JXA），一次性调用。shell channel 跨平台、持久会话
- `speech_channel`: 并行。语音 + shell 同时运行，context_messages 各自独立，构成 duplex 演示
- `notebook_channel`: 不重叠。notebook 是文件系统 CRUD，shell 是进程交互
- `app_store_channel`: 参考其 StatefulChannel + virtual_children + context_messages 模式
