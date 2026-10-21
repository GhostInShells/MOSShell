---
title: Interactive Shell Channel — 持久交互终端与 shell node
status: in-progress
status_note: >-
  2026-10-17 设计复活并定稿：shell channel 承载感知面状态机，nodes/os/shell 做
  os-control 可视化。实装排队中，尚未接线到任何运行时。
priority: P1
created: 2026-06-03
updated: 2026-10-17
depends: [ai-terminal, moss-os-control]
milestone:
description: >-
  基于 pexpect/PTY 的持久多 session 交互终端。shell channel 承载感知面状态机、
  数据模型与协议接口；nodes/os/shell 在其上做审批卡、可视面与审计。
---

# Interactive Shell Channel

> Use `moss features set-status interactive-shell-channel <status> -m "note"` to update state.

## Motivation

`bash:exec` 与所有 subprocess-based channel 的共同局限：每次调用都是一次性进程，
没有持久会话状态。Ghost 因此做不到三件事：

- 保持 shell 环境（venv 激活、cwd、环境变量）跨命令存活
- 让进程在后台持续运行（dev server、训练任务），异步感知其输出
- 与交互式程序对话（REPL、数据库 CLI、ssh、claude code）

**与 terminal 的分工**：terminal 管**进程**（一次性命令、卡片审批、退出通知）；
本 feature 管**持久交互会话**（PTY、环境存活、交互点等待）。两个独立协议，不互相实现。

**为什么历史上搁置**：持久 PTY 的能力面太强（ssh → telnet → 任意交互程序），
而当时没有人机协作边界。现在边界成立：

- 开启会话是唯一审批动作，人类在审批卡上参与编辑（尤其 password 等 secret）
- 开启之后人类只做**视觉监控 + 急停**，不再逐条审批
- 所有交互 append-only 审计，secret 永不落盘

因此可以正式化，而不是继续停在 alpha。

## 2026-10-17 复活的结论

### KD1 — 分层：channel 是感知面状态机，node 是可视化

```
core/terminal/pexpect_session.py     PTY 运行时（零 MOSS 依赖）
        ↓
channels/shell_channel.py            感知面状态机 + 数据模型 + 协议接口（基础）
        ↓
nodes/os/shell/                      审批卡、xterm、多 session 可视化（os-control 面）
```

- `shell_channel.py` **不删除**，作为基础持续演进。2026-06 的实现是 alpha 底座。
- 感知面状态机需要的数据结构、状态迁移、接口全部在 channel 里做完整。
- node 不持有任何会话真相，只实现审批、渲染与 Matrix 接线。它可以在 channel
  跑通之后独立替换，channel 也可以脱离 node 用于 MCP / 测试 / 其他 Ghost。
- 这是 `moss-os-control` KD6 的既有路线：alpha channel 是实现底座，node 是产品面
  （terminal 已经走过一次）。

### KD2 — 审批在 `open`，且是唯一的人机边界

- `open` 生成一张审批卡：argv / cwd / env / startup / profile 可见、可编辑。
- **secret 只由人类填写**（ssh password 等）：模型上下文、notice、signal、审计里
  都只见占位符；accept 后由 node 在匹配到约定 expect（如 `password:`）时注入。
- accept 之前，人类可以现场改写 spec；模型收到的 accept signal 携带**脱敏后的最终 spec**。
- accept 之后，人类只剩视觉监控与急停（`interrupt` / `close`），没有逐条审批。
- **`approve` 绝不能是 channel command**。模型不能自批。审批只通过 `Approver`
  协议从 node 面进来；审批状态仍属于 channel 状态机（`pending_approval`）。

否掉的方案：逐条输入审批（打断类人终端的连续性，terminal 已覆盖进程审批）；
auto-spawn 无审批（能力面太强，没有边界）。

### KD3 — 多 session 是平表控制面

- 一个 node 承载多个 session；surface 上以 tabs / 卡片并列展示运行中的终端。
- session 按 `name` 寻址，不物化成 virtual children。控制面是平的：
  「一个 channel + 一组吃 name 的动词」。参考 `deepseek_harness/channel.py`。
- session 与 node 同生命周期：node 重启即全灭，v1 不做跨重启 reattach。
- 感知面用 `named_notices` 按 session 发布状态（支持宣布消失），不用单值 notice。

### KD4 — 流式 reader，不是 expect 消费

2026-06 的实现用 `expect(prompt)` 即读即消费，有三个结构性问题：

1. 命令运行中的输出不在任何视图里，“后台长任务可感知”是假的。
2. 两次 expect 之间没人读 PTY，输出超过内核缓冲后子进程会被反压。
3. REPL / TUI 永远等不到 shell prompt，segment 边界错位。

v2 的模型：

- 每个 session 一个独立 reader 线程，持续消费 PTY。
- 输出只进 **append-only buffer**；`Cursor` 是 buffer 上的绝对位置，不是 expect 的副作用。
- prompt 检测与 expect 匹配降级为 stream 上的观察者，不拥有消费权。
- **单流双面**：raw 流 → 人类面（xterm.js，与 terminal 的流式 stdout 同一表现层）；
  清洗/结构化流 → 模型面（notice / command result / peek / read）。
- ANSI 清洗必须**增量**处理（转义序列可能跨 chunk 切开），不能对 chunk 独立处理。
- 交互点统一定义为：expect 命中 / prompt 回归 / 进程退出 / 需要模型决策的提示。
  密码提示不算交互点，它在 secret 绑定里被 node 消化掉。

### KD5 — 输出是 raw 真相 + projection profile

- raw bytes 是唯一真相，始终保留给人类面与审计。
- 模型面是 projection：session 在 `open` 时绑定一个 `OutputProfile`。
- `OutputProfile` 是接口：增量 parser（chunk → events）+ model renderer；
  human renderer 由 node 选择（xterm raw 或结构化卡片）。
- node 不硬编码输出语义。默认 profile 是 shell/line；ssh 与 claude code 各是 profile。
  这也是后续自定义输出结构的接缝：parser 可插拔，不影响 session/游标/审计。

### KD6 — 审计与 secret 隔离是 v1 的承重件

> 不做审计就不能授权。

- 每个 session 一份 append-only `audit.jsonl` + raw 输出文件；
  每次输入输出、状态迁移、审批决定都落一行。
- secret 值只活在 channel 内存里；进入审计、notice、signal、model view 之前统一 redaction。
- 审计落点由 `AuditSink` 注入：默认内存，node 传 `runtime/sessions/`。
- channel 不依赖具体文件系统布局，node 决定持久化位置。

### KD7 — 感知面遵守注意力纪律

- `notice` 只放计数（sessions / pending / running / waiting / exited）。
- `named_notices` 只放 **watched** 的 session 状态；默认 silent background。
- **不用 `context_messages` 塞 live tail**——hot 面会被 session 数量放大成噪音。
  当前终端画面用 `peek` 拉取。
- 信号只给交互点与异常退出；等待的完成通过 signal 交给 mindflow 异步接住。
- watch 分级沿用 dsh 的 `row / notify / next`，不发明第二套注意力模型。

## 数据模型

```python
class SessionState(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    STARTING = "starting"
    READY = "ready"          # prompt 已就绪
    RUNNING = "running"      # 有命令在跑
    WAITING = "waiting"      # 注册了 expect 等待
    CLOSED = "closed"
    EXITED = "exited"
    FAILED = "failed"

@dataclass(frozen=True)
class SecretSpec:
    name: str
    expect: str              # 命中后注入 secret，不进模型面

@dataclass(frozen=True)
class SessionSpec:
    name: str
    command: str             # 启动命令（ssh / bash / claude ...）
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    startup: list[str] = field(default_factory=list)
    profile: str = "shell"
    secrets: list[SecretSpec] = field(default_factory=list)

@dataclass(frozen=True)
class OutputSpan:
    session: str
    start: int               # buffer 绝对偏移
    end: int

@dataclass
class WaitSpec:
    pattern: str
    timeout: float
    state: Literal["waiting", "matched", "timed_out", "cancelled"]

@dataclass
class Exchange:
    id: int
    session: str
    source: Literal["model", "human", "system"]
    input: str               # 脱敏后
    output: OutputSpan | None
    wait: WaitSpec | None
    state: Literal["sent", "waiting", "completed", "timed_out", "interrupted", "exited"]
    created_at: float
    settled_at: float | None = None

@dataclass
class SessionInfo:
    name: str
    spec: SessionSpec        # 脱敏后
    state: SessionState
    cursor: int              # 当前输出末尾
    exchange_ids: list[int]
    started_at: float
    exit_code: int | None = None
```

## 协议接口（channel 定义，node 实现）

- `Approver.approve(spec) -> ApprovalDecision`
  - `accepted: bool`
  - `spec: SessionSpec`（人类可改写）
  - `secrets: dict[str, str]`（只在内存里）
  - `note: str`
- `Observer`：`on_chunk(session, raw, offset)` / `on_session(SessionInfo)` /
  `on_exchange(Exchange)` / `on_interaction(session, exchange_id, kind, tail)`。
  node 的 xterm 与卡片都吃这个流。
- `AuditSink.record(event: dict)`。
- `OutputProfile`：`parse(session, chunk) -> list[OutputEvent]` /
  `render_model(events, state) -> str`；profile 注册表由 channel 持有。
- `Signaler`：默认走 `CommandUtil`，node 注入 `matrix.send_signal_to_ghost`。

## 命令面

```
open(name, command, cwd="", env="", startup="", profile="shell", secrets="")
    → 回执 pending approval；人类审批后 session 启动并回 signal
send(name, text, expect="", timeout=5)
    → 回执 exchange id；不阻塞。expect 命中 / 超时 / 退出都回 signal
wait(name, expect, timeout)
    → 纯等待，不发输入；命中后回 signal
peek(name, lines=20)
    → 当前终端画面的非消费视图
read(exchange, offset=0, limit=0)
    → 读取该 exchange 的完整或部分输出
interrupt(name) / key(name, char)
close(name)
sessions()
watch(name, level=row) / unwatch(name)
```

命令结果默认 tail -200 + folded 标记；完整内容走 `read(exchange)`。
这是 2026-06 三段式截断的延续，只是 context 段被 `peek` 取代。

## node 侧职责（`nodes/os/shell`）

- `main.py`：Matrix 进程、ephemeral port、WebView 声明、`url` notice。
- 审批卡：可编辑 spec + masked secret 字段；accept 后把 `ApprovalDecision` 交回。
- xterm.js：订阅 raw 流，多 session tabs；人类只读 + 急停按钮。
- exchange 卡片：输入、来源、输出范围、状态；可回溯。
- 审计查看：直接读 `runtime/sessions/` 的 JSONL。
- 不实现任何 shell 语义；所有会话真相来自 channel。

## 验收场景

### ssh

审批卡填 host / 用户 / secret → accept → 自动在 `password:` 注入 → 远端 shell。
模型可以发远端命令、`wait` 长任务、`peek` 当前画面；人类全程可视。
secret 不出现在模型上下文与审计里。

### claude code

在 PTY 里跑 `claude`。人类面看原始 TUI（xterm）；模型面走 profile：
按键交互，或绑定 `--output-format stream-json` 的结构化 profile。
同一个 session、两种渲染，验证 profile 边界。

## v1 范围

- 单 node、多 session；bash / zsh；shell prompt 会话 + 显式 `expect`。
- 审批 + secret 注入 + 审计 + 视觉监控 + 急停。
- 不做：人类在运行中直接输入、跨重启 reattach、逐条审批、virtual children、
  telnet 特化、分布式 session。

## 历史与继承

2026-06-03 创建，06-09 定稿交互模型并实装 alpha：

- `src/ghoshell_moss/core/terminal/pexpect_session.py` — PexpectSession
- `src/ghoshell_moss/channels/shell_channel.py` — As Channel 薄适配
- 11 个单测

旧设计仍然有效的部分：游标 + segment、tail -200 + folded 标记、`read_output`
按需拉取、JSONL 审计作为权限基石。

被 2026-10-17 推翻的部分：blocking `expect` 消费模型、单 session、channel 作为最终产品、
“Cell 共享终端”作为二期（现由 node + surface 承接）。旧文与讨论原文见
`discuss/2026-06-09_interaction_model_design.md` 与 `git log --follow -- FEATURE.md`。

## 当前状态与下一步

设计已定稿，实装排队中。落地顺序：

1. channel 数据模型 + 状态迁移 + `Approver` / `Observer` / `AuditSink` / `OutputProfile`
   接口，用假实现写单测。
2. PexpectSession v2：reader 线程、append-only buffer、增量 ANSI 清洗、交互点观察。
3. shell_channel 命令面 + notice / signal / 多 session。
4. `nodes/os/shell` 可视化：审批卡、xterm、tabs、exchange 卡片。
5. ssh / claude code 两个场景验收。

## Implementation References

- `channels/terminal_channel.py` — 卡片 + 全异步回执 + signal 注入的先例
- `nodes/os/terminal/` — 一进程一面一 channel、ephemeral port、surface 接线
- `nodes/os/file_editor/` — 共享感知与审批边界的产品化先例
- `deepseek_harness/channel.py` — 平表多会话感知面（notice 计数 + watched named notices）
- `channels/module_channel.py` — As Channel (L0) 的反射姿态
