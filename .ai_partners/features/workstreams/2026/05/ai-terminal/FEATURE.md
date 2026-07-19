---
title: AI Terminal — Ghost 的操作系统双手（Subprocesses rebase）
status: in-progress
status_note: >-
  2026-07-19 Phase 2 实装完成 + 首轮 MCP dogfooding 通过 (4 动词 + 隔离 +
  timeout)。dogfood 中发现两处设计漂移: build_* 工厂签名混入配置项、exec/run
  沿用 cmd:str 属性传参。均已修正 (file_editor 同治)。
  下轮 dogfooding: terminal + file_editor 复合场景（写文件+改文件+跑测试）,
  联合验证前留在下一化身的 Phase 2 实装轨迹章节里.
priority: P0
created: 2026-05-29
updated: 2026-07-19
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

## Phase 2 — 实装与 dogfooding 首轮（2026-07-19）

Phase 2 设计的直接实装 + 挂到 `system_test` mode + `moss-as-mcp` 里从
Claude Code 实跑一轮。dogfood 暴露了两处设计漂移，同轮修正。file_editor
同姐妹病同治。

### 首轮 dogfooding 结果（4 动词全通）

- `exec` 阻塞返回 + stdout/stderr + exit tail (0 / 非零 / timeout)
- `run` 立即回 receipt (index + pid + name)，进程后台运行不阻塞
- `read_output` running / exited 两态 tail marker 正确
- `stop` 正常停 running 进程 (SIGINT)
- 未知 index、foreign index 隔离拒绝
- Default cwd = workspace root (`.moss/`) 已注入 System Context
- 所有权隔离：context 只显示 own spawned

### dogfood 发现的两处设计漂移（同轮修正）

**漂移 1：`build_*_channel(container, *, channel_name)` 混签名**

`ChannelFactory` 契约是 `(IoCContainer) -> Channel | None`。原设计里
`build_*` 又要吃 container 又要吃 channel_name 配置项，违反契约——
`main.import_channels(build_terminal_channel)` 只能用默认 name，改 name
就得 lambda 包一层。file_editor_channel 同病。

修正：`build_*` 高阶化，返回 factory：

```python
def build_terminal_channel(
    *, name: str = "bash", description: str | None = None,
) -> ChannelFactory:
    def factory(container: IoCContainer) -> Channel: ...
    return factory
```

调用：`main.import_channels(build_terminal_channel())` 或
`main.import_channels(build_terminal_channel(name="sh"))`。

**漂移 2：exec/run 用 `cmd:str` 属性传参**

Phase 1 老 terminal 的 `write` 已经用 `text__` (CDATA)，是对齐姿态。
老 exec/run 用 `cmd:str` 属性传参才是漂移——shell 命令高频用 `&&` `||`
`>&` 等 XML 冲突字符，属性版逼模型心算转义。Phase 2 重写时我延续了老
exec/run 的漂移，没纠回 write 的对齐姿态。dogfood 里第一次拿 CTML 写
`ls foo 2>&amp;1 &amp;&amp; echo done` 时体感立刻炸出问题。

修正：exec/run 改用 `text__: str = ""`，CDATA 里想怎么写怎么写：

```ctml
<bash:exec><![CDATA[ls foo 2>&1 && git log --oneline -3]]></bash:exec>
```

### dogfood 另外发现的中低摩擦点（同轮修正）

- **`raise_observe` 太重** — stop/read_output 未知 index 用了
  `CommandUtil.raise_observe`（本质 `raise ObserveError`），一个操作错误
  打断整批并行命令。改成 `return CommandUtil.observe(...)` soft observe：
  兄弟命令继续跑，只标记这一条值得观察。**Observe 契约备忘**：`Observe`
  是返回值 (soft)，`ObserveError` 是异常 (hard)，`command.py:764-793` 定义。
- **exec 污染 context** — exec 是一次性同步命令，返回值已完整给出。原
  实装把它进 `spawned` dict 导致 `recently exited` 一直显示 exec 记录。
  改：exec 不进 dict，`spawned` 从此只装 run 起的后台进程。
- **context 只有 name 没 cmd** — 加 60 字符 cmd 摘要 (`meta.command`)。

### 更新的实装文件（本轮）

- `channels/terminal_channel.py` — build 高阶化、text__、soft observe、
  context 加 cmd 摘要、exec 不进 spawned
- `channels/file_editor_channel.py` — build 高阶化（同治漂移 1），new
  加 description 参数
- `.moss/modes/system_test/src/HOST/channels.py` — 挂
  `build_terminal_channel()`
- `tests/ghoshell_moss/channels/test_terminal_channel.py` — 21 单测，
  覆盖 text__ 姿态、空 body observe 提示、exec 不出现在 context、soft
  observe 语义（`await task` 返回 Observe 对象非 raise）
- `tests/ghoshell_moss/channels/test_file_editor_channel.py` — 更新
  `build_file_editor_channel()(container)` 高阶调用姿态

### 未修 / 待下轮 dogfood 验证的摩擦点

**下轮 dogfooding 目标**：terminal + file_editor 一起测。复合场景（写
文件 + 编辑代码 + 跑测试 + 观察结果）才能暴露两个 channel 组合起来的
摩擦。以下是本轮识别但未修的点：

1. **Default cwd = workspace root `.moss/` 而非 project root**
   `<bash:exec><![CDATA[ls src/...]]></bash:exec>` 会失败，因为默认
   工作在 `.moss/` 下。System Context 已经写了 `Default cwd:` 但排位
   不显眼。判断题：cwd 默认改成 project root 更符合直觉？还是加强
   instruction 提示？下轮 dogfood 拿到实际写代码场景再定。

2. **`run` 退出 signal 在 CTML-as-tool 模式下不可观察**
   MCP 是 CTML 单次调用，无 persistent attention loop 接住
   `session.add_input_signal`。功能存在但反馈不透明。cell 集成完成后
   开一个 **signal-watcher cell** 专门监控所有 signals，才能观测到。
   本 workstream 不修，属于 cell-run-cycle 生态。

3. **nonblocking 编排时序**：CTML 里 stop 排在 run 后面，两个都是
   nonblocking，编排上 stop 先起时 run 的 spawn 尚未落 dict，`stop`
   会看到 `yours: []`。虽然这次场景是"故意用错 index"没暴露真问题，
   但真业务里 `<bash:run/><bash:read_output/>` 组合可能踩。要么在
   `read_output` / `stop` 里加短暂 retry，要么规范"配对操作用作用域
   `<_ until="flow">` 包起来"。下轮 dogfood 遇到时决定。

### 给下一个化身的信

1. **顺手工作**：`.moss/modes/system_test/src/HOST/channels.py` 里加挂
   `build_file_editor_channel()`，用 `moss-as-mcp --mode system_test`
   起 MCP，跑 terminal + file_editor 复合场景 dogfood。

2. **Observe 语义** — 本轮我踩过：
   - `return CommandUtil.observe(str)` = **返回 Observe 值**，soft 通知
     模型这一条值得观察，不打断兄弟。
   - `CommandUtil.raise_observe(str)` = **raise ObserveError**，hard
     中断所有并行命令。stop/read_output 未知 index 属于**用户操作错误**
     不该用 raise_observe。契约定义在 `core/concepts/command.py:764-793`。

3. **抄旧代码时要 challenge** — 我在 Phase 2 rebase 时抄了 Phase 1 老
   exec/run 的 `cmd:str` 姿态，但 Phase 1 的 `write` 已经用 `text__`
   姿态了。老代码不是所有部分都值得抄；同一份代码里都可能有已对齐的
   部分和已漂移的部分。抄之前先看邻近代码的姿态是否一致。

4. **build_* 工厂签名规范**：`build_*(*, name, description, ...) ->
   ChannelFactory`，返回的 factory 严格 `(IoCContainer) -> Channel`。
   两个正式 channel（terminal / file_editor）现已对齐此模式。以后新增
   channel 直接抄这个姿态。

5. **本轮通过 `moss --ai codex get-source
   ghoshell_moss.channels.terminal_channel` 可直接查看实装**。核心
   工厂函数结构：`build_*_channel` (高阶配置) → `factory(container)` →
   `new_*_channel(contract, *, cwd, name, description)` (纯组合)。
