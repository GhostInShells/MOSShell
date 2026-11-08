---
title: Dsh Node
status: draft
priority: P1
created: 2026-11-07
updated: 2026-11-09
depends: []
milestone: dsh 0.2.0-rc.2
description: >-
  Two dsh nodes (process launcher + web control) that dogfood the dsh-layer control
  surface before dolores lands it: users' sessions are read-only, spawned agents are
  memento branches, and sessions are rebuildable carriers of a unique trajectory.
  It also carries the **first implementation** of the memento-agent contract.
---

# Dsh Node

> 归因：`他` = thirdgerb 说的；`我推` = moss 的推断，未经裁决；`待定` = 已知未决。

## Why

- 在 dolores 实装前，先找一个**破坏面低**的地方做 dogfooding，把"dsh 层面向 ghost 的控制面"练熟。（他）
- 节点本身也要能用：离开 ghost 也跑得起来，有自己的观测面。（他）
- 接缝层 `ghoshell_moss.deepseek_harness` 的定位 = **支持多版本 dsh 的 channel**，不直接属于 dolores；它自己也有多版本控制问题。（他）
- 能开在 dsh connection 上的接口/controller 必须由我们提供，**不许调用方再发明一遍**。（他）

## 形状：两个 node

launcher 与 web connection 是两个概念，因此是两个 node。（他）

| node | 拥有什么 | 通道面（草案） |
|---|---|---|
| dsh launcher | 进程：start/stop/restart、home 初始化与安装、token/web_url、输出与退出 | `start / stop / status / logs / home / install` |
| dsh web | 连接：感知、用户 session 只读、agent 控制、signal | `sessions / agents / env / …` |

- 组合方式：launcher 报 `web_url()`，web node 用它 attach。
- singleton：**一台 home 一个 launcher**；**一个 host:port 一个 web**（沿用现有 flock 语义）。
- 引擎要暴露两个 facade：`DshProcess` 与 `DshControl`；今天 `DshLauncher(DshConnection)` 的 is-a 是历史痕迹，这一步正好拆开。（我推）

## 层与依赖

- 引擎 `ghoshell_dsh`：launcher / connection / session / blueprint / plugin / skills，**不含 MOSS 概念**。
- 接缝 `ghoshell_moss.deepseek_harness`：会话台账、watch 级别、signal 派发、CTML 通道、节点入口。
- 判据（我推）：接缝里出现 dsh 的 RPC 名/字段名 = 放错层；`node.py` 里出现 RPC 名、自己维护台账、或写等待循环 = 层没做够。
- extras：新增 `[dsh]`（`ghoshell-processes` + `ghoshell-dsh` + engine 需要的 `httpx`/`websockets`）；`[dolores]` 从 `[host]` 解耦是另一条线。
- **导出污染（可测）**：只装 `ghoshell-moss` 的环境里 `import ghoshell_moss` 与 `moss --help` 必须成功；`import ghoshell_moss.deepseek_harness` 应报"缺 dsh extra"的清楚错误，而不是让 moss 起不来。

## 版本化节点

- `nodes/dsh/v020/` 这样的目录结构，每代一个；本质是**样例**。`main.py` 直接来自 `deepseek_harness/v0_2_0/node.py`（源码定义 + 薄壳）。（他）
- **`.dsh` 已存在时初始化不能覆盖**：settings / storages / workspace.json 不动；profile patch 只补缺行；插件安装幂等。（他）
- 推论（他）：这套机制意味着，未来多数 dsh 开源项目都能变成"可控制的 dsh 项目"。
- per-node venv 属于未来扩展性；先走"调用方安装 / `[host]` 自带"。（他）

## 感知面

| 感知 | 来源 | 现状 |
|---|---|---|
| workspace（仅 project 路径内） | `connection.workspaces()` + `attach_workspace` | 已有；加"限定项目根"过滤 |
| providers | `session.llm_providers()`（live / declared） | 已有 |
| agent presets | 无面 | **缺口**（engine 补只读路由，或 bash `--dump-config`） |
| plugins | 无面 | **缺口**（profile patch + node_modules 扫描） |
| 环境（home/binary/version/token/web_url/进程态） | launcher + connection | 半有；token / `web_url()` 面待显式化 |

原则（他）：**稳定感知走 channel，迭代探索走 bash**；dsh 自身的迭代逻辑仍走 bash。

## Agent 模型（这一波要验证的核心）

底层命题不变（他）：**Agent = memento branch**；branch 是身份，dsh 是运行时；运行时数据走 dsh，轨迹走 branch。

**D09 深夜，抽象面整体重写并已落盘**：`src/ghoshell_moss/agents/blueprint/memento_agent.py`
（提交 `321818ab`）。**这个 workstream 承载它的第一版实现** —— memento-agent 不单独开 workstream。
契约以那个文件为准，本节只做索引，不重抄。

### 决策索引（细节在契约文件里）

| 决定 | 一句话 | 落点 |
|---|---|---|
| **异步是原语** | `prompt(message, mode=queue\|steer)` 只回执、不等轮；观测走 `on_commit` + `state()`。同步只是补充，永不作为原语 | 模块头 "The interaction is asynchronous" / `MementoAgent.prompt` |
| **没有 close，只有 reboot** | 起 session 时写一条 commit 当定位点，于是**指针永不为空**；"从轨迹重建"是唯一的重启语义 | `MementoAgent.reboot` |
| **一层，不做 fan-out** | 只有 hive 能 spawn，agent 不能；team 与树形留到下一期 | 模块头 "No fan-out" / `Hive.spawn` |
| **配置是运行时能力** | preset 只暴露 `config_schema()`；`configure(patch)` 在下一轮边界生效，缺省即保持 | `AgentPreset.config_schema` / `configure` |
| **读归 hive** | `read`（commit view）/ `transcript`（区间在引擎里的模型行为）/ `chat`（对着那一段问一句，不留痕）；原始事件层不暴露 | `Hive.read/transcript/chat` |
| **声明由委派方写** | 第一条 commit 是唯一例外，之后只有载体写 | `Hive.spawn` |
| **`Link` 不再建模** | 引擎坐标落进 session-start commit 的 `metadata`，由实现定义，memento 不解释 | `reboot` |

### 走过的岔路（不要再走一遍）

1. **`step(message) -> str`**：把"跑完一轮"做成原语，等于让 ghost 阻塞在一个 agent 上，swarm 退化成一个队列。
2. **"收窄抽象"时把 read / chat / history 一起删掉**：那是删能力，不是收窄边界。DSH 前两天做的补丁（全局读 commit ref、对区间提问）正是 agent 的关键。
3. **拿 `Memento.on_event('confluence')` 当投递触发器**：投递是 agent 内部总线的**直达动作**，memento 记录只是同一时刻的附属写；没有活 session 才排队。

### dsh 侧已核实的事实（可直接依赖）

- `session/prompt` **原生**有 `mode="queue" | "steer"`：`prompt` 的语义不需要我们发明。
- `session/run()` 是阻塞单轮，**一个 session 只允许一个 in-flight**；agent 层不该用它 —— 订阅 turn 事件即可。
- `history(before_seq)` / `surface_messages()` / `instruction()` 提供读面。
- `chat_off_record`：站在会话上下文上问一句，不 append、不建 session、不落盘。
- `DshSessionRef.to_anchor(at, note)`：区间折点，**明确有损**（缺 seq 就报错，不替你猜）。
- `seed_from_log`：ref → 冷 seed。`TranscriptView`：约定的呈现协议（工具结果、reasoning、机制帧都不进）。
- 配置面：`LlmConfigPatch.applied_to(base)`（缺省即保持，`reasoningEffort="off"` 是真指令）；
  权限是 `sandbox`（read-only / workspace-write / full-access）+ `approval`（ask / never）。

## 观测面

- 节点自己要有 terminal / file-editor 式的面：发给 agent 的高级命令在界面上可见，必要时加 trust 闸口。（他）
- dsh 运行时可视化**不另做**——dsh 自带观测插件足够；独立做价值不大。（他）

## 落地顺序（明天从这里开始）

1. **Hive 实现**：一个 `Memento` + presets + 委派名册；`spawn` 写声明 commit 并起第一个载体。
2. **Session 适配**：把 dsh session 包成 `MementoAgentSession`；`prompt` 映射到 `session/prompt(mode=...)`，
   turn 结束回调接 `on_turn`，`configure` 映射到 `LlmConfigPatch` + 权限面。
3. **reboot**：折 branch → 造 briefing/seed → 起新 session → 写 session-start commit（engine / session id / 生效配置）。
4. **channel 面**（按 `channel_builder` 三层）：
   `instruction` 讲协议；`notice` 给蜂群摘要；**`named_notices` 每个 agent 一片**（只有分片能表达"消失"）；
   commands：`spawn / prompt / cancel / reboot / configure / archive / read / chat`；一步完成发 mindflow signal。

## 待定

1. 重建触发：上下文水位强制 + op 可选？（我推）
2. branch 存放：节点自己的 memento store（我推），还是 ghost ego store 里的兄弟 branch。
3. 冷读归谁：engine 补 `read_session`（我推），还是节点直接开 persistence 文件。
4. 节点退出时 live dsh session：留（branch 可恢复）还是 dispose；重进时如何接管。
5. op 词集合与 level 映射表：先定最小集（continue / stop / need_human / error）还是一步到位。
6. **config 的落点**（他 D09）：运行时直接 set，不建配置文件；只有当"重建时配置要对齐"真的需要时，
   才把 config 显式放进 branch 目录、按目录读。

## Not in scope

- dolores 的正式集成（`dolores-concurrent-mind` 的 `ghoshell-dsh-integration.md` 是另一条线）。
- 独立的 dsh 可视化观测面。
- **树形 fan-out / team 建模**（他 D09 定）：这一期只有 ghost → hive → agents 一层。
  dsh 提供 subagent 就用，不提供不管。
- **`export_anchor(note=)` / `chat(anchor)`**（他 D09）：不建，但不许从地图上消失 —— 前者是"区间折成点并留 note"，
  后者是"按文件里的点提问，不需要活会话"。
- 原始事件层（debug 用，永不上 agent 面）。

## Materials

- **契约（已落盘）**：`src/ghoshell_moss/agents/blueprint/memento_agent.py`（`321818ab`）
- 节点：`nodes/deepseek-harness/{NODE.md,main.py}`（web attach + runtime channel）
- 接缝：`src/ghoshell_moss/deepseek_harness/{channel,runtime,surfaces,session,launcher,trajectory}.py`
- 引擎：`packages/ghoshell-dsh/src/ghoshell_dsh/v0_2_0/`
- 频道构造：`src/ghoshell_moss/core/blueprint/channel_builder.py`（三层：instruction / notice / context）
- 配置惯例：`src/ghoshell_moss/core/speech/speech_module.py`（schema 进 docstring、当前值进 notice）
- 铺垫工作流：`memento-confluences`（已完成：recipients / confluences / children / archive / events）
- 相关：`.ai_partners/features/workstreams/2026/11/dolores-concurrent-mind/ghoshell-dsh-integration.md`
