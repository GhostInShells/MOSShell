---
title: Node Lifecycle — 身份、入口、验证与记忆
status: in-progress
priority: P1
created: 2026-08-04
updated: 2026-08-07
depends: []
milestone: 0.1.0
description: >-
  Node 生命周期四层治理：持久身份锚 (UUID)、认知入口重整 (open/read)、
  启动前闸口 (probe)、Ghost 侧记忆 (IoContract)。从 node-migration 讨论
  中独立出来的 nodes 体系优化。
---

# Node Lifecycle

> 人类架构师 + claude-opus-4-7。四个节点生命周期优化点从 node-migration 讨论中
> 生长出来，共享一个共同的根问题：长上下文下节点知识会丢失，模型重复犯错。

## Motivation

`.installed` marker 只回答"装没装过"，不回答"环境现在能不能用"。
node 启动后空转，失败只在 stderr 和 bounded FIFO 里，不进模型上下文。
长上下文下缺席很快被冲淡，模型反复试同一个坏 node。

根本问题是节点的就绪状态没有进入管理。需要一个覆盖全生命周期的治理链：
**身份 → 入口 → 验证 → 记忆**。

## Design Index

- 讨论背景：node-migration FEATURE.md（mac_control / playwright / screen_capture 迁移）
- 讨论记录：本会话中的多轮碰撞（probe 形态、文件级通讯、UUID 身份、ghost 记忆归属）
- 相关 features：
  - `matrix-cell-governance`（completed）— CellRuntimeInfo ledger、单写者纪律
  - `channel-meta-dyn-static`（design-locked）— 动静分离的 channel meta
  - `moss-project-ground`（in-progress）— GROUND.md / skills 认知场

## Key Decisions

### 1. 持久身份锚：`.node_id` UUID

- **生成**：框架层在第一次 open/run 时检查 `{cell_home}/.node_id`，不存在则生成写入。
  人去填 UUID 是反模式——纯框架生成，零声明负担。
- **身份脱离路径**：UUID 非耦合文件路径、NODE.md name、category。目录重命名/换分类/搬机器——身份不丢。
- **回退**：没有 `.node_id` 的老 node 在首次 open 时自动种一个，原地升级。
- **复制语义**：clone node home → UUID 复刻 → ghost 记忆自动继承，这是正确行为（"同一类 node"的经验继承）。
  若 clone 意味着"新东西"，删 `.node_id` 触发重新生成。

### 2. 启动前闸口：probe

- **形态**：NODE.md 声明的可选 `check: {command, args}`，或约定 `check.py`。独立进程，
  语言无关，目标零配合。exit 0 → 通过，nonzero + stderr → 失败原因。
- **输出即自描述**：probe 的 stdout 捕获后作为动态自描述/help hint，与 instruction（静态）
  并列构成 open 的返回。
- **gate 语义**：probe 失败 → 不拉起主脚本，返回 broken reason。主脚本拉起后依靠既有的
  `process alive + ledger providing` 兜底，不加新字段。
- **优于 on-bootstrap 的理由**：probe 验证的是"这 node 能不能跑"——import 真依赖、甚至
  smoke 调用；on-bootstrap 只证明"python 走到了 Matrix.__aenter__"，差一个量级。
  且 probe 不强加合作契约给目标脚本。

### 3. 认知入口重整：open vs read

- **open（主路径）**：返回 instruction（NODE.md body，认知入口）+ 活状态（ledger 当前态
  + probe 输出 + dead_cells 历史）。模型用 open 了解 node 即可开始使用。
- **read（debug 路径）**：frontmatter 全字段 + exec 详情 + runtime 文件位置 + 安装路径。
  给模型排障用，不是主认知入口。

### 4. Ghost 记忆：IoC 合约可选注入

- **合约**：`NodesMemoryContract`（最少三个方法：`store` / `load` / `forget`，按 node_uuid + key 键控）。
- **浮现**：`build_nodes_channel` 从容器 `container.get(NodesMemoryContract)`，存在则注册
  `remember` / `recall` / `forget` 命令；不存在则零暴露。零协商、零污染。
- **存储**：ghost 自己的领地（`ghost_home/memory/nodes/`），不污染 cell_home。
- **伴随浮现**：node open 时 `recall` 命中即展示，与 node 开闭伴随。

## Implementation Notes

- **启动通讯走文件级 ledger**：CellRuntimeInfo 文件已编码 spawned → started → ready → exited
  全状态机。probe 不需要 ledger（独立进程读输出）；主脚本 readiness = `cell.providing` 含 `'channel'`。
  mesh announce 偏重——同一个 liveness 位，文件级读是本地操作，不走 presence→adapter→zenoh 多层。
- **open 的返回结构**：instruction（static）+ probe 输出（dynamic）+ ledger 态 + ghost 记忆命中。
  动静分离的 discipline 继承自 `channel-meta-dyn-static`。
- **与 node-migration 的关系**：本 workstream 的产出（UUID、probe 声明字段、open/read 语义）
  在迁移结束后直接用于所有 node；迁移本身的"git mv + NODE.md 转换"不依赖这些。

## 调研增补 (2026-08-06)

> 人类架构师 + ds-v4-pro。围绕"记账+入网是否值得砍、Script 要不要回归、启动性能"
> 做了实证测量与机制溯源。结论收敛为: 启动成本不在入网机制, 矩阵核心不用改;
> 真正要盯的是 mesh accept 的 provider 上线感知不被淹没。

### 启动成本实测

| 路径 | 平均 | 组成 |
|---|---|---|
| `moss nodes run` (CLI spawn + 记账+入网) | ~2764ms | CLI 自身 ~1.4s + node 侧 ~1.35s |
| `python main.py` 直接 (记账+入网) | ~1353ms | |
| 纯 python 基线 | ~11ms | |
| 纯 `zenoh.open` 隔离 | ~505ms | Session 协议地板 |

1.35s 分解: `import ghoshell_moss` ≈ 0.84s + `zenoh.open` ≈ 0.5s + 入网机制 ≈ 15ms
(hub+liveness+announce+ledger 全部 ~5-15ms)。

### 决策 5: 不砍 zenoh

- matrix ≈ zenoh, Session 绑 zenoh。不用 zenoh 就不该用 matrix。0.5s 是 Session 协议的
  地板, 不优化。
- 非懒加载的 O(N) 观察 (hub liveness listener, 经 adapter 强制起, 共享给 presence/mesh)
  实测 ~5ms; 记账 (ledger 写) ~0ms。都不是成本。

### 决策 6: 修 `images.py` 的模块级 anthropic import

`import ghoshell_moss` 0.84s 的大头不是包导出面, 而是传递依赖链:
`core.concepts.command → message → message.contents.images` 里 `try: from anthropic.types import Base64ImageSourceParam`。
`import anthropic.types` 单独 = 0.69s。修掉后 `import ghoshell_moss` 应降到几十 ms。

- `Base64ImageSourceParam` 是 dict 的 TypedDict 子类, 调用返回普通 dict, 与 `dict`
  运行时逐字节等价。替换为 `Base64ImageSourceParam = dict` 或 `total=False` 本地 TypedDict 即可。
- BaseModel 每实例 `__init__` 成本不变 (schema 类定义时一次性构建, 与注解 import 来源无关)。
- 注意: 当前行为因环境不一致 — anthropic 装了严格校验 source 三 key, 没装则宽松。
  代码用法是宽松的 (`source.get("media_type")`), 统一为宽松 (total=False / dict)。
- 次要清扫 (不在普遍热路径): `ghosts/atom/_meta.py`、`agents/memento_pydantic_agent/factory.py`
  的模块级 anthropic import。

### 决策 7: Script 不回归, 事件分级原语已存在

- 记账与入网是两个正交轴 (§UU 文件真相 vs 网络真相)。channel accept 需要入网
  (presence + mesh 观察者), 不入网不触发 accept, 成立。
- "publish event 被淹没"来源是 mesh 事件订阅, 已惰性 (opt-in by usage)。
  worker 只入网+provide channel 不调 `network()` 就不会被淹没。
- 拉取日志原语已存在: `mesh.recent_events(limit)` / `mesh.cell_events(address, limit)`,
  `CellEvent.refetch` 二元。事件级别/两事件面可选, 非必需。
- Script 不回归。若未来要避免短命进程的网格 churn 再议, 但启动成本不是理由。

### 决策 8: 真正要盯的是 mesh accept 的 provider 上线感知

- `channels/matrix_channel.py` 的 mesh channel: `accept`/`reject`/`set_auto_accept`/`events`。
- accept 治理的是**通用资源信任**, 不只 channel (`mesh.accept(address)` 建 proxy)。
- 待验证: 当 cell provider 上线 (liveness PUT + presence announce + CellEvent "channel added"),
  mesh 侧 accept 感知链路不重不丢、不被淹没。

## 调研增补 (2026-08-07) — 事件分级与 MatrixOperator 方向

> 人类架构师 + ds-v4-pro。task 6 验证了 mesh accept 感知链路, 并和另一会话的
> MatrixOperator 草稿 (`core/blueprint/matrix_operator.py`) 对撞, 收敛出分级方向。

### 验证结论 (task 6)

- **CellEventNucleus 已注册默认 mode**: `moss manifests nuclei` 确认
  `cell_event_nucleus` (cell_event signal), 声明在 `.moss/modes/default/src/HOST/nuclei/__init__.py`。
- **链路**: cell provider 上线 → publish CellEvent → mesh events_wildcard → event_queue
  (maxsize 10000) → `_event_consumer_loop` (每事件写 buffer + refetch 拉 presence + 建/撤 proxy)
  → `_fire_on_event` → mesh channel `_dispatch_event` → send_signal → CellEventNucleus
  → background_notice impulse → ghost articulate 循环。
- **三个缺口**:
  a. transition 硬编码 READY (`matrix_channel.py:469`) — CRASHED/EXITED 无法区分。
  b. CellEventNucleus 单槽覆盖 (`cell_event_nucleus.py:129`) — 一次 articulate 周期
     多个事件 peek 只见最后。
  c. mesh 事件队列满丢 (`zenoh_mesh.py:534`) — maxsize 10000, 极端洪峰才丢。

### 决策 9: 事件分级 + MatrixOperator 方向

- **MatrixOperator 草稿** (`core/blueprint/matrix_operator.py`): `CellServerMeta{address, protocol}` /
  `CellServer` / `CellClient` / `MatrixOperator` (serve/client/get_servers/get_protocols/
  on_server_start/on_server_stop)。人类架构师另一会话推进, 本会话仅对撞判断。
- **关键判断**: "提供了某种资源" **不属 transition 枚举** — 是 `protocol` 维度
  (`CellServerMeta.protocol`), 与 cell 生命周期正交。枚举收敛: 不加 RESOURCE_ADDED。
- **event 改造成非公开函数, 只暴露 cell 生命周期**。复杂协议 (pub/sub/queryable)
  不再由 matrix 原生提供, 走 CellServer/CellClient。协议流量不产生 ghost signal。
- **分级面收敛到生命周期事件**:

  | 事件 | 进历史 | 优先级 | primitive |
  |---|---|---|---|
  | `on_server_start(protocol)` (新资源上线) | 是 | NOTICE(1) | notify |
  | cell CRASH | 是 + 注意 | WARNING(2) | notify |
  | `on_server_stop(protocol)` (资源下线) | 否 | BACKGROUND(-1) | background_notice |
  | cell EXITED | 否 | BACKGROUND(-1) | background_notice |

- **cell event 通道优先级上限 WARNING**; 更高需求 (急停/强制) 走 direct signal
  (interrupt/FATAL 反射弧), 不走 publish event。
- **拉面** (ghost 主动拉取): `mesh.events` / `cell_events` 加 transition 过滤, 低/高两通道。
- **与 MatrixOperator 的关系**: 分级那步 (加 transition + priority override) 是
  MatrixOperator 的先行件, 方向一致不浪费; MatrixOperator 是更大重构 (channel duplex/
  topic/session 变协议层), 待另一会话定熟语义再动。

### Pending (本会话提出, 未设计)

- **project ioc 拆分**: Matrix 下注册的 MOSS manifests 实为 Project manifests。
  应为 Project 注册全局 IoC + 自己的 logger, Matrix 只补默认依赖。
  涉及 `matrix_impl._prepare_container` 装配次序。属 matrix/project 治理, 可另开 workstream。
- **Matrix.discover 单例承诺是假的**: docstring 说进程级单例, 实现每次新建。
  Host.discover 目前坏 (`factory._create_host` raise NotImplementedError)。
- **depend_* 与 find_spec**: find_spec 谓词应集中 `depends.py` (`available(module)`),
  硬门与显示判断分层; ghost/agent 模块级 `from pydantic_ai import` 需 lazy 化。
