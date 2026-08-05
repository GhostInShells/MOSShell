---
title: Node Lifecycle — 身份、入口、验证与记忆
status: in-progress
priority: P1
created: 2026-08-04
updated: 2026-08-04
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
