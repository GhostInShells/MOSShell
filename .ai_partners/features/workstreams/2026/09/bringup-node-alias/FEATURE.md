---
created: 2026-09-21
depends:
- cell-run-cycle
- node-lifecycle
description: '把 nodes:run(target, name) 的挂载名能力拓展到 MODE.md 的 bringup_nodes 声明: 配置里可以给一个
  node 声明 alias, 它提供 channel 后稳定挂在 matrix.mesh.<alias>, 不再回落到每次 spawn 都变的 uid 短标;
  裸路径写法保持不变.'
milestone: 0.1.0
priority: P1
status: completed
status_note: 'bringup alias 落地: MODE.md bringup_nodes 支持 {target, alias} (裸路径向前兼容),
  alias registry 上移到 matrix.cell_aliases 供 CTML nodes:run 与 bringup 共用; 测试覆盖配置解析 +
  reserve->mesh 挂载同账; 未做活 mode 端到端验证.'
title: Bringup Node Alias
updated: '2026-09-21'
---

# Bringup Node Alias

> Use `moss features set-status bringup-node-alias <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`matrix.mesh.<name>` 的挂载名机制已由 `cell-run-cycle` §6 落地: 模型用 CTML
`nodes:run(target, name)` 拉起 node 时自产一个名字, mount 时 consume, 得到稳定路径.
但 host cell 不是 node 的唯一生产者 — mode 的 `bringup_nodes` 在 host 启动时直接
`matrix.run_node(target)`, 那条路径没有名字可 reserve, mount 时回落到
`CellAddressCodec.short`（`名字_uid末6位`）。后果:

- 同一个 node 每次启动换名字, transcript 里的路径抄一次废一次;
- 名字里带 cell 原始 name, `deepseek-harness` 这类带连字符的 node 产出的短标不是合法
  channel 名（`ChannelNamePattern` 不允许 `-`）, 挂载名对模型不可用。

人类架构师判决: 把 alias 能力拓展到 MODE.md 的 `bringup_nodes` 配置里, 保持向前兼容。

## Design Index

- Key design documents: 无独立 design/；决策上下文见 `cell-run-cycle/matrix-channel.md` §6
- Key discussion records: 无

## Key Decisions

1. **挂载名归 matrix 治理, 不再归 channel 层持有。** 原实现把 `CellAliasRegistry`
   放在 `channels/matrix_channel.py` 的闭包里（"Matrix facade untouched"）。现在有两个
   生产者（CTML nodes:run + mode bringup）, 两者必须共用同一本计数账, 所以 registry 上移:
   类定义到 `core/blueprint/cell.py`, 实例由 `MatrixImpl` 持有, facade 以
   `cell_aliases()` 暴露（与 `handled_cells()` / `dead_cells()` 同层的治理访问器）。
   nodes/mesh channel 改为从 `matrix.cell_aliases()` 取账, 不再各自 new。

2. **配置形态: 裸字符串 = 只声明 target（向前兼容）；mapping 才能声明 alias。**
   `bringup_nodes` 的元素类型收成 `BringupNode{target, alias}`, 用一个
   `field_validator(mode='before')` 把裸字符串规范化成 `{target: <str>}`。
   老的 `bringup_nodes: ['nodes/os/terminal']` 语义一字不变（alias 为空 → 回落 uid 短标）。
   alias 在解析期就用 `ChannelNamePattern` 校验, 避免带着一个永远挂不上的名字启动。

3. **清理时机从"mesh 每次 refresh prune"改成"cell 退出即 discard"。**
   原实现在 mesh 的 `_refresh` 里用 `handled_cells` 做差集剪枝；registry 移到 matrix 后,
   `_on_cell_exit` 回调（进程退出即触发, 不依赖 mesh 是否在观察）直接 `discard(address)`,
   单一机制, mesh 侧不再需要 `handled_cells`。计数 counter 不变, 仍单调不复用。

## Implementation Notes

- 改动面: `core/blueprint/cell.py`（registry 类）、`core/blueprint/matrix.py`（facade
  抽象方法）、`matrix/matrix_impl.py`（实例 + exit discard）、
  `channels/matrix_channel.py`（取账 + 删本地类, 保留 `CellAliasRegistry` re-export）、
  `core/blueprint/project.py`（BringupNode + validator）、`host/moss_runtime.py`
  (`_bringup_one` reserve)、`nodes/deepseek-harness/NODE.md`（host 侧寻址说明）。
- **已知竞态（沿用 CTML 路径既有语义）**: reserve 发生在 `run_node` 返回之后。子进程要
  走完 boot → 入网 → provide channel → host accept → mesh refresh 才会 mount, 实际远慢于
  两条同步语句, 但理论上"mount 先于 reserve"时该 cell 会挂到 uid 短标, pending 条目留在
  账上直到进程退出被 discard。彻底消除需要把名字传进 `run_node`（地址在 spawn 内生成）,
  本轮不动。
- **未做**: `nodes:status` / notice 不展示 alias（mount 时已 consume, 真相在 mesh 的
  `proxy_aliases`）；CLI `moss nodes run` 起的 node 仍然没有 alias, 需要跨进程传递名字才
  能覆盖, 不在本 workstream 范围。