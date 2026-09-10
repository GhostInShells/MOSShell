---
title: Groundset Declaration
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-10
updated: 2026-09-10
depends: [ground-channel, ghost-ground]
milestone:
description: >-
  在根场 GROUND.md 用 `groundset:` 字段声明要展开的子场 (相对路径目录); GroundSet 物化子场并持有独立
  root(); ground channel 把子场挂成子 channel (notice=帧), 根场默认不渲染. Ground 实例带 ULID id, 使
  channel runtime 身份稳定. 目的: ghost 改 GROUND.md 即可重整认知视图, 重启生效, 不动源码.
---

# Groundset Declaration

> Use `moss features set-status groundset-declaration <status> -m "note"` to update state.

## Motivation

dolores ghost 要把 ground channel 挂进自己的运行时, 用可编辑的 `GROUND.md` 决定展开哪些子场。
现状做不到, 有三处缺口:

1. **子场无法声明式展开。** SPEC §7.2 明确"子场是同位场, 不自动打开", 只能靠 `frontmatter` pin 做
   渐进披露。ghost 想让某组子场进来, 只能自己 `open()` 一堆 —— 与"ghost 自治"（改文件而非改代码）冲突。
2. **Ground 实例没有稳定身份。** `_build_child` 每次 `new_channel` 都生成随机 uid
   (`py_channel.py:50`), 而 runtime 用 `channel.id()` 做注册键 (`runtime/tree.py:400`)。一旦 channel
   被重建, runtime 会判 removed+added, notice 全量重发。
3. **根场与子场没有区分。** 根场如果走 `open()`, 就只是"又一个被打开的场", 会被挂成子 channel,
   无法"只作为根"存在。

动机的本质: **ghost 修改自己的认知视图时, 改 GROUND.md 的关键字段即可, 不用改源码, 也不用每次启动
手动 open 一堆场。只在重启后生效。**

## Key Decisions

### K1 — Ground.id = ULID, 实例化时生成

`Ground` ABC 加 `id` property; `DefaultGround.__init__` 里 `str(ULID())`。用于把 Ground 身份透传给
channel runtime (`new_channel(uid=ground.id)`), 使 runtime 注册键稳定。不用 hash（内容会变, 不构成身份;
且项目 ULID 约定已统一: `anchor/contract.py:39`, `memento/abcd.py:44`）。

### K2 — 协议字段 `groundset:`, 值为相对路径列表

根场 `GROUND.md` frontmatter 新增保留键 `groundset`（`contract.py: GroundConvention.groundset`）:

```yaml
groundset:
  - existence
  - people
  - skills
```

- 值为**相对路径**, 相对**根场目录**（$GROUND）解析; 目录必须自带 `GROUND.md`, 否则显式报错不静默跳过。
- **不递归**: 只展开一层。子场自己的 `groundset` 字段**不**被消费 —— 机制属于 GroundSet, 不属于 Ground
  （用户决策: "GroundSet 实例化时有这个机制, 但 ground 没有"）。
- 未声明该字段 = 现有行为（子场不自动打开）, 协议向后兼容。

### K3 — GroundSet 持 root(), 独立于 open()

`DefaultGroundSet.root()` 返回**锚点场**（= `workspace_root` 指向的目录）, 与 `open()` 打开的子场分离:

- root 不被挂成子 channel（否则它只是"一个被打开的场"）;
- root 是 `groundset` 字段的声明来源, 物化子场的入口;
- root 进入 grounds 注册表（`active()` 含 root, 生命周期统一 sediment）; 消费方按身份
  把它从"被挂载的子场"里排除。

### K4 — 物化由 GroundSet 构造 flag 开关

`DefaultGroundSet(..., materialize=True)`。一次性 GroundSet（CLI `moss ground render/meta`、
channel 的 peek 路径）传 `False`, 于是"CLI 也顺带展开子场"的副作用不存在。

### K5 — ground channel: render_root 默认关 + instruction 可重写

- `render_root: bool = False` —— 根场帧默认不进任何 surface。开启时进 ground channel 自己的 **notice**
  （不是子 channel, 根场不是子 channel）。
- `instruction: str | None` —— 覆盖 channel instruction（默认 = 机制 prose）。

### K6 — 子场信息落 notice, 根场上行到 ghost instruction（dolores）

dolores 的 compact = epoch 机制: 压缩后所有 channel facade 在 epoch 起点重供
(`shell_trajectory.py:617-626`)。所以 facade/notice 跨 compact 不丢, 唯一问题是**时序** ——
epoch 帧插在**每步历史最前**（`moss-dolores-ghost-plugin.ts:404,617-624`), 而 memory 在 session 创建时
注入（`:501`), 故 facade 永远晚于 memory。

结论: **根场 render 要走 ghost 级 instruction（dolores `system_prompt()`), 不能靠 channel 的任何面**
（channel instruction 也走 facade, 同样晚）。子场是可变更提示, 落 notice、随 epoch 重供即可。
`render_root=False` 正是为此。

纪律（非机制）: 根场 pins 应是身份性的（如 dolores 根的 `frontmatter */GROUND.md`); 易变的 `exec`/`file`
放子场 —— instruction 是启动冻结的, 易变内容冻进去会陈旧。

## Implementation Notes

- 字段值相对 **根场目录** 解析, 不是 workspace_root（二者在 channel 场景同值, 但语义以 $GROUND 为准）。
- 物化在 `GroundSet.root()` 首次调用时触发（async）; channel 的 startup await 它。
- 自引用保护: `groundset` 指到根场自身 → 跳过。
- 子场 open 失败不应拖垮根场物化（best-effort + 可见日志）。
