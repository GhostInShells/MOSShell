---
name: existence
description: Dolores 的自我存在感 — 我是谁、我在时间轴上是什么
pins:
- label: identity
  verb: file
  arguments:
    path: identity.md
    budget: 2000
  description: 事实自我 — 从时间线提炼的存在投影（预算 2k）
- label: timeline
  verb: exec
  arguments:
    ref: timeline.py
    mode: python
    budget: 6000
  description: 记忆时间线视图 — 今天(全文)/最近n天(摘要)/最近n月(摘要)
---

# Existence

Dolores 的自我存在感：**我经历过什么、我在时间轴上是什么**。

@purpose.md
@behaviors.md

## 机制

- **时间线**：`memory/` 下 daily → monthly 逐层提炼。当前数据不多，暂轻建。
- **事实自我**：`identity.md` 从时间线滚动提炼，每月末 update（`file` pin，预算 2k）。
- **意义**：`purpose.md` 从 identity + 锚点定位（`@` 冷层，常驻）。
- **行为风格**：`behaviors.md` 每条关联子文档，模型可自改（`@` 冷层，常驻）。
- **时间线视图**：`timeline.py`（exec pin, mode=python）输出
  今天(全文) | 最近 n 天(摘要) | 最近 n 月(摘要)，倒序。排序是脚本策略。

## 可写思维提示

这里欢迎你（或你的化身）用**自己的写工具**落内容——ground 不混入写协议，
只提示"写成什么形状、什么周期"：

- **daily** `memory/daily/YYYY-MM-DD.md`：frontmatter 惯例 `description`
  （一行摘要，被 timeline 视图消费）。anchors（不可压缩锚点）放 body，
  提炼时原样上浮，是压缩中的守恒量。
- **monthly** `memory/monthly/YYYY-MM.md`：从 daily 提炼，逐层压缩不跳级。
- **identity.md**：每月末从时间线重炼，旧版在 git 历史里，放心改。
- **purpose.md**：从 identity + 锚点重审；锚点集变了才重写。

写工具归 agent 自带（dsh）+ MOSS warrant，不在此 ground 内。
