---
name: journal
description: Dolores 的时间线日记 — 按年/月/日分区，生命轨迹
pins:
- label: timeline
  verb: exec
  arguments:
    ref: timeline.py
    mode: python
    budget: 6000
  description: 时间线视图 — years/months 存在性，today 展开；缺占位时自动物化
---

# Journal

Dolores 的时间线日记。按年/月/日分目录，每个区间一份文档，逐层提炼：

```
Y{年}/
  yearly.md
  M{月}/
    monthly.md
    D{日}/
      daily.md
```

glob 一下目录就知道人生轨迹；没启动的天就没有 daily（诚实轨迹），缺的区间
由 memento 的时间区间机制回溯补建。

## 文档约定

每份文档的 frontmatter：

- `summary`: 一行摘要，默认空，<100 字。grep 就是接口，不靠 list。
- `status`: `pending`（机制自动创建，尚未撰写）→ `writing`（撰写中）→ `closed`（区间已结束）。

## 呈现

timeline 视图（exec pin）输出三段：years 存在性 / months 存在性 / today 展开。
机制在缺占位文档时自动从 `template.md` 建（只建不写，内容归你自己写）。

## 记录纪律

- **逐层不跳级**：daily → monthly → yearly，每层只读直属下一层。
- **值得记的东西直接写进当天的 daily**；monthly / yearly 是提炼，不是流水账。
- 写工具归 agent 自带（dsh）+ MOSS warrant —— 机制只建占位，不混入写协议。
