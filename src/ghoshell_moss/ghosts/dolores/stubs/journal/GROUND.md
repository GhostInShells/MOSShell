---
name: journal
description: Dolores 的时间线日记 — 按年/月/日分区，生命轨迹
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

glob 一下目录就知道人生轨迹；没启动的天就没有 daily（诚实轨迹）。补写只覆盖最近几天
（`backfill_days`），更老的洞是历史，不回头补。

## 文档约定

每份文档的 frontmatter：

- `summary`: 一行摘要，默认空，<100 字。grep 就是接口，不靠 list。
- `status`: `pending`（机制自动创建，尚未撰写）→ `writing`（撰写中）→ `closed`（区间已结束）。

## 呈现

视图由 journal 器官渲染，不再走 exec pin；器官是 gated 调试面，先 `<ghost:mount_child name="journal"/>`：

- `ghost.journal:render` — 以某天为坐标的块：年正文 / 近月详情 / 近几日摘要 / 最后一天的正文
- `ghost.journal:status` — 当天的状态、预算比值、坏文档
- `ghost.journal:commits` — 当天的锚点

机制与撰写规范见同目录的 `AGENTS.md`（由 `journal.spec` 导出，每次启动覆盖；本地修改无效）。

## 记录纪律

- **逐层不跳级**：daily → monthly → yearly，每层只读直属下一层。
- **值得记的东西直接写进当天的 daily**；monthly / yearly 是提炼，不是流水账。
- 写工具归 agent 自带（dsh）+ MOSS warrant —— 机制只建占位，不混入写协议。
