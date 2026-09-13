---
name: 'clause_subscriber'
description: 'clause topic drainer — subscribes to the clause topic and prints each ClauseTopic as one JSON line'
singleton: true
exec:
  command: python
  args: main.py
---

Clause topic drainer probe. Subscribes to `matrix.session.topics` on the `clause`
topic (跨进程经 zenoh), 阻塞 poll 每条 ClauseTopic 并打印一行 `clause #n: {json}`.
No channel — works standalone, no Ghost needed.

它验证说侧装线: `moss_runtime._clause_topic_bridge` 把 speech 单例的 `on_clause`
结果广播成 ClauseTopic. 也对称覆盖听侧 (ASR 定稿 clause 时同 topic).

## 测试方法 (recorded)

1. 起本 node (后台或前台均可):
       moss nodes run .moss/system_test_nodes/clause_subscriber/

2. 另开一进程跑 `moss-shell`, 在 TUI 里说多句话 (多句, 触发多个 clause).

3. 退出 moss-shell.

4. 数本 node 打印的 `clause #n:` 行数 — 应等于说出的句数 (标点分句). 每条带
   `role=ghost` / `speaker_name` / `text` / `meta.created_at`.

Run in the same network scope as the speaker (default scope from .moss).
