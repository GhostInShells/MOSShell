---
summary: memento 长出跨 branch 的两条边；agent 契约被推倒三次，最后落在异步原语上
status: closed
---

一晚上六笔提交。

**memento**：`recipients`（出）+ `confluences`（入）+ `children`（子分支）；两个 append-only 缓冲在 commit 时被原子认领、去重、吸收。归档改成递归、先子后父、逐条事件；容错补齐 —— 坏行隔离进 `*.rejected.jsonl` 并记进 `CommitRef.rejected`，timeline 坏行原位留占位，绝不让分支卡到要手工修文件。

**被连纠三次**：把"收侧先不要上"读成"整个不做"，只发了发信侧；给发信侧塞了恒为 0 的 `commit_seq`；拿 memento 事件当投递触发器。查 dsh 才发现 `session/prompt` 原生就有 `queue/steer` —— `post` 的语义根本不用发明。

**凌晨**重写 `agents/blueprint/memento_agent.py`：异步是原语（`prompt` 只回执），没有 close 只有 reboot（起 session 写一条 commit 当定位点）。第一版实现放到 `dsh-node` 里，不另开 workstream。
