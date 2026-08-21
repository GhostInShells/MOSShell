---
name: 'warrant_auth'
description: 'Warrant startup-authorization demo node — asks a human to approve launch via matrix.warrant'
singleton: true
exec:
  command: python
  args: main.py
---

Warrant 启动授权演示 node。启动时 `matrix.warrant.require(StartupPermission)`
发出审批问题（QA namespace `_warrant`），由 `moss nodes answer-node --namespace
_warrant` 或其它 watcher 应答。批准 → 提供 `ping` channel 常驻；拒绝 → 退出。

CTML 调用示例:

```
warrant_auth:ping
```
