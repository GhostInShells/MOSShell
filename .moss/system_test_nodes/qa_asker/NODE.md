---
name: 'qa_asker'
description: 'QA exchange — asker node, issues approval questions periodically'
singleton: true
exec:
  command: python
  args: main.py
---

QA Asker demo node. Issues an approval question every 5 seconds via
ZenohQAManager (provided by IoC). Logs answers when they arrive.

Run together with qa_watcher to verify cross-process QA exchange.
