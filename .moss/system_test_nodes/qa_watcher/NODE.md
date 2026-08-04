---
name: 'qa_watcher'
description: 'QA exchange — watcher node, auto-answers approval questions'
singleton: true
exec:
  command: python
  args: main.py
---

QA Watcher demo node. Watches for approval questions and auto-approves them.
Uses ZenohQAManager from IoC.

Run together with qa_asker to verify cross-process QA exchange.
