---
name: 'qa_pusher'
description: 'QA exchange — issues 3 varied questions and awaits answers'
singleton: true
exec:
  command: python
  args: main.py
---

QA Pusher demo node. Issues one question of each kind (confirm, input, choose)
to the public QA namespace via ZenohQAManager (from IoC). Awaits each answer
and logs the result. Exits after all three are resolved.

Run together with moss-ghost or moss-shell to answer the questions via TUI.
