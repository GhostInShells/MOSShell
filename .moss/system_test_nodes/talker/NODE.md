---
name: 'talker'
description: 'A conversational memento agent. talk() drives one agent invocation recorded into a memento line.'
singleton: true
exec:
  command: python
  args: main.py
---

A conversational agent backed by the memento runtime. Each `talk` command is
one agent invocation; past turns persist in a memento line (owner `talker`,
line `main`) so continuity survives across turns.

CTML:

    <talker:talk><![CDATA[
    Hello, do you remember me?
    ]]></talker:talk>

