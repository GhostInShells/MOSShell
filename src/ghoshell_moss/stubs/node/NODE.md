---
name: '{name}'
description: ''
singleton: true
exec:
  command: python
  args: main.py
---

Write this node's capability self-explanation here, for the Ghost that will run
it — the same plane as a skill. What the node is for, the CTML calls that drive
it, the boundaries the Ghost must respect. Capability only: keep internal
decisions and pointers into this node's implementation out of it.

The Ghost finds this node in its shell and opens it through its channel; the
channel mounts automatically. Working on the node means working on its channel —
read the channel builder and Matrix, and see Matrix and Matrix channel.
