---
name: '{name}'
description: ''
singleton: true
# command: 'python' is a convention — resolved to the spawner's current
# executable (sys.executable). Use it when this node runs in the same
# environment as MOSS. Only write an absolute interpreter path when the
# node needs its own venv / runtime.
exec:
  command: python
  args: main.py
---

Body describes what this node does — the model reads this when the channel
is accepted. Include capability summary + CTML invocation examples.
