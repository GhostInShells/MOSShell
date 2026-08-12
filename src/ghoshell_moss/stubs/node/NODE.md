---
name: '{name}'
description: ''
singleton: true
# persist: true (default) = 常驻 node cell — provide channel 长期运行,
#   生命周期事件进 ghost 感知.
# persist: false = 一次性 run-to-completion cell — 改变调用方行为模式:
#   nodes:run 会阻塞等它退出并返回 stdout/stderr/exitcode (标准 bash 调用),
#   事件静默 (event_level=DEBUG, 不打扰 ghost 但 mesh:events 可拉取),
#   不 provide channel.
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
