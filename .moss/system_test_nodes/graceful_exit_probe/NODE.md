---
name: 'graceful_exit_probe'
description: 'Regression probe: double-SIGINT during teardown must exit gracefully with no ERROR.'
singleton: true
exec:
  command: python
  args: main.py
---

A one-shot reproduction harness for the cell-graceful-exit contract.

It exposes a single `graceful_exit_probe:ping` command so the channel is live
before teardown. A background thread then fires two SIGINTs: the first cancels
the run loop, the second lands while `Matrix.arun` is still joining the draining
main coroutine.

Verification is by exit code + cell log — run it and confirm:

- the process exits 0,
- `moss.node__graceful_exit_probe.log` (or `moss.log` when run directly) gains no
  `ERROR` entry from `project.py:845`.

Do not leave this node running as a service; it terminates itself.
