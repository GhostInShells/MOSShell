---
name: 'contracts_dump'
description: 'Debug node: dump all bound contracts and providers from matrix.container on start'
singleton: true
exec:
  command: python
  args: main.py
---

Script node — runs once, prints the full IoC container state, then exits.
No channel; purely a debugging tool.

On start, iterates `matrix.container.contracts()` (all bound contract → instance)
and `matrix.container.providers()` (all registered providers with their contract bindings).

### Usage

```
moss nodes run .moss/system_test_nodes/contracts_dump
```
