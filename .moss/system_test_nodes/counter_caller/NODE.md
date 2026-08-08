---
name: 'counter_caller'
description: 'V1 validation: discovers and calls counter service via matrix operator'
singleton: false
exec:
  command: python
  args: main.py
---
Counter caller — discovers the counter service, calls inc + echo, prints results.
V1 validation: if this node prints results, the operator surface works.
