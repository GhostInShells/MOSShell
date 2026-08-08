---
name: 'counter_service'
description: 'V1 validation: provides counter/echo service via matrix operator'
singleton: true
exec:
  command: python
  args: main.py
---
Counter/echo disposable test service.
Provides two queryable keys: inc (returns incrementing count), echo (returns payload unchanged).
Used by the matrix-operator V1 validation plan.
