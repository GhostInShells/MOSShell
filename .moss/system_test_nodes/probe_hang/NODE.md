---
name: 'probe_hang'
description: 'system test node — check never exits, to verify bringup does not block startup'
singleton: true
exec:
  command: python
  args: main.py
check:
  command: python
  args: -c "import time; time.sleep(3600)"
---

System test node. The `check` probe sleeps for an hour, so it never exits —
it exists to prove a hanging probe does not block mode startup, and does not
starve sibling bringup nodes. The node body is the default empty template
(it never launches, because the probe never passes).
