---
name: 'probe_timeout'
description: 'system test node — check hangs, but declares a timeout, to verify probe timeout enforcement'
singleton: true
exec:
  command: python
  args: main.py
check:
  command: python
  args: -c "import time; time.sleep(3600)"
  timeout: 2
---

System test node. The `check` probe sleeps for an hour but declares
`timeout: 2`, so it must be terminated after 2s and reported as a broken
reason ("probe timed out after 2s") instead of hanging the spawn path.
