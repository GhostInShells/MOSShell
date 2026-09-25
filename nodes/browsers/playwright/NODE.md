---
name: 'playwright'
description: 'Playwright browser — open/close live browser runtimes via a module-runtime hub (exec/aexec/history)'
category: browsers
singleton: true
exec:
  command: .venv/bin/python
  args: main.py
---

Playwright browser control. The node provides a hub channel (`playwright`)
rooted at `domains/`; opening a domain materializes a live browser runtime the
model drives by writing Python. Heavy dependency (playwright) → own venv.
