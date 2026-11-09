---
name: dsh_v0_2_0
description: dsh 0.2.0 node — own a dsh web process and show its page as a window
category: dsh_swarm
singleton: true
exec:
  command: python
  args: main.py --port 23080
check:
  command: python
  args: check.py
---

A dsh 0.2.0 node. The cell owns a dsh web process, announces its page to the screen as the
`dsh 0.2.0` window, and keeps serving until the cell is closed. The delegation surface
(Hive) is not wired yet.

It is started through the matrix node channel; the node's own channel exposes environment
reads only:

- `status` — home, profile, port, running state, and the dsh web URL.
- `web_url` — the dsh web URL, including the per-launch token.

The port is fixed in the launch arguments (`23080`; the dsh node family takes 23080+);
append `-- --port N` to override it. Other launch arguments: `--home DIR`, `--profile
NAME`, `--title NAME`.

Install first — see `INSTALL.md`.
