---
name: dsh_v0_2_0
description: dsh 0.2.0 node — own a dsh web process, show its page as a window, and drive the agents inside it
category: dsh_swarm
singleton: true
exec:
  command: python
  args: main.py --port 23080
check:
  command: python
  args: check.py
---

A dsh node. It runs its own dsh web process, shows that page as a window on the screen, and
holds the agent sessions you drive through it.

## The channel

- `status` — home, profile, port, running state, and the page URL.
- `web_url` — the page URL, including the per-launch token.
- `agents` — a child channel holding the agents. It stays **closed** until you open it with
  the parent channel's `mount_child`; then its own commands become available. Opening it is a
  deliberate act, so the node stays small until something is actually being driven.

## Driving an agent

Every agent is addressed by a name you choose.

1. **Declare a place first.** A workspace is a directory inside the node's home, and the only
   place an agent may act in. `create_workspace(path, name)` declares one — nothing is
   predefined, and a path outside the node's home is refused.
2. **Open.** `open(name, workspace, preset, readonly)` creates one. `preset` selects the
   composition (empty takes the deployment's default); `readonly` makes it read-only from
   creation, before its first message.
3. **Talk.** `send(name, text)` is fire-and-forget: it answers `ok` or `failed`, and the answer
   arrives later as a `dsh` message signal carrying the agent's name, its session and a
   timestamp — a callback, not something you said.
4. **End.** `cancel` stops the turn in flight; `retire` ends the membership. `state` and
   `presets` are the cold reads; the notice carries only the roster.

Sessions are runtime-only. After the node restarts, `watch(name, session_id)` is the only way
back to one.

## Running it

Started through the matrix node channel, on port 23080 — append `-- --port N` to move it.
Launch arguments: `--home DIR`, `--profile NAME`, `--title NAME`. Install first — see
`INSTALL.md`.
