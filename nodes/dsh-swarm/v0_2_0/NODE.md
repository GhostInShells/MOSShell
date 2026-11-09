---
name: dsh_0_2_0_rc_2
description: dsh 0.2.0 node — own a dsh web process, show its page as a window, and drive the agents inside it
category: hive
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
- `workspaces` — the places an agent may be created in: name, id, and the directory each one
  bounds. `create_workspace(path, name)` declares a new one; nothing is predefined, and a path
  outside the project the node belongs to is refused.
- `agents` — a child channel holding the agents. It stays **closed** until you open it with
  the parent channel's `mount_child`; then its own commands become available. Opening it is a
  deliberate act, so the node stays small until something is actually being driven.
- `ui` — the page itself, also a closed child. It runs a piece of code inside the page's own
  cordis context, so `ctx.get('layout')` is the live dsh service: layout, sidebar, panels and
  focus are driven through dsh's own API rather than by touching the DOM. `describe(name)` reads a
  service's methods first, so the surface explains itself from the running page.

  One constraint worth knowing: a page only polls while its window is in the foreground. A
  backgrounded window cannot be driven, and `attached` is how you find that out.

## Driving an agent

Every agent is addressed by a name you choose.

1. **Declare a place first.** A workspace is a directory inside the project the node serves — the
   place an agent may act in. It is declared on the node's own channel, not on this one — ask
   `workspaces` for what exists, `create_workspace` for a new one.
2. **Open.** `open(name, workspace, preset, readonly)` creates one. `preset` selects the
   composition (empty takes the deployment's default); `readonly` makes it read-only from
   creation, before its first message.
3. **Talk.** `send(name, text)` is fire-and-forget: it answers `ok` or `failed`, and the answer
   arrives later as one message tagged with the hive's name, carrying the agent's name and a
   timestamp — the turn's final answer, and nothing in between.
4. **End.** `cancel` stops the turn in flight; `retire` ends the membership. `state` and
   `presets` are the cold reads; the per-agent roster rides on named notices, and the notice line
   carries the count.

Sessions are runtime-only. After the node restarts, `watch(name, session_id)` is the only way
back to one.

## Running it

Started through the matrix node channel, on port 23080; pass `extra_args="--port N"` there to move
it.
Launch arguments: `--home DIR`, `--profile NAME`, `--title NAME`. Install first — see
`INSTALL.md`.
