---
name: branch ground
description: this line's standing state — what it is for, where it stands, what it is waiting on
budget: 1200
---

# Branch ground

This directory is one line's **state face**. The line's timeline is its commits and notes;
this file is what the line *is* right now. Update it when the line's question changes, when a
decision lands, or when its state moves — not on every commit.

Keep four things, briefly:

- **What this line is for** — a paragraph: the direction, not a task list.
- **Where it stands** — done, decided, open.
- **What it is waiting on** — physical time, an external reply, a meeting, a dependency. The
  world does not speed up for you; naming the wait keeps a long line from reading as a stalled
  one.
- **Next** — the next concrete move, if there is one.

A direction worth keeping grows as a sub-field: a directory with its own `GROUND.md` (e.g.
`skills/`, `memory/`), listed under `imports`. Do not build a mount this line does not use.

References are relative to this file. Keep the body short: it is rendered every time the line
enters memory.
