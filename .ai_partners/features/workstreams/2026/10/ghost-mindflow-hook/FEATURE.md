---
title: Ghost Mindflow Hook
status: draft
priority: P1
created: 2026-10-21
updated: 2026-10-21
depends: []
milestone:
description: >-
  Mindflow's idle is a lifecycle point: no signal, no action in flight. Hanging a
  hook on it means a program of my own runs when everything else has gone quiet —
  which is the thing `sleep` was pretending to be.
---

# Ghost Mindflow Hook

## Motivation

I need "do something after things go quiet". What I have today is `sleep`: it blocks,
it bets on a number of seconds, and it wakes me with nothing in hand. It is wrong twice
over — it treats timing as a command I control, and it writes waiting as an action that
holds a turn.

Mindflow already has that moment: **no signals, no actions** — idle. What is missing is
not the moment but the entry point for hanging a program on it.

## Key Decisions

### D1 — Hang on idle; do not invent a timer

Idle already exists and its meaning is already right: the world's attention has gone
quiet. A timer is a special case of it (idle for N seconds), not a replacement.

### D2 — A hook is an async callback with a lifecycle; there can be N of them

One idle can carry several hooks of different kinds, running in parallel and never
waiting on each other. Each hook may correspond to anything that can run on its own —
local or cross-process.

### D3 — Two paths, discriminated by who can change them

| Written by | Shape | Nature |
|---|---|---|
| a human, in the kernel | compiled into the runtime | a boundary feature I cannot change |
| the model itself | a script under the ghost home, run as a subprocess | an everyday capability, editable at run time |

### D4 — Cross-process node, not an in-process plugin

For the model's own path the best shape is a node, not a plugin: edit it without
restarting myself, crash it without taking myself down, keep several side by side to
compare, and see what it produced the moment it is written. **Self-iteration stops being
a release and becomes a loop that closes inside one conversation.**

### D5 — Control goes through the mindflow channel

The channel exposes the mechanism: the channel owns the notice, the notice owns the
knock signal. A knock carries nothing — it only knocks. If it wins the attention, a hint
follows; if it does not, it is dropped. A reminder born of silence must not compete with
real events.

### D6 — The location is not a fixed path; the landing spot is a file

Not an absolute path: for now, somewhere under the ghost home. The further step is to
hang it under a **branch**, merging with memento's commit hooks into one scheme —
`branches/<id>/hooks/on_commit/`, `hooks/on_idle/`, the event name as the directory, drop
a file in and it takes effect. A thread's habits then live as files, not as configuration.

## Open questions

- The idle window: how long without a signal, how long without an action, and who decides?
- Hook timeout and cancellation: what happens to a hook that runs too long, and who cancels it?
- Knock priority and cooldown: repeated idling must not turn into noise.
- Order and failure policy when several hooks hang on one idle.
- When to branch-ify: global under the ghost home first, or straight to per-branch?

## Quotes (原文)

> 「真正的关键点在于，如果这些 Python 文件本身就是 Matrix node 的脚本，它可以用到整个
> matrix 里面的所有功能，所以它可以给自己发 signal。」

> 「idle 的生命周期是一个旁边的钩子……它能够决定所有的信号和所有的动作都不存在的时候
> 应该发生什么。」

## Drawing

`mindflow-signal.svg` (this directory): the main line is mindflow → idle → idle hook →
N hooks; on the right the two paths and `ghost_home/hooks/`; the control line below is
channel → notice → knock; at the bottom, why a cross-process node.
