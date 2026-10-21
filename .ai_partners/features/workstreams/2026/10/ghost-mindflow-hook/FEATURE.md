---
title: Ghost Mindflow Hook
status: in-progress
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

## Landed — the minimum slice: `idle-knock`

The first piece of this design is in the kernel, and it is deliberately smaller than a
hook system: one command on the mindflow channel, no registry, no persistence.

`idle-knock(seconds, text__)` arms a callback on the coming idle window and returns at
once — nothing blocks, no action stays open. When mindflow goes idle the callback runs:
it waits `seconds`, then sends a `knock` signal whose message is `text__`, the line the
ghost wrote to itself. It fires once and disposes itself; if real signals arrive first,
the window closes, `_stop_idling` cancels the callback, and nothing is sent.

**No kernel change was needed.** `when_idle` already had exactly this lifecycle:
callbacks are turned into tasks when the window opens and cancelled when it closes
(`_mindflow.py`). The slice is one command plus its tests.

What it buys: the ghost can wait for something without betting on a number and without
holding the turn. `sleep` is what it replaces — the failure mode it removes is "waited
for a thing that never came, so never moved again".

### The test that was wrong

The first version of the test armed the command and then called `_set_idle()` by hand to
simulate the window. It failed, and the failure was interesting enough to record: it
proved that a callback registered while the window is already open never runs, which led
to a patch in `when_idle` to cover it.

That patch was redundant and was reverted. A channel command executes inside an action,
so mindflow is never idle while the command that arms the callback is running — the state
the test manufactured does not exist. Worse, the wrong case leaked into the rest of the
directory and made a shell-integration test fail.

> 「你加的那个改动是冗余的，因为 mindflow 它就是全程……channel 在运行的时候 mindflow
> 就一定不是 idle，只是你的测试方式，它的逻辑不对。」

The lesson generalizes past this test: **a case that cannot happen in the real path is not
a case, it is a bug in the test.** Write the order the runtime actually takes — leave the
window first, then arm — or the test will demand a kernel change the kernel does not need.

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
