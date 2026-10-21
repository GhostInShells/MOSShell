---
title: Memento Hooks
status: draft
priority: P2
created: 2026-10-21
updated: 2026-10-21
depends: []
milestone:
description: >-
  Memento's lifecycle events (commit, branch) can trigger behaviour and produce side
  effects. The trigger is not a framework but a file: drop an executable in the right
  directory and it runs; take it away and it does not.
---

# Memento Hooks

## Motivation

Memento already has a timeline: branches, commits, the raw conversation of each span, and a
directory of its own for every anchor. But its lifecycle serves exactly one purpose today —
writing the note. Any other "do something at this moment" need can only be met by the model
remembering to do it inside the conversation: archiving a to-do list, moving working files
into the anchor, rendering a field for this thread.

"Because I remembered" has been shown to be false (the D20–D21 retrospective: prompt-level
rules fail, mechanisms hold). Forgetting is not occasional — it is the necessary outcome of
talking, judging and writing in the same turn.

What is wanted: memento's lifecycle should trigger behaviour and produce side effects, and
the triggering itself should not become a burden.

## Key Decisions

### D1 — The trigger is a memento lifecycle event, not polling

The moments worth hanging on are `commit` (an anchor lands) and then `branch` (a line opens).
Not a timer, not "whenever the model remembers" — the event already carries the meaning that
this stretch is over.

### D2 — Three mounting points, discriminated by who can change them

The same thing can hang in three places. The difference is not capability but *who may edit it*:

| Mount | What it is | Self-iterable at runtime |
|---|---|---|
| 1. Kernel | after a commit, scan these directories and run what is there — this sentence only | no |
| 2. One thread | hooks and `GROUND.md` inside `branches/<id>/` | yes |
| 3. The whole ghost | default hooks and templates under the ego root | yes |

Anything I must be able to rewrite while running, and see take effect immediately, belongs in
the files of 2 or 3. The kernel keeps only the mechanism that goes looking.

### D3 — No harness modelling

A hook is a file, not a registration API. No plugin registry, no DSL, no manifest.
Its one condition for taking effect is that it exists in that directory.

The reason is the same as D2's: a registration API requires the caller to remember to
register — back to the prompt layer. A file is declarative; the commit walks past it and
sees it.

> thirdgerb: 「这种 memento hook 应该是跟 git commit 的那些 hook 一样。」

That is the shape: the name of the file says when it runs, `chmod` turns it on, deleting it
turns it off.

### D4 — Rendering is the same mechanism, not a second one

`branches/` and `commits/` are separate directories precisely so each can carry its own
`GROUND.md`: the branch one answers "what is this thread now", the commit one answers "what was
it then". One set of ground templates, used in two places — the present and the past.

Do not split this into two concepts. Ground rendering is part of the same mechanism as the
hook, and the two are tightly coupled. The clearest case is a to-do list: the working plan we
agreed on lives in the branch directory, and the ground reads it out directly. The hook decides
what a commit leaves behind; the field decides what a reader sees — same files, same directory,
one mechanism.

### D5 — A side effect must never hold up the commit

Hooks are a bypass: the commit landing is the main path, the hook runs beside it. A failure
must not roll back the commit and must not block it. Reuse the shape memento's bypass already
has — a running state, failures left on record, no automatic retry, content still readable.

## Key quotes

- 「这种更通用的机制让这些生命周期能够触发一些行为、产生一些副作用。但是触发行为和产生副作用的方式不做那种强建模。」（D21）
- 「这种 memento hook 应该是跟 git commit 的那些 hook 一样。」（D21）
- 「ground 实际上是属于 memento 渲染的时候要用的机制……两者应该是强挂钩的。」（D21）

## Open questions

- **Hook input**: commit coordinate, branch directory, raw span, or just a directory to read?
- **Timing**: before the commit (able to alter it) or after (it only sees the result)?
- **Order and verdicts**: several hooks in one directory — in what order; if one fails, do the
  rest still run?
- **Form**: executable file with a shebang, or may `*.py` be run by an interpreter?
- **Portability**: Windows has no shebang; what does the strategy degrade into?
- **Moving working files**: are `todo.md` / `plan.md` moved by a hook or by the kernel? Today
  it is the former, but it is the single most common case and may deserve to be in the kernel.

## Drawing

`sidecar-map` (an svg works piece; ops in `works/sidecar-map.jsonl`): left is the branch
directory as a working area, right is the commit directory as a snapshot, the arrow between
them is a hook moving things across, and the bottom row names the three mounting points.
