---
name: 'file_editor'
description: 'a dialogue thread over a readable text file — both sides append actions, mutations apply in FIFO order, diffs flow back as replies'
category: os
singleton: true
exec:
  command: python
  args: main.py
---

File editor is a shared dialogue surface over one readable text file. It is
**not** an editing tool — the dialogue is the purpose, and export to disk is the
only real side effect (its dialogue is the approval). The model opens a thread
over a file, streams full-text `write` proposals that humans watch in real time,
and may `rewind` / `reference` / `export`; humans confirm, reject, or reply by
editing the action's description or its effect, and the reply is delivered as a
unified diff.

The interaction is a `Thread` bound to a text file, and the actions appended to
it *are* the append-only log. Every action carries the effect it would produce,
computed when it is appended — so a diff is readable before anyone decides
anything, and confirming computes nothing, it only records a verdict. The head
and the version list are derived from those verdicts rather than stored.

Rejecting an action cascades to every later action on the same thread (`rewind`
is how an already-confirmed step is undone, so the log stays append-only).
`rewind` is an ordinary action whose payload points at an earlier action or the
loaded baseline; `reference` is the one action with no effect (display a file
region for shared view); `export` carries the text it would write and is the
only real side effect.

Three independent axes: (1) pure data structures + a durable append-only store
(`src/ghoshell_file_editor/`), (2) the communication protocol (the model-facing
channel + the human web surface), (3) the web UI. Axes 1–2 are landed and
unit-tested; axis 3 (`index.html`) builds on them.
