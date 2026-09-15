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
over a file, streams mutations (`write` / `str_replace` / `insert` / `rewind`)
that humans watch in real time; humans reply by editing the action's
description or its effect, and the reply is delivered as a unified diff.

The interaction is a `Thread` bound to a text file: both sides append actions,
mutations apply in FIFO order, and each confirmed mutation appends a `Version`
to a linear, append-only chain. Rejecting an action cascades to later actions on
the same thread; `rewind` is an ordinary action whose result equals an older
version's content. `reference` is the one side-effect-free action (display a
file region for shared view).

Three independent axes: (1) pure data structures + a durable append-only store
(`src/ghoshell_file_editor/`), (2) the communication protocol (uplink
interactions / downlink streaming / queries), (3) the web UI. Axis 1 is landed
and unit-tested; axes 2–3 build on it.
