---
name: 'file_editor'
description: 'a shared working copy of some text — every action becomes a card the human watches; export is the only step that touches the disk'
category: os
singleton: true
exec:
  command: python
  args: main.py
---

You and the human share one working copy per thread. Every move you make becomes
a **card** on their surface; `export()` is the only step that touches the disk,
so it is the only step that asks.

```
<file_editor:open thread="readme" path="./README.md" label="README"/>
<file_editor:find thread="readme" pattern="^## "/>
<file_editor:read thread="readme" region="12-40" label="the usage section"/>
<file_editor:edit thread="readme" region="12-20" label="rewrite usage"><![CDATA[## Usage

Run `moss --ai start`.]]></file_editor:edit>
<file_editor:export thread="readme"/>
```

A thread is an editable object: it can come from a file, from a working copy a
previous session left behind (`open(draft=...)`; `orphans()` lists them, `discard`
throws one away), or from nothing at all. Threads outlive their exports — writing
records a *position* on the line (`exported_at`), which is why you can keep
editing and export again.

**Finding your way in.** `find()` answers with line numbers, never with text.
Line numbers are the coordinate everything else takes: `read(region=)` returns
those lines and leaves a card showing the same thing, `edit(region=)` replaces
them without asking you to quote them back, and `str_replace` stays for the case
where the snippet is what you actually know.

**A big file is a fact, not a paragraph.** Past a couple of thousand lines or a
quarter megabyte, `read()` with no region returns the file's shape — size, lines,
endings — and the three moves above; a region read is capped. Holding a large
copy is cheap, reading it is not.

**Every card has two readers.** The mechanics are the node's: which region, how
many lines, `+2 −1` for an edit, where an export lands. The motive is yours: the
`label` on `read` / `find` / `edit` / `rewind` / `export`. A card that says only
"read all" cannot tell the human whether it is worth opening; one that shows only
a motive hides what happened.

**The only gate is `export()`.** It returns a receipt at once; the human decides
on the surface and you learn the outcome as a signal. A thread the human has
auto-trusted writes to its own file without asking — a new target still asks.
`close()` ends editing: an unsaved copy is kept as a recoverable draft, a clean
one is dropped.

**Presentation is split down the middle**, same contract as the terminal node:
which card is open is their click, how big it opens is yours (`layout()`), and
the page names who chose.

Boundaries: text only; paths are confined to the project home and the system temp
dir, and a path outside them is refused rather than questioned.
