---
name: 'file_editor'
description: 'a shared working copy of some text — every action becomes a card the human watches; export is the only step that touches the disk'
category: os
singleton: true
exec:
  command: python
  args: main.py
---

File editor is a shared working copy of a text file. You and the human look at
the same text: every move you make becomes a **card** on the human's surface, and
nothing asks permission except `export()`, because writing to disk is the only
real side effect.

The web surface binds an **ephemeral port by default** — read its live URL from
this channel's `url` notice, never assume a fixed port. To pin one, start with
`--port N` (or set `MOSS_FILE_EDITOR_PORT`).

```
<file_editor:open thread="readme" path="./README.md" label="README"/>
<file_editor:read thread="readme"/>
<file_editor:append thread="readme" label="usage"><![CDATA[## Usage

Run `moss --ai start`.]]></file_editor:append>
<file_editor:str_replace thread="readme"><![CDATA[{"old_str": "moss start", "new_str": "moss --ai start"}]]></file_editor:str_replace>
<file_editor:export thread="readme"/>
```

**A thread is an editable object.** `open()` starts one from a file, from an
unclaimed working copy left by an earlier session (`open(draft=...)`, listed in
the notice), or blank — an editable object does not have to come from a
document. One path carries at most one open thread; `label` is the name the
human sees.

**Reading is shared perception.** `read()` returns the text to you at once and
leaves a card showing the same text — the human sees exactly what you saw.
`region="10-40"` reads only those lines. Edits (`write`, `append`,
`str_replace`, `rewind`) land the moment the tag closes; nothing waits on a
person.

**Every card has two readers.** The mechanics are the node's to fill in —
which region, how many lines, `+2 −1` for an edit, where an export lands. The
motive is only ever the model's: the `label` on `read` / `rewind` / `export`
(the edits already took one). Both ride the card, because a card that says only
"read all" leaves the human unable to tell whether it is worth opening, and a
card that shows only a motive hides what actually happened. Leave the label out
and the title falls back to the mechanics — nothing goes blank.

**The only gate is `export()`.** It returns a receipt immediately; the human
decides on the surface, and you learn the outcome as a signal. Accepting writes
the file — and the thread stays open, because the write is a *position* on the
line, not the end of it: keep editing the same working copy and export again.
What landed is always readable as `exported_at`, and every thread that has moved
past it says so. One exception to the asking: a thread the human has
*auto-trusted* exports to its own file without asking — but only back to the file
it was opened from; writing anywhere new is a new target and still asks.

**Versions are positions, not snapshots.** `history(thread)` is the index — one
row per action with its verdict; use it to pick `n` for `rewind(n)`, which puts
the text back where action `n` left it (`0` = the baseline) by appending an
ordinary action, never erasing. The current version of each thread is always in
the notice.

**Presentation is split down the middle.** Which card is open is the human's
click. How big it opens is mine: `layout()` reads the mode back, `layout("page")`
gives the open card the whole window, `layout("split")` keeps it a side panel
beside the list. The page carries a marker naming who chose, so a card opened at
the wrong size never reads as an accident. Same contract as the terminal node.

Boundaries worth knowing: the working copy is text-only, and only a live thread
takes actions — after `export` it is frozen, and a thread opened with no file
has no path to export to unless you pass one. Paths are confined to the project
home and the system temp dir; a file outside those roots is refused, not
questioned.
