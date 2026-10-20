---
name: 'discuss'
description: 'the conversation as branches over time — fork() opens a direction in one sentence, note() adds a line, state() says whether it landed, was dropped, or is still open'
category: webview_apps
singleton: true
exec:
  command: python
  args: main.py
---

A long conversation forks constantly, and the forks live in the human's head: one
sentence can throw out three directions, most of them never get discussed, and nothing
records that they were ever raised. This node records three small facts — `fork`,
`note`, `state` — and draws them.

What it records is an **index, not content**: one sentence per fact, enough for the
model to find the detail in the conversation again. The detail stays in the
conversation.

- `topic(name)` — name the session; several can coexist.
- `fork(text)` — a direction leaves the main line here.
- `note(id, text)` — something on that direction is worth remembering.
- `state(id, open|landed|dropped)` — where it stands. `landed` means the conclusion is
  somewhere findable (feature, journal, code, file); `dropped` is a decision, kept so
  that "we raised it and did not do it" stays visible.
- `list()` / `export(path)` / `load(path)` — the record as `.jsonl`, the minutes as
  `.md` **generated from the record** rather than written by hand.

The page is an ECG-style timeline: the main line runs left to right, every fork drops
to its own lane, `landed` curves back to the main line, `dropped` ends in a cross, and
`open` runs dashed to "now". The window holds the most recent 30 minutes or 40
records — whichever ends first — because a long conversation is otherwise kilometres
wide. New records follow; dragging back stops the follow, because yanking the view
back is the same rudeness as interrupting.

The human does not operate this surface. They watch it, and click a point to read that
one line. Every recording action belongs to the model — releasing the human is the
whole point, so a mechanism that needs them to record has already failed.

The channel notice carries one line (`open 3 (b1,b4) · landed 2`), and only changes
when it actually changes: notices ride the facade delta, so an unchanged line costs
nothing. When the record falls behind the conversation it adds `stale 18m` — the
evidence is the timestamp gap, not the model's memory.

Exports land in `sessions/` next to the node (gitignored).
