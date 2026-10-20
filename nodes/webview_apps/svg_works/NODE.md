---
name: 'svg_works'
description: 'vector works that stay mutable — a work is a thread with an append-only action list; write/set/remove change nodes in place, checkout replays a prefix. nothing reloads.'
category: webview_apps
singleton: false
exec:
  command: python
  args: main.py
---

A work is a **thread**: an addressable object with its own drawing and its own
action history. The model never re-writes a whole document to change one node —
it sends an **op** addressed by CSS selector:

- `write(name, addr, chunks__)` streams a fragment under that address (live token by
  token, recorded once as a single op with its final text);
- `set_attr(name, addr, attr, value)` changes one attribute — the smallest change
  there is, and the one that makes "画到一半能改" true;
- `remove_node(name, addr)` drops a subtree.

The op list is the truth; the DOM on screen is its projection. `checkout(name, n)`
puts the reader back at the first n ops and the page replays them — replay, not undo,
so the ops after n are still there and moving forward again costs nothing.

Two rules keep replay honest:

- ops must be structural and side-effect free (no timers, no network, no random), or
  replaying twice would not give the same document;
- streaming is live but the record is not the stream.

The layout is the mechanism's shape: the drawing on the left changes in place, the
action column on the right grows downward, one card per op, each card titled by the
model's `note` (the motive) with the mechanics underneath. Clicking a card moves the
reader. Collapsing the cards (fullscreen) hides the list, never the record.

The surface binds an **ephemeral port by default** — read the live URL from this
channel's `url` notice. Pin one with `--port N` or `MOSS_SVG_WORKS_PORT`.

Not a singleton: run several instances, each gets its own port.

The design record lives in `.ai_partners/features/workstreams/2026/10/svg-works/`.
