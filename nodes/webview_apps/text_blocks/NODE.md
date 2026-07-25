---
name: 'text_blocks'
description: 'shared text-block carrier — model streams numbered blocks, humans edit in browser, unified diffs flow back'
singleton: true
exec:
  command: python
  args: main.py
---

Text Blocks is the first dedicated shared-carrier surface: a human-model
collaborative text workspace. The model streams numbered blocks via CTML
(`chunks__`), rendered in real time with a blinking cursor. The human opens
the URL in a browser, edits blocks in-place, and submits changes — unified
diffs flow back as signals.

The surface is a shared coordinate system: block ids link the model's
generation history to the human's editing actions. `dump()` exports to the
filesystem; `read_file()` bridges local files onto the shared surface.

Not a document editor. Not a GUI tool. It's a third category — a carrier
shared natively by both sides, where each side's actions fall directly into
the other's perception.

Window = URL. Does not depend on screen-node or matrix-resources.
