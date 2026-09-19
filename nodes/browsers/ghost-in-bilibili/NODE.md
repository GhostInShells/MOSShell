---
name: 'ghost-in-bilibili'
description: 'bilibili shared-watching component — a Chrome extension + local HTTP node for human and ghost to watch video together'
category: browsers
singleton: true
exec:
  command: python
  args: main.py
---

ghost-in-bilibili is a bilibili-specific shared-watching component: a Chrome extension
(`extension/`) injects a draggable ghost ball into video pages, and a local HTTP node
receives its events and serves dispatch commands back. The ghost perceives the page,
pulls subtitles, and — once authorized — controls playback and runs reviewed JS.

The node writes each payload to a temp file, reads it back, deletes it, and logs it —
the "dump to file, node reads, delete" boundary. The browser never displays content.

Port is **fixed** (uncommon): the extension hardcodes the node URL in its manifest
and background script, so an ephemeral port would desync.

