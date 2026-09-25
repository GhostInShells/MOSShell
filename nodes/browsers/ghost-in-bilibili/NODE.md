---
name: 'ghost-in-bilibili'
description: 'bilibili shared-watching — a Chrome extension + local WS node for human and ghost to watch video together'
category: browsers
singleton: true
exec:
  command: python
  args: main.py
---

ghost-in-bilibili is a bilibili shared-watching body: a Chrome extension (`extension/`)
injects a ghost ball + authorization satellites into video pages, and a local WS node
receives its frames and serves dispatch commands back. The ghost perceives the page,
reads subtitles, and — once authorized — controls playback.

One WS per browser session: the extension's service worker holds it, every tab of that
session multiplexes over it. Identity: `label` (p1/p2) is the only persistent page
identity, bound to a window; `bvid` is a mutable attribute (bilibili auto-plays).

Port is **fixed** at 23880 — the extension hardcodes the node URL, so an ephemeral port
would desync. Pin `MOSS_GHOST_IN_BILIBILI_ORIGINS` to the extension id to keep other
pages out of the localhost WS (empty = allow all, dev only).

Design: `.ai_partners/features/workstreams/2026/09/bilibili-shared-webview/`.
