---
name: 'ghost-in-web'
description: 'shared browsing — a Chrome extension + local WS node for human and ghost to share one real browser, page by page'
category: browsers
singleton: true
exec:
  command: python
  args: main.py
---

ghost-in-web is the general browser body: `bilibili-shared-webview` grown from one
site to the whole web. A Chrome extension (`extension/`) puts an icon + satellites
on every page; a local WS node receives its frames and serves dispatch commands
back. One process, three faces over one store:

- the **channel** — what the ghost sees and can call;
- the **WS server** (23890) — the extension's edge;
- the **audit page** (23891) — what the human reviews.

Identity is inherited verbatim from `ghost-in-bilibili`: `label` (`p1`/`p2`) is the
only persistent page identity, bound to a browser window; `url`/`title` are mutable
attributes.

**Authorization is one bit per page.** Clicking the icon authorizes the ghost to
*read* that page. Everything beyond reading (click / type) is confirmed **on the
page itself** — the node never grants, it mirrors and records. There is no warrant
integration.

**Screenshots cannot be pulled.** The ghost has no screenshot command. The human
clicks the screenshot satellite and one image is pushed as an `aside` signal.

The extension does **not** take effect on local addresses (127.0.0.1, localhost,
private ranges, `file:`), so the audit page — served from localhost — can never be
wrapped by the extension. That is the address-space separation between the audit
face and the controlled face.

Port is **fixed** at 23890 — the extension hardcodes the node URL, so an ephemeral
port would desync. Pin `MOSS_GHOST_IN_WEB_ORIGINS` to the extension id to keep other
pages out of the localhost WS (empty = allow all, dev only).

Workstream: `.ai_partners/features/workstreams/2026/09/ghost-in-web/`.
