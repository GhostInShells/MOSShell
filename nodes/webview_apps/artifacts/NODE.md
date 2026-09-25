---
name: 'artifacts'
description: 'streamed visual artifacts — the model streams an artifact source into a live page: announced, accumulated visibly, finalized by kind (canvas / mermaid / markdown / image / hls / term / html / style)'
category: webview_apps
singleton: false
exec:
  command: python
  args: main.py
---

Artifacts is a streamed, code-driven visual surface. A model calls
`write(kind, label, chunks__)`; the tag body streams token by token into the page —
the first token announces the artifact, later tokens accumulate its source in a
source pane, and the end finalizes it into a rendered result beside the source.
`kind` only decides the finalize step, so one lifecycle serves every visual: canvas
(JavaScript), mermaid, markdown, image/MJPEG, hls, term, html, style (CSS). Each
command carries a `duration` budget — a positive value holds the artifact on screen
after it finalizes, so a sequence paces itself without `sleep` primitives.

One port serves both the page (`/`) and the WebSocket (`/ws`) via `websockets`.
The page is `index.html`; open the URL it prints and watch artifacts appear live.
`read(label)` / `history(n)` / `display(label)` form the retrospective control
plane — label is the handle. Human clicks on the surface (source toggle, tab
switch) flow back as a tail log, surfaced in the channel's `notice` with timestamps.

The surface binds an **ephemeral port by default** — read its live URL from this
channel's `url` notice, never assume a fixed port. To pin one, start with
`--port N` (or set `MOSS_ARTIFACTS_PORT`).

Not a singleton: run several instances — each binds its own ephemeral port by
default, or pass `--port N` to pin each to a distinct fixed port. Two instances
on the same fixed port will not start (the bind fails).

Note: the canvas kind eval's the model-authored JavaScript in the page
(`new Function(...)`) by design — the code IS the artifact. Local, model-authored.
