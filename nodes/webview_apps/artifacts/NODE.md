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

A card is also a file. `export(label, path)` writes `<label>.artifact.md` — frontmatter
for the machine fields (`label` / `kind` / `description` / `created_at`) plus any key
this node does not know, then the source verbatim as the body. That is the discoverable
protocol: an ordinary markdown file with a title block, readable by anything, no node
required. `import_file(path)` reads one back. Frontmatter keys the node does not
recognise are kept and written back, so a later version can add metadata (a deck's
narration line, say) without breaking files already on disk.

Cards land in this node's `artifacts/` directory (gitignored) by default; `--dir` or
`MOSS_ARTIFACTS_DIR` points it elsewhere, for a session that wants a working folder of
its own. Memory stays the store — the directory is not watched and nothing is written
back implicitly, so export and import are the only doors.

One export is one card. A collection of cards is a deck: order, transitions and
narration belong to whatever reads the directory, not to this node.

Not a singleton: run several instances — each binds its own ephemeral port by
default, or pass `--port N` to pin each to a distinct fixed port. Two instances
on the same fixed port will not start (the bind fails).

Note: the canvas kind eval's the model-authored JavaScript in the page
(`new Function(...)`) by design — the code IS the artifact. Local, model-authored.
