# file_editor

A human-observable working copy of a text file. The model opens a thread, reads
and edits it, and every action becomes a card on a web surface the human watches.
The only step that asks for approval is `export` — writing to disk.

## What it does

One process, two faces over one store (`src/ghoshell_file_editor/`):

- **channel** — the model's side: `open` / `read` / `write` / `append` /
  `str_replace` / `rewind` / `export` / `close` / `threads` / `history`.
- **surface** — the human's side, a card stream at `http://127.0.0.1:8767`.
  Cards carry the action kind, the thread, and a state; clicking one opens the
  three tabs: effect (markdown), full, history.

Durability is a mirror, not a log: every edit rewrites the thread's draft under
`runtime/drafts`, so a crash loses the history but not the text. `export` is the
final chapter — it ends the thread and releases its payloads. A thread the human auto-trusts exports to its own file without asking; paths are confined to the project home and the system temp dir.

## Setup

No install steps — the node shares the MOSS environment. It needs the `host`
extra for `websockets`.

## Usage

```bash
moss nodes run nodes/os/file_editor          # foreground, Ctrl+C stops
python main.py --port 9000                   # debug directly
```

Open the URL the node prints to watch cards appear.

## Development

- `src/ghoshell_file_editor/structure.py` — pure data + functions (axis 1)
- `src/ghoshell_file_editor/store.py` — threads, drafts, export verdicts
- `src/ghoshell_file_editor/channel.py` — the model-facing commands
- `src/ghoshell_file_editor/surface.py` — the web surface + signals
- `index.html` — the card stream and the three tabs

Run the tests from the repo root:

```bash
cd nodes/os/file_editor && ../../../.venv/bin/python -m pytest tests -q
```
