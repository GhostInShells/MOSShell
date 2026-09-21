# decision_tree

A decision-tree outline for guided creation. The graph is the plan, not the
picture: a visualized, layered state machine the human and the model shape
together. A decision is a move on that machine — the model issues it, the human
confirms it.

## What it does

One process, two faces over one store (`src/ghoshell_decision_tree/`):

- **channel** — the model's side: `create_tree` / `open_tree` / `trees` /
  `create_node` / `link_node` / `update_node` / `read` / `focus` / `history`.
  Each mutation appends one event to `tree.jsonl`; the current tree is a fold.
- **surface** — the human's side: the tree on the left, the detail pane on the
  right, on an ephemeral port (reported in the channel's `url` notice). Click a
  node to see its state + directory listing; confirm action cards flow back to
  the model as a signal.

Three storage tiers, three lifecycles:

- `DECISION_TREE_ROOT.md` — the graph's meta (statuses + edge vocabulary + plan)
- `tree.jsonl` — nodes + state, an append-only log; the channel is its only writer
- `graph-nodes/{name}/` — a node's content, created lazily only when it has material

## Setup

No install steps — the node shares the MOSS environment. It needs the `host`
extra for `websockets`.

## Usage

```bash
moss nodes run nodes/webview_apps/decision_tree   # foreground, Ctrl+C stops
python main.py --port 9000                        # debug directly
```

Open the URL the node prints to see the tree and the detail pane.

## Development

- `src/ghoshell_decision_tree/meta.py` — pydantic models (tree meta, node state, seeds)
- `src/ghoshell_decision_tree/store.py` — the fold over `tree.jsonl` + path boundary
- `src/ghoshell_decision_tree/channel.py` — the model-facing commands
- `src/ghoshell_decision_tree/surface.py` — the web surface + frames
- `index.html` — ECharts tree + detail pane

Run the tests from the repo root:

```bash
cd nodes/webview_apps/decision_tree && ../../../.venv/bin/python -m pytest tests -q
```
