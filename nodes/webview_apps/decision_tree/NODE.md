---
name: 'decision_tree'
description: 'a decision-tree outline for guided creation — the graph is the plan, not the picture; nodes are state machines the human and model move together'
category: webview_apps
singleton: false
exec:
  command: python
  args: main.py
---

Decision tree is a creation outline, not a picture. The graph comes first: the
human and the model discuss a tree of decisions, each node carries its own state
machine (open → discussing → researching → decided), and a decision is a move on
that machine — issued by the model, confirmed by the human. The graph only grows
through `create_node`; the only other primitive is changing a node's state.

Three storage tiers, three lifecycles (see the feature's FEATURE.md):

- `DECISION_TREE_ROOT.md` — the graph's meta: statuses table, edge vocabulary, plan
- `tree.jsonl` — nodes + state, an append-only log; the channel is its only writer
- `graph-nodes/{name}/` — a node's content, created lazily only when it has material

The web surface is the human's view: the tree on the left (ECharts), the detail
on the right. Click a node to see its state and directory listing; click a file
to open it in the OS. Action cards — create / update — carry a confirm button;
confirming flows back to the model as a signal.

The surface binds an ephemeral port by default — read its URL from the channel's
`url` notice, never assume a fixed port. To pin one, start with `--port N` (or
set `MOSS_DECISION_TREE_PORT`).
