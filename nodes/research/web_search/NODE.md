---
name: 'web_search'
description: 'research surface — search tasks as nodes, sources as evidence inside them, typed edges as the reproducible path; the ghost points the view, the human reads and clicks back (prototype, mock data)'
category: research
singleton: false
exec:
  command: python
  args: main.py
---

Run: `moss nodes run nodes/research/web_search`, then open the URL from the
`url` notice. The page also opens straight from disk; only the WS uplink needs
the node.

This node is the **product end state prototype** for making a ghost's search
visible. The page is a shared research surface: a d3 task graph on the left, the
selected task's motivation / actions / sources / artifacts on the right, a source
reader, and a replayable timeline. The graph data is mock data embedded in
`index.html` — no search is executed yet. The goal is to settle the shape before
wiring a real search agent.

**Three granularities, one node identity.**

- **L1 · query / fetch** — one API call and its return, plus follow-up detail
  reads. They live inside a task as `actions`, not as graph nodes.
- **L2 · search task** — a motivation, a research question, a batch of
  actions, and a result exit (`artifacts`). This is the unit that appears on the
  graph; a task may contain sub-tasks (double-click a node to collapse/expand).
- **L3 · task graph** — typed edges between tasks (`expands` / `supports` /
  `derives` / `compares` / `supersedes`), each carrying a `basis` that says why
  the association exists. The path is the reusable knowledge topology.

**The rule that keeps the graph readable:** only L2 goes on the graph. L0/L1 are
node internals; a detail query becomes its own node only when it gains an
independent motivation.

**The channel is a view controller, not a search executor.** `focus(task_id)`
points the human's graph, `replay(step)` scrubs the timeline, `view(name)`
switches the side panel. Human clicks come back in `notice`. The surface binds an
**ephemeral port by default** — read the URL from this channel's `url` notice,
never assume a fixed port. To pin one, pass `--port N` or set
`MOSS_SEARCH_SURFACE_PORT`.

**Next step (not built here):** a functional dsh search agent preset whose
session events feed `search/task.*`, `search/action.*`, `search/source.*`,
`search/link.*` into an append-only log; this page becomes the projection over
that log. Task boundaries and edges must be emitted explicitly (task-frame
tools), not inferred from tool-call order.
