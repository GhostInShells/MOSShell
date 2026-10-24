---
name: 'ideas'
description: 'thinking governance as a graph — one `graph.json` per group; nodes carry label / body / concept / weight / flag, edges are entities with ids, and the page is one view of the same file'
category: thought
singleton: false
exec:
  command: python
  args: main.py
---

Ideas is the **capture and governance surface for surplus thinking** — and the shared
surface two minds point at when they align. It is a graph, not a mind map: a mind map
answers "what comes next", a graph answers "where is the bottleneck, which branch can be
cut, what contradicts what".

**The graph is the truth.** A group is a directory holding one `graph.json`:

```json
{
  "rev": 18,
  "meta": {"concepts": {"core": "#e8f5ea", "storage": "#58a6ff", "risk": "#d0715c"}},
  "nodes": {
    "n1": {"label": "motive", "body": "", "concept": "core", "weight": 1.0},
    "n10": {"label": "storage", "concept": "storage", "weight": 0.8, "flag": true}
  },
  "edges": [{"id": "e3", "from": "n1", "to": "n10", "kind": "detail"}]
}
```

- `label` is 1–8 characters: what the bubble draws. The id (`n7`) is the identity and
  never changes, so renaming cannot break an edge.
- `concept` is a word from the group's own colour scale in `meta.concepts` — a shared
  vocabulary, extended by whoever is drawing. It is not an enum in code.
- `weight` decays with generation: a child is `parent × 0.8`, floor 0.2. Depth is size,
  so cutting a branch is a decision about weight, not taste.
- `flag` is the human's red mark: *this point has a problem*. The ring outside the bubble
  is dashed red; that is the whole protocol of "I disagree here".
- `kind` is `detail` (grows out of), `depend` (a depends on b) or `related`.

**`ops.jsonl` is the account, not the truth.** One line per change, with its rev. It
exists for `diff` — "what moved since I last looked" — and may be deleted at any time.

**The read side is the whole point.** `tree` is the cold read: the entire structure in
one call. `critical` gives the spine (longest chain + branch points), `path a b` and
`leaves` are the pruning questions, `diff <rev>` is what changed.

**Start / debug**

```
moss nodes run nodes/thought/ideas
python cli.py tree
python cli.py add "storage" --from architecture --concept design
python cli.py set storage --flag
python cli.py link "storage -> md, storage => version control"
```

The root defaults to `ideas/` beside this node, overridden by `--root` or
`MOSS_IDEAS_ROOT` (the agreed handoff: whoever starts the process decides the directory).
`examples/` is copied in only when that directory does not exist yet; `ideas/` is
gitignored wholesale, so what ships is disposable by design.

Port is ephemeral — read the live URL from the channel's `url` notice, or pin it with
`--port N` / `MOSS_IDEAS_PORT`.
