---
name: 'ideas'
description: 'thinking governance — an append-only event log over one or more `ideas/` directories, projected onto a d3-force graph where ideas surface over time, sized by priority and ringed by state'
category: thought
singleton: false
exec:
  command: python
  args: main.py
---

Ideas is the **capture and governance surface for surplus thinking**. It is not a
webview app that happens to draw circles: the log is the mechanism, the page is one
way to read it.

**The log is the truth.** Every idea lives as events in `<root>/ideas/ideas.jsonl`,
one line each, append-only: `born` / `state` / `priority` / `tag` / `untag` / `link` /
`unlink` / `note`. The current state of an idea is a *fold* of its events — nothing is
stored as a snapshot, so "how did this idea get here" is always answerable. Each idea
also has a readable `<id>.md` beside the log: the human face of the same record.

**Directories are the shards.** A root is any directory that carries an `ideas/`
subdirectory. `open(root)` mounts one; several can be mounted at once — one directory
is one cluster, and clusters can be linked. There is no database, and there is no size
problem that a new directory cannot solve.

**The page folds too.** The surface is handed the raw event list and folds it itself,
so the timeline is an exact prefix of the record: dragging it back is "step earlier",
not "reload an old snapshot". The force layout is kept alive across timeline moves —
a drag grows the picture, it never reshuffles it.

States: `spark` → `active` → `waiting` → `done` / `retired`. Size is priority; the ring
carries the state. Relations start with two words only: `related`, `depend`.

Port: ephemeral by default — read the live URL from this channel's `url` notice. Pin
one with `--port N` or `MOSS_IDEAS_PORT`.
