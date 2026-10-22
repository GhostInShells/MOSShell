---
name: 'ideas'
description: 'thinking governance — ideas are plain `*.idea.md` files grouped by first-level directories; frontmatter carries state / priority / tags / links, and the page is one view of those files'
category: thought
singleton: false
exec:
  command: python
  args: main.py
---

Ideas is the **capture and governance surface for surplus thinking**. It is not a
webview app that happens to draw circles: the files are the mechanism, the page is one
way to read them, and `cli.py` is the one implementation that edits them.

**The file is the truth.** One idea is one `*.idea.md`: frontmatter carries `state` /
`priority` / `tags` / `links`, the body carries the thinking. Editing a file by hand is
a normal move, not a workaround. There is no log, no fold and no replay — the timeline
was a mechanism the problem never asked for. If a trace is ever wanted it is a
disposable file beside the data, never a second truth.

**A group is a directory.** The root holds first-level subdirectories; each one is a
group, its `README.md` is that group's own context, and its `*.idea.md` files are the
ideas. The root defaults to `ideas/` beside this node and is overridden by `--root` or
`MOSS_IDEAS_ROOT` — the env var is the agreed handoff, so whoever starts a process
decides which directory that run governs.

**The data root is disposable.** What ships is `examples/`; it is copied into `ideas/`
only when that directory does not exist yet, and `ideas/` is gitignored wholesale.
Delete it and the next run starts empty — deleting the examples is a supported move, so
seeding never happens over an existing root.

**The name is the identity, and it is semantic.** `duplex-gap.idea.md` says what it is
before anything is opened; an opaque id would force you to open a hundred files to find
one. Renaming therefore rewrites references — `rename` sweeps the root — and a link
whose target is gone is drawn as a cut line rather than hidden.

**A dead link is not an error, it is information.** Rendering checks liveness once, so
"I wrote that relation" and "I misremembered" never look alike.

States: `spark` → `active` → `waiting` → `done` / `retired`. Size is priority; the ring
carries the state. Relations start with two words only: `related`, `depend`.

**Command line.** `cli.py` is the same mechanism without the page:

```
python cli.py list --group voice --state spark
python cli.py spark "the sentence" --group voice --priority 2
python cli.py state duplex-gap active
python cli.py link duplex-gap attention-cue --kind depend
python cli.py rename duplex-gap duplex-gap-between-clicks
```

Port: ephemeral by default — read the live URL from this channel's `url` notice. Pin
one with `--port N` or `MOSS_IDEAS_PORT`.
