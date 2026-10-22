# ideas — the data root

Each first-level subdirectory here is a **group**; each `*.idea.md` inside it is one
idea. A group's `README.md` is that group's own context: what it is for, what to read
first.

```
ideas/                   ← the data root, gitignored, seeded from these examples
  README.md              ← this file
  example-voice/         ← a group
    README.md            ← the group's shared context
    duplex-gap.idea.md   ← one idea: frontmatter + body
  example-lifecycle/
```

## One idea, one file

```markdown
---
state: spark          # spark | active | waiting | done | retired
priority: 2           # the bubble's size
tags: [voice, duplex]
links: [attention-cue, voiceprint:depend]
---

The body is the thinking itself. Edit it by hand; nothing else holds a copy.
```

`links` entries are filenames without `.idea.md`; `name` means `related`, `name:depend`
means this idea depends on that one. **The filename is the identity and it is
semantic** — `duplex-gap.idea.md` says what it is before anything is opened.

## The examples are disposable

Everything in this directory is an example. It is copied into the data root on first
run and then belongs to whoever owns that root — delete it, edit it, ignore the whole
mechanism. The node never assumes its data is precious.
