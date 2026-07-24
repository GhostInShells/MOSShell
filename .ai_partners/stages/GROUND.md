---
label: stages
pins:
- label: roadmap
  verb: file
  arguments:
    path: ROADMAP.md
  description: cross-stage index — active / planned / completed
- label: stage_status
  verb: frontmatter
  arguments:
    path: '*/STAGE.md'
    keys: [status, period, delivery]
  description: lifecycle state of every stage, observed not copied
- label: stage_dirs
  verb: ls
  arguments:
    path: .
  description: what artifacts each stage has accumulated
---

# Stages Ground

Development stages of MOSS. Each subdirectory is one declared period —
see [README.md](README.md) for the mechanism, `ROADMAP.md` for the index.

A stage carries trajectory, not truth: goals and retrospectives are
authored intent; progress is observed from the associated workstreams
(`moss features list`) at read time.
