---
name: commits
description: What one anchor was — what it left behind
pins:
- label: contents
  verb: ls
  arguments: {path: "$CWD", depth: 2, limit: 40}
  description: What this anchor's directory holds
---

# The anchor directory

This is the commit view's ground point. It is rendered only when one commit is read in
detail: resolution walks up to `commits/` and stops, and **only the current anchor's
directory** is rendered — there is no per-commit copy of this ground.

## What is inside

- The plan and tasks whose status had ended in that stretch, moved in by `on_commit`.
- Whatever a hook wrote itself.
- Not every anchor has a directory — empty is not broken.
