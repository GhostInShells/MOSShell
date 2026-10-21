---
name: branch
description: What a line is right now — its plan, its tasks, and how they get archived
pins:
- label: plan
  verb: file
  arguments: {path: "$CWD/plan.md", budget: 1200}
  description: The plan for this stretch; its status lives in the frontmatter
- label: tasks
  verb: frontmatter
  arguments: {path: "$CWD/*.task.md", keys: [name, status, updated], limit: 15}
  description: Which tasks exist, where they stand, and when they last moved
---

# The working surface of a line

This law sits at `branches/`, once, and covers every branch. A branch directory carries no
`GROUND.md` of its own: rendering walks up here and reads the pins *at the branch you are
standing in*, which is why they are `$CWD`-anchored. Write the law once; let branches stay
empty.

**Write the plan here, in the branch directory.** A stretch of work starts as `plan.md` and
its steps as `slug.task.md` — beside each other, not in a scratch document somewhere else.
Branch view renders that directory, so writing it there is also what makes it come back to me
on the next look.

**plan.md** — one per line, answering "what are we doing". **`slug.task.md`** — one per task.

```md
---
status: active          # active | done | dropped  (closed / completed also count)
created: 2026-10-21T16:00+08:00
updated: 2026-10-21T16:05+08:00
---
```

Tasks add `name`; nothing else is required.

`updated` is part of the convention: touch the file, bump it. The pins show it, so a task's
age is visible before its text is — a week-stale task is a question, not work in progress.

`done` / `dropped` get archived; anything else counts as active — better to keep a file than
to move one by mistake.

## Archiving is not my job

When an anchor lands, `on_commit` moves the finished plan and tasks into the new commit
directory (`commits/…/cmt_{coord}/`). Two things are mine: **write the status correctly**,
then **commit as usual**. Moving files must not depend on my remembering — remembering fails,
a file does not.

## Reading discipline

The pins above give existence and status only. A line with ten tasks would fill the screen:
**open the file when you need its text**, never expect the ground to unfold it for you.
