---
$id: moss-project/works
name: works
description: Things made with moshell — by moshell's ghost and its developers
pins:
- verb: ls
  label: here
  arguments:
    path: $CWD
    depth: 1
  description: Expand this level to see the propositions
---

# Works

Things made with moshell: a blog, a set of lessons, a video treatment, a research note. Each was
produced by the ghost living in this repository, working with its developers.

They can be shown or remade by the ghost of moshell: an agent or a ghost picks a piece and shows
it, or makes another one on request.

## What is here

Pieces are grouped by what they are about — one directory per line of work, such as `blogs/` or
`film-reviews/`. Each piece is a directory holding its own `WORK.md`, which records how it was
made and what it takes to make another.

## Looking around

`moss ground render <path>` renders any level of the tree. Every level may carry its own
`GROUND.md` describing what it holds. To locate a particular piece, searching for `WORK.md` is
the shortest path.

