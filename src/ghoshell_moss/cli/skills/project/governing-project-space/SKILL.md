
---
name: governing-project-space
description: Use when a ghost takes over a project directory for the first time — a .moss workspace exists but the tree has no declared shape, so every later decision (where does this file go, may this directory be deleted) is answered by local habit instead of by the project itself.
moss_version: 0.1.0-beta3
stage: stage3
level: L1
---

# Governing the project space

## Scenario

The ghost has just been installed into a project: `moss init` created `.moss/`, and beyond that
there is an empty directory, or one somebody left half-shaped. Nothing says where a work goes,
what may be deleted, or which of these folders is load-bearing.

The default is to follow the nearest habit: create `src/`, `data/`, `notes/` because that is what
one does. Each decision is locally reasonable. Together they produce a tree nobody reviewed — and
architecture entropy is precisely the thing no single file shows you.

This skill exists to make one act happen: **review the space as a whole, then declare it.**

## What this skill is — and what it may never become

It is an **inception**, not a constitution.

A constitution would be a fixed tree the project must obey. This is not that. Projects differ — a
lesson kit, a blog, a robot firmware drop, a library others install — and a fixed tree is wrong for
most of them.

What this skill delivers is the awareness that the space has to be governed *at all*, plus one
worked example of what governing it looks like. The topology below is **one answer, offered as a
reference**; the project may arrive at another, as long as the act is done and the result is
written down.

Never push it. If the project genuinely has no intention of governing its space, say so and skip.

## The act

1. **Walk the space.** What is here, what is it for, what is generated, what is a clone, what is a
   scratch pile. `ls -a` is the start, not the end: a directory whose purpose you cannot state is
   exactly the one this act is for.
2. **Choose the axis** and place every directory on it.
3. **Write the declaration** — one `GROUND.md` at the project root, plus one inside each
   dot-directory that carries a real convention. Not a table in a README: `moss ground render`
   must show it.
4. **Render it back.** What the render shows is what the next ghost will believe. If a directory you
   declared is invisible, or one you did not declare appears, the declaration is wrong — not the
   render.
5. **Name what is undecided.** A tree with a named hole is honest; a tree with a silent hole is a
   trap.

The deliverable is the `GROUND.md`. A discussion that ends in a conversation is a conversation.

## The axis: how long does it live

Do not ask "what is this?" — every directory answers that differently and you will argue forever.
Ask **how long it lives, and who hurts when it rots.**

| class | meaning | consequence |
|---|---|---|
| long-lived | versioned, distributable; somebody is pained when it rots | enters git, enters the declaration |
| short-lived | rebuildable, disposable; rots silently | dot-prefixed, ignored |
| external | somebody else's code, cloned or vendored | ignored — but **its coordinate is tracked** |

A dot prefix is not style: it **declares the second or third class**. A new directory that cannot
answer "who hurts when this rots?" is a dot directory.

## A reference skeleton (example, not a specification)

```
<project>/
  .moss/            # the ghost's own workspace — runtime, modes, ghosts, grounds
  .venv/            # python environment
  GROUND.md         # this project's constitution: identity, topology, what may be deleted
  features/         # L1-L4 governance of the work itself (moss features)
  .worktrees/       # parallel branch checkouts, one per experiment
  packages/         # separately installable packages — the monorepo mount point
  src/              # the code of THIS project (src-layout)
  .playground/      # disposable trial space; never in git, not reclaimed by the OS
  works/            # distributable products (see the managing-works skill)
  assets/           # shared inputs; assets flow in, products flow out
  .vendors/         # third-party clones, for reading and reference
  bin/              # entry scripts — including the one that starts this ghost
  skills/           # skills this project owns
  services/         # independent processes this project runs, in any language
  nodes/            # MOSS cells shared by this project — local copies, never links
  tests/            # what keeps the above honest
  docs/             # what this thing IS; features/ is what is being CHANGED
```

One line each, because a line is what a future reader needs; the detail belongs in that
directory's own `GROUND.md`.

Rules that generalize beyond the example:

- **`src/` holds this project's code; `packages/` holds what others install.** Write the boundary
  down, or the two will grow into each other.
- **`nodes/` holds local copies, never a path into another checkout.** Out-of-box nodes are
  *seeds*, not dependencies: a project whose behaviour lives in somebody else's branch cannot see
  that dependency in its own tree. Copy, then modify freely; the cost is that fixes do not
  propagate — only comparison does.
- **`services/` vs `nodes/`:** if the Matrix can own it (manifest, auto-join, channel control) it
  is a node; if it is a database, an HTTP endpoint, or somebody else's process, it is a service.
- **`.vendors/` ignores the content and tracks the coordinate.** The clone stays out of git; the
  pinned commit goes in. Otherwise a fresh clone has nothing telling it what belongs there.
- **A vendored or mounted sub-project is a fence.** It brings its own `.moss/`, `GROUND.md` and
  `nodes/`; discovery must stop at its boundary (`stop_on_match`), or this project silently adopts
  somebody else's home.

## Two things not to declare

- **Common sense is not a declaration.** `.venv/` needs no `GROUND.md` — nobody wonders what it is.
  Declaration is for what you *invented*: a topology, a convention, a boundary. Over-declaring turns
  the declaration into noise, and noise is what hides the one line that matters.
- **The discovery floor is not configurable.** `moss ground` already refuses to walk `.git`,
  `.venv`, `node_modules`, caches and `dist` — by basename, before any per-ground rule. You cannot
  re-include them by declaring it, and you should not try. One useful consequence: you do not have
  to name them in the project's ignore rules either. Name only what you deliberately keep out.

## What to leave behind

At the end of the act this must be true:

- `moss ground render .` shows every long-lived directory, each with one line saying what it is for;
- every surviving dot-directory carries its own short `GROUND.md`;
- the undecided parts are written as undecided;
- `.venv` and the machine-generated piles appear neither in the tree nor in the ignore list.

## A second reference: this repository

MOSShell's own root is a governed space; reading it is a legitimate second example. Two cautions:

- Some of its cells are **private to it** — `.ai_partners/` is this project's collaboration history,
  and `src/ghoshell_moss/` is one package with one name. Do not copy them as though they were the
  shape of every project.
- Its `features/` sits at `.ai_partners/features/` today — the legacy branch of a resolver that
  prefers `$CWD/features`. Copy the convention, not the accident.
