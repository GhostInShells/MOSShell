---
name: managing-works
description: Use when a ghost or an agent enters a project that keeps producing things which are not code — a blog, a curriculum, a charter, a research line — and the project has no way to say what it has, which of it matters, or how to make another one.
moss_version: 0.1.0-beta3
stage: stage3
level: L2
---

# Managing works

## Scenario

This is not a filing convention for people. It is a **human–model collaboration strategy**, and for a
ghost it answers a practical question: *the project I am inside keeps producing things — where are they,
which ones matter, and can I make another one?*

A project without such a strategy loses its work quietly: the tenth piece is invisible to the eleventh,
nobody can say what the collection contains, and every new product is made from scratch.

>> 从这一段需要强调, 本 skills 提供一种策略供参考

## What this skill is — and what it may never become

It offers **a topology and two reference strategies**. It is not a constraint. A project may replace
either strategy with an equivalent mechanism of its own, as long as the governance act itself is done.

What it actually delivers is not the topology. It is the **awareness that the space has to be governed
at all** — that a product left in a scratch directory is a product that has already been lost, and that
somebody has to answer for where it lives, a year from now. The topology is one answer to that; a project
may find another, but the question cannot be skipped.

And it must never be pushed: if the project has no intention of governing its works at all, the correct
behaviour is to skip this skill entirely and say so.

## The topology (reference)

```
<space>/<proposition>/<work>/
```

- **proposition** — a concrete, recurring line of products (`blogs/`, `curriculum/`, `film-reviews/`),
  never an activity type (`making/`, `ops/`): activity is what skills are for.
- **work** — one delivered thing.
- A space holding no work yet is legitimate: a production system that nobody has used is still a system.

## Strategy 1 — declare the space (optional)

A `GROUND.md` at the root of a works space tells a reader four decisions were made:

1. the **directory topology** — what the levels mean *here*;
2. the **classification plan** — how new work is filed without asking;
3. the **surface strategy** — how a reader discovers what exists, and what is generated rather than typed;
4. the **attention budget** — how much of a reader this space is allowed to cost.

Any equivalent mechanism may carry these four decisions instead — a skill with its own scripts will do.
What matters is that the decisions exist somewhere the next reader already looks.

This strategy is only meaningful where `moss ground` is integrated. Where it is not, put the same four
decisions where the project already keeps its local rules.

*Exit*: someone reading only this can add a work without asking anyone.

## Strategy 2 — record the making (optional)

A `WORK.md` inside a work says the context of making is worth keeping, for four reasons:

1. a work should be **re-creatable**, not merely stored;
2. the **environment and resources** it depended on are part of it;
3. if the product does not explain itself (an html page, a blog), **how to use it** has to be said;
4. all of the above should be **cheap to discover** — greppable, not read-through.

If you use it, three things are mandatory: **motive**, **method**, **product**. The method is the body —
the capability chain, and the judgement at each fork where a reproduction would have to decide the same
thing you did. Everything else is judgement.

A directory-level `AGENTS.md` is the same species of convention: a place a reader expects the local rule
to be, one level above the files it governs.

*Exit*: the work could be reproduced with every product deleted.

## The judgements behind the two strategies

- **Does this project keep producing things?** If it is one throwaway file, build nothing.
- **Is this a work or a proposition?** Recurring with an audience → a proposition with works inside;
  one-off with an audience → a work.
- **Do the products ship?** What a reader does with them decides it: open them → ship; regenerate them →
  do not.
- **Is the making worth keeping?** If nobody will ever make a second one, a record is decoration.

## Exploration path

1. `templates/GROUND.template.md` and `templates/WORK.template.md` beside this skill — starting points,
   not forms to fill.
2. Two live instances: this repository's own `works/` (a blog product line; tier `all`) and the ghost
   home's `works/` (tier `process`). Read their `GROUND.md` first — they differ on purpose, and the
   difference is the lesson.
3. `moss ground` — how a space declares itself and how a reader expands it. This is the tool Strategy 1
   depends on; read its behaviour before promising anything on its behalf.

## How to trust this

Four checks, in this order. The fourth is the one that decides whether the skill deserves to exist.

1. Does the official moshell repository's own `works/` follow this mechanism? If not, the skill is
   describing something nobody actually does.
2. Can a reader discover the works a project already has — cheaply, without reading everything?
3. Can those works be reproduced, or used, by someone who was not there?
4. Did steps 1–3 produce **governance value** — fewer questions, less duplication, a corpus that can be
   reasoned about? If not, stop: the effort is being spent on form.

If the answer to 4 is no — or if the project has not decided to govern its works at all — this skill
should be skipped without ceremony.
