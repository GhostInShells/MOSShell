---
created: 2026-10-21
depends: []
description: Make `works` a mechanism rather than a naming habit — a governance model that lets an instantiated moshell project (and the ghost inside it) hold an unbounded, heterogeneous body of works, keep a few of them as few-shots, and tell which ones are worth distributing. Templates, tiers and directory names are delivery detail; the mechanism is the deliverable.
milestone: null
priority: P1
status: in-progress
status_note: 2026-10-21 — third rewrite. The first draft was a memo to its author; the second argued from one private incident. This one states the requirement (scaling governance / few-shots / dogfooding) and treats everything concrete as detail.
title: Works Mechanism — works 是机制，先是种子，再长成树根
updated: '2026-10-21'
---

# Works Mechanism

## 1. Why now

moshell is becoming a way to **instantiate projects**, not just a framework. Every instantiated
project will produce works: a *Wandering Earth 3* prediction video, a set of primary-school lesson
materials, a charter, a blog. Three requirements follow, and none of them is met today.

**1. Scaling governance.** The corpus is unbounded and heterogeneous — different media, different
audiences, different lifetimes, more arriving every week. A model or a ghost placed into such a project
must know how to govern that corpus, or at the very least be handed a reference means of doing it.
Without one, every producer invents his own filing, the tenth work is invisible to the eleventh, and
"what do we have" becomes a question nobody can answer.

**2. Few-shots in the repository.** moshell must keep a few works in tree, because they are how whoever
receives this repository — a person or a model — learns what this project can actually do. A README
says what the tool is; a work shows what it makes.

**3. Dogfooding toward distribution.** moshell should use its own system to produce works worth
distributing. That is stage 3's goal, and it is also the only way the mechanism ever gets tested at
product quality instead of demo quality.

Everything concrete — templates, distribution tiers, directory names, file layouts — is **delivery
detail**. What this feature builds is a **mechanism**: a way for any moshell project and any ghost home
to hold a growing body of works without losing them, and without losing the ability to say which ones
matter.

## 2. What was on disk before this was written

The author's information surface, so the next instance can tell confirmed from assumed.

| Read | What it settled |
|---|---|
| `works/README.md`, `works/GROUND.md` | a frame already exists in this repo: five content rules and a named space. It is a habit of one repository, not a mechanism |
| `works/blogs/**` | a proposition can carry a whole product line with its own `CLAUDE.md`, `.gitignore`, `_sidebar.md` — evidence that a work may be a site, not a document |
| `nodes/webview_apps/svg_works/NODE.md` and its `works/` | a precedent worth borrowing: a work exported as a pair (`.jsonl` = the actions, `.md` = the thinking), directory gitignored, `git add -f` as the deliberate act of publishing |
| `src/ghoshell_moss/cli/skills/AGENTS.md` | skills live at `skills/<category>/<name>/SKILL.md`; its **zero-context handoff** is already the acceptance idea a skill is held to |
| `.ai_partners/CLAUDE.md` | the collaboration failure modes this document must not reproduce — silent todo, doc-first delivery |
| `.moss/ghosts/moss/` | existence / journal / people / skills / features exist; there is no `works` |

**Not verified:** whether a *foreign* moshell project can adopt the result. That is track ② below and is
outside this feature on purpose — a mechanism validated only at home has not been validated.

## 3. The judgment

### 3.1 A mechanism, not a habit

A convention that lives in one repository's README dies with that repository's readers. A mechanism has
to be findable at task start, adoptable by a project that has never heard of it, and reproducible after
its author is gone. The test is not "does it look organised here", it is "can a project that arrives
with ten thousand works still say which ones matter".

### 3.2 Four tracks, four lifetimes

They run in parallel and must not be collapsed into one plan. Collapsing them is what made the first
draft of this document read like a memo to its author.

| Track | What it is | Lifetime | Where it stands |
|---|---|---|---|
| ① **this feature** | the record of how the mechanism was born | short; closes when accepted | 2026-10-21, manually accepted |
| ② **the skill** in `moss skills` | the root: how anyone else grows works | medium; re-validated and revised | first version written; **validation is future** — a fresh directory, a zero-context project, see whether its ghost can start one |
| ③ **the two trees** (project `works/`, ghost home `works/`) | where works are produced for years | long; governance keeps changing | planted 2026-10-21 |
| ④ **each concrete work** | one delivered product | ends when the work ends | one leaf so far |

### 3.3 Four layers: seed → root → trunk → leaf

- **seed** — a works *system* is a base mechanism of moshell, so it grows from a **feature** (this one);
- **root** — that feature's deliverable is a **skill**: how to govern works in a project or a ghost home;
- **trunk** — moshell's own `works/` and the ghost's `works/` are grown by **following the skill**;
- **leaf** — a single work grows from a trunk, never from a feature.

Two sentences that are easy to collapse and must not be — written down because the author got it wrong
twice in one evening:

> A works **system** is a mechanism → grown from a feature.
> A single **work** is a product → grown from the system.

### 3.4 works is tertiary production

**features is primary production** (it builds capability and mechanism); **works is tertiary** (it
delivers things to people). That sentence settles ownership:

- a feature decides *what the tools are*;
- a work decides *what gets delivered*.

So `小学教育` is **not** "a feature that grows several works". It is one proposition directory under
`works/`; the lessons inside are individual works, governed independently — with or without the
features mechanism, whichever the producer prefers.

### 3.5 What a work is, and is not

A work **is the product**. `works/blogs/` is a blog; it may drive ten openbox nodes and it is still a
blog. `WORK.md` is the work's entry document — the thing that makes it reproducible — not the work.

Classification is by **concrete proposition** (`blogs/`, `film-reviews/`, `curriculum/`), never by
activity type (`making/`, `ops/`): activity is what skills are for. And a directory holding no
`WORK.md` is legal — a works space is a production system first, a museum second.

## 4. What was delivered on 2026-10-21, and how it is accepted

Delivered (detail, revisable):

1. `src/ghoshell_moss/cli/skills/project/managing-works/SKILL.md` and two templates beside it —
   the root, in the place a task-start reader already looks.
2. `works/GROUND.md` — spec pointer, topology, distribution tier `all`; the five content rules in
   `README.md` untouched.
3. `.moss/ghosts/moss/works/` — `GROUND.md` plus `.gitignore` at tier `process`.
4. One leaf, recorded as methodology only: a lesson series delivered to a real pupil, with every
   product ignored and the re-fetch instructions written into the document.
5. Two drawings of the model, kept beside this file:

   - `tree-logic.svg` — seed → root → trunk → leaf, plus the one sentence that must not be
     collapsed;
   - `four-rows.svg` — the four tracks and their different lifetimes.

**Acceptance is manual**, because track ① is judged by a person:

- read the skill cold, with its templates: *could I start a works space for a new proposition without
  asking the author anything?*
- read the one leaf's document with its products out of reach: *could someone produce them again from
  this text alone?*
- neither `GROUND.md` lists works; the two spaces declare different distribution tiers;
- `git status` shows documents only.

## 5. Out of scope, on purpose

- **② the skill's validation in a foreign project** — needs a new directory and a zero-context project;
  a later feature's business.
- **③ long-term governance** — the two trees will keep changing their method; that is normal use, not
  this feature's delivery.
- **④ the quality of any individual work** — including the first leaf. A work is judged by its audience.

## 6. Reversal

Everything here is addition: one feature directory, one skill directory, one `GROUND.md` per tree, one
work directory. Delete them and the repository is back — no migration, no schema, nothing to unlearn.


