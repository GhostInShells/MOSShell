# MOSS Stage Tracking

## What

A **stage** is a declared development period with an explicit start, direction,
and delivery target. Each stage is a directory under `.ai_partners/stages/`,
anchored by a `STAGE.md` file that carries:

- **Background** — where this stage comes from, prior context
- **Motivation** — why this stage, why now
- **Goals** — what the stage intends to achieve (declarative, few)
- **Plan** — the path: key workstreams and their ordering
- **Acceptance** — what "done" means, the gate conditions
- **Associated Workstreams** — names of features / debates / regressions involved
- **Retrospective** — what actually happened (written at close, empty before)

A stage's lifecycle lives entirely inside its own STAGE.md:
**planning → active → completed**. The file travels from intent to record.
The closed STAGE.md *is* the milestone of the stage — no separate closing
artifact is created.

## Why

Every other artifact in `.ai_partners/` is anchored to **what has already
happened**: features are reverse indexes from code to context, milestones
record moments that occurred, debates and regressions bind to existing
designs or executable reality. The epistemic ground of the whole system
is fidelity to the past.

Stages are the one artifact that **faces forward**. They exist because
intent is not derivable from code:

1. A model surveying the codebase can only infer intent from current state —
   and that inference always reads as "be cautious, harden the foundations."
   When the project is actually in an exploratory sprint, this inference is
   exactly wrong, and models end up pressuring the human toward defensive
   engineering.
2. The human engineer drifts too. A declared stage is an anchor against drift.
3. External observers (contributors, users) cannot tell whether iteration
   is directed or random without a visible plan.

The design resolves the tension between forward-facing intent and the
system's fidelity principle with one rule: **a stage carries trajectory,
not truth.** Goals, motivation, and retrospective are authored intent and
memory. Progress and status are *observed* from the associated features at
read time — never copied into the stage file, where they would rot.

## How

### Create a stage

```bash
cp -r .ai_partners/stages/_template .ai_partners/stages/YYYY-MM-identifier
# edit STAGE.md: frontmatter + all sections except Retrospective
# add an entry to ROADMAP.md under Planned
```

### Activate a stage

Set frontmatter `status: active`, move its ROADMAP.md entry to Active.

### During a stage

- Link workstreams in **Associated Workstreams** as they start — by name
  only, never by path. Names resolve through `moss features list`; a
  renamed workstream is still traceable through git history, while a
  hard path is just a broken link.
- Record milestones — planned or emergent — as files under `milestones/`.
  An emergent milestone that changes the plan should say so (see the
  template's Stage Impact section).
- If direction is redrawn mid-stage, append a dated note to STAGE.md.
  A single dated section is enough; only promote to a subdirectory when
  a second file of the same kind appears.

### Close a stage

1. Write the **Retrospective** section: actual delivery, lessons, redirections.
2. Set frontmatter `status: completed`.
3. Move the ROADMAP.md entry to Completed.
4. Optionally promote defining moments to the global
   `.ai_partners/milestones/` (curated highlights, hand-written by model
   collaborators — distinct from the stage's operational `milestones/` log).

## Conventions

### Naming

Stage directories: `YYYY-MM-{identifier}/` — start year-month plus a short
identifier (`2026-07-beta1`, `2026-08-v0.1.0`).

Milestone files: `YYYY-MM-DD-{identifier}.md` — dated, self-explaining.

### Status

| Status | Meaning |
|--------|---------|
| `planning` | Declared, not started |
| `active` | In progress |
| `completed` | Closed with retrospective written |
| `cancelled` | Abandoned (rare; say why in Retrospective) |

### Content principles

1. **Trajectory, not truth.** Never copy feature status into a stage.
   Status is observed from the associated artifacts at read time.
2. **Names, not paths.** Associations are by artifact name, resolved
   through each mechanism's own index (`moss features list`, the
   regressions directory, ...). No relative-path coupling.
3. **Retrospective is not a task log.** Record key decisions, surprises,
   and redirections — not a checklist replay.
4. **Goals are few and declarative.** Three to five. If a goal needs a
   progress bar, it belongs in a feature, not here.
5. **Historical stages are optional.** Pre-mechanism history lives as
   background in the feature that produced this convention
   (`stage-tracking-convention`); back-filling stage files is not required.

## Relationship to other mechanisms

| Mechanism | Granularity | Temporality | Carries |
|-----------|-------------|-------------|---------|
| **features** | one workstream | persistent, backward-indexing | design context, decisions, dead ends |
| **stages** | one development period | weeks/months, forward-facing | direction, goals, retrospective |
| **milestones** (global) | one moment | historical, curated | highlights hand-written by collaborators |
| **debates** | one design discussion | L3 | colliding positions on architecture |
| **regressions** | one executable suite | repeatable present | model-runnable verification flows |

Stages say *what a period is for*. Features say *how a specific thing is
built*. Milestones say *what moment was reached*.

## Observability

Stages are governed by `moss ground`, not by a dedicated CLI. The intended
observation surface: `@` this README plus `ls` of the stage directories,
at most a glob over `*/STAGE.md` frontmatter. If ground lacks an observation
verb this needs, extend ground — see the `stage-tracking-convention` feature
for the gap list.

This mechanism is deliberately not wired into `moss start`. It must prove
itself in use; if it doesn't earn its keep, delete the whole directory.
