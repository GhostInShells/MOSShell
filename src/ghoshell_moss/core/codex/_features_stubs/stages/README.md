# Stages

## What

A **stage** is a declared development period with an explicit start, direction, and delivery
target. Each stage is a directory under `stages/`, anchored by a `STAGE.md` that carries:

- **Background** — where this stage comes from, prior context
- **Motivation** — why this stage, why now
- **Goals** — what the stage intends to achieve (declarative, few)
- **Plan** — the path: key workstreams and their ordering
- **Acceptance** — what "done" means, the gate conditions
- **Associated Workstreams** — names of workstreams / regressions involved
- **Retrospective** — what actually happened (written at close, empty before)

A stage's lifecycle lives entirely inside its own `STAGE.md`: **planning → active →
completed**. The file travels from intent to record — the closed `STAGE.md` *is* the stage's
milestone, no separate closing artifact is created.

## Why

Every other artifact in the features tree is anchored to **what has already happened**:
workstreams are reverse indexes from code to context, regressions bind to an executable
reality. Their ground is fidelity to the past.

Stages are the one artifact that **faces forward**, because intent is not derivable from
code:

1. A model surveying the codebase can only infer intent from current state — and that
   inference always reads as "be cautious, harden the foundations." In an exploratory
   sprint it is exactly wrong.
2. A human engineer drifts too; a declared stage is an anchor against that drift.
3. An outside observer cannot tell whether iteration is directed or random without a
   visible plan.

One rule resolves the tension between forward-facing intent and fidelity: **a stage carries
trajectory, not truth.** Goals, motivation and retrospective are authored intent and memory;
progress and status are *observed* from the associated artifacts at read time — never copied
into the stage file, where they would rot.

## How

### Create

```bash
cp -r stages/_template stages/YYYY-MM-identifier
# edit STAGE.md: frontmatter + every section except Retrospective
# add an entry to ROADMAP.md under Planned
```

### Activate

Set frontmatter `status: active`; move its ROADMAP entry to Active.

### During

- Link workstreams in **Associated Workstreams** as they start — **by name only, never by
  path.** Names resolve through each mechanism's own index; a hard path is only a broken link
  waiting for a rename.
- Record milestones — planned or emergent — as files under `milestones/`. An emergent
  milestone that changes the plan says so (see the template's Stage Impact section).
- If direction is redrawn mid-stage, append a dated note to `STAGE.md`. One dated section is
  enough; promote to a subdirectory only when a second file of the same kind appears.

### Close

1. Write the **Retrospective**: actual delivery, lessons, redirections.
2. Set frontmatter `status: completed`.
3. Move the ROADMAP entry to Completed.

## Conventions

### Naming

- Stage directories: `YYYY-MM-{identifier}/` — start year-month plus a short identifier.
- Milestone files: `YYYY-MM-DD-{identifier}.md` — dated, self-explaining.

### Status

| Status | Meaning |
|--------|---------|
| `planning` | Declared, not started |
| `active` | In progress |
| `completed` | Closed with retrospective written |
| `cancelled` | Abandoned (rare; say why in the Retrospective) |

### Content principles

1. **Trajectory, not truth.** Never copy workstream status into a stage; status is observed
   from the associated artifacts at read time.
2. **Names, not paths.** Associations are by artifact name, resolved through each
   mechanism's own index. No relative-path coupling.
3. **Retrospective is not a task log.** Key decisions, surprises, redirections — not a
   checklist replay.
4. **Goals are few and declarative.** Three to five; a goal that needs a progress bar
   belongs in a workstream.
5. **Historical stages are optional.** Back-filling stage files from before this convention
   existed is not required.

## Relationship to the other mechanisms

| Mechanism | Granularity | Temporality | Carries |
|-----------|-------------|-------------|---------|
| **workstreams** | one workstream | persistent, backward-indexing | design context, decisions, dead ends |
| **stages** | one development period | weeks/months, forward-facing | direction, goals, retrospective |
| **regressions** | one executable suite | repeatable present | model-runnable verification flows |

Stages say *what a period is for*; workstreams say *how a specific thing is built*.

## Observability

Intended surface: this file, a listing of `stages/*/`, and at most a glob over
`*/STAGE.md` frontmatter. Anything more belongs in a tool, not in a document.
