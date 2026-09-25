# Memento — Cognitive Trajectory System

Memento is MOSS's cognitive-trajectory infrastructure. It answers: how does a
Ghost remember, reflect, and fork — not by storing conversation logs, but by
anchoring experience in immutable commits and keeping interpretation forever
revisable.

## Core concept: trajectory is the first citizen

A Ghost lives on **lines** (timelines). Each line has a stable identity (uid)
and a human-readable name (head pointer). It extends forward by recording
**moments** into **staging**. When staging is frozen, it becomes a **commit** —
an immutable cognitive anchor. Commits can never be deleted or modified, but
their **annotation** (summary, title, thread tags) can be revised at any time.

```
record moment → staging (live, mutable)
                  ↓ commit (freeze)
               commit (immutable anchor)
                  ↓ annotate (aperture 2)
               commit + updated annotation
```

## Data model

| Concept | What it is | Mutability |
|---------|------------|------------|
| **Owner** | An identity that writes. One owner = one directory under the memento root. | — |
| **Line** (branch) | A stable uid (brn_ ULID) + a movable name + a ref pointer + staging workspace. | staging: mutable; ref: updated at each commit; name: movable/renameable |
| **Commit** | A frozen snapshot of moments. Born immutable, lives forever. | Never. Members frozen; annotation appends new versions (last-wins). |
| **Moment** | One frame of experience. An envelope with a plain-text content field + opaque payload. | In staging: overwritable (last-wins). Once in a commit: frozen. |
| **Staging** | The live edge of a line — moments not yet committed. | The only overwritable file in the system. |
| **Annotation** | Commit interpretation: title + body. Appended as new versions, last-wins. | Always revisable. Old versions forever addressable. |
| **Witness** | A git sidecar that records every file change as a verifiable audit trail. | Optional. Append-only (daemon). |

## Key design choices

- **uid vs name**: Every line has a stable uid (brn_ ULID) that never changes. The
  human-readable name is a movable pointer (heads/{name} → uid). Deleting a line
  removes only the head pointer — the workspace and all commits survive.
- **fork-over-reset**: To revisit a past commit, create a NEW line from it. The
  old line is preserved (or marked abandoned). There is no destructive rewind.
- **reference confluent (融汇)**: One line can submit its ref to another line.
  The recipient's commit parent chain is NOT altered — the confluent is recorded
  as a separate associative event in confluents.jsonl.
- **content field**: Every moment carries a plain-text `content` field, populated
  by the recording agent. Enables CLI views to render human-readable output
  without parsing opaque payload.

## Two apertures

1. **Aperture 1 (input queue)**: A side channel feeds commits into the main
   attention stream via `Memento-Ref` messages. "Advisory" side effect.
2. **Aperture 2 (annotation layer)**: Revise interpretations of past commits
   without touching the main flow. "Memory reconsolidation" side effect.

## Storage layout (FORMAT v3)

```
{root}/{owner}/
  meta.json                     # owner identity (overlay). Optional.
  commits.jsonl                 # append-only commit log (physical time order)
  branches.jsonl                # append-only branch index (uid, name, status, fork_ref)
  checkouts.jsonl               # fork event records (deriving side appends)
  confluents.jsonl              # reference-confluent event records
  heads/{name}                  # plain-text: branch_uid (one line, no JSON)
  ws/{branch_uid}/
    ref                         # {origin, commit_id[, moment_id]}
    staging.jsonl               # live moments (t:"moment" rows)
    status.json                 # lifecycle status + task description
  commits/{YYYY-MM}/cmt_<ULID>/
    meta.json                   # parent pointer
    moments.jsonl               # frozen member moments (t:"moment" rows, no header)
    notes.jsonl                 # annotations (commit_note + moment_note)
```

- **Y-m sharding**: `cmt_<ULID>` → UTC year-month is a pure function (no index needed).
- **commits.jsonl**: owner-level append-only log. Line order = physical time.
  Crash recovery predicate.
- **branches.jsonl**: append-only branch registry. Appended on create, status change,
  and delete (abandoned tombstone). Full-search API reads this file.
- **checkouts.jsonl**: fork events. Appended by the deriving side (local-only, zero
  cross-owner coordination).
- **heads/{name}**: lightweight pointer files. glob = active branch list.
  `moss memento branch delete` removes this file (appending an abandoned tombstone
  to branches.jsonl); workspace and commits survive.
- **ref**: JSON `{origin, commit_id[, moment_id]}`. Updated at each commit.
- **Commit autonomous dir**: born frozen, lazily created. Only `notes.jsonl` can be
  appended after creation.

## Typical workflows

**Dumb memory (degenerate form)**: One line, auto-commit. The simplest path —
record, commit, window. No fork vocabulary needed.

```
moss memento init
moss memento branch create <owner>/main
moss memento branch record <owner>/main '{"text":"hello"}'
moss memento branch commit <owner>/main -m "first"
moss memento branch window <owner>/main
```

**Fork**: Start a new line from any frozen commit (cross-owner allowed).

```
moss memento branch create <owner>/idea-x --from-ref <other-owner>/cmt_xxx
```

**Delete (head only)**: Remove the head pointer. Workspace and commits survive.

```
moss memento branch delete <owner>/old-idea
```

**Full branch index**: See all branches including abandoned.

```
moss memento branch list-all <owner>
```

**Annotation (aperture 2)**: Revise a commit's interpretation.

```
moss memento commit annotate <owner>/cmt_xxx -m "Updated summary" -t "Title"
```

**Inspect**: Show full commit content, or view the sliding window.

```
moss memento commit show <owner>/cmt_xxx [--notes]
moss memento commit space <owner>/cmt_xxx   # path to commit directory
moss memento branch staging <owner>/main     # live (unfrozen) moments
moss memento owner log <owner>               # physical commit timeline
moss memento branch list-all <owner>         # all branches (full index)
```

## Key invariants

1. Commit members (moment frames) are frozen on creation — never modified.
2. Annotations append new versions, last-wins (by file byte offset, not timestamp).
3. Payload is opaque — memento never parses or rewrites it.
4. Fork only from frozen commits, never from staging.
5. Single parent chain only — confluents are associative events, not parent-chain merges.
6. Deleting a line removes the head pointer only. Workspace and commits survive forever.
7. Every line has a stable uid. The human-readable name is a movable pointer.
