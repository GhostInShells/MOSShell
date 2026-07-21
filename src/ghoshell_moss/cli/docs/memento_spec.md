# Memento — Cognitive Trajectory System

Memento is MOSS's cognitive-trajectory infrastructure. It answers: how does a
Ghost remember, reflect, and fork — not by storing conversation logs, but by
anchoring experience in immutable commits and keeping interpretation forever
revisable.

## Core concept: trajectory is the first citizen

A Ghost lives on **lines** (timelines). Each line extends forward by recording
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
| **Line** (branch) | A timeline: a name + a ref pointer + staging. | staging: mutable; ref: movable (reset) |
| **Commit** | A frozen snapshot of moments. Born immutable, lives forever. | Never. Members frozen; annotation appends new versions (last-wins). |
| **Moment** | One frame of experience. An envelope: payload is opaque to memento. | In staging: overwritable (last-wins). Once in a commit: frozen. |
| **Staging** | The live edge of a line — moments not yet committed. | The only truncatable file in the system. |
| **Annotation** | Commit interpretation: title + body. Appended as new versions, last-wins. | Always revisable. Old versions forever addressable. |
| **Witness** | A git sidecar that records every file change as a verifiable audit trail. | Optional. Append-only (daemon). |

## Two apertures

1. **Aperture 1 (input queue)**: A side channel feeds commits into the main
   attention stream via `Memento-Ref` messages. "Advisory" side effect.
2. **Aperture 2 (annotation layer)**: Revise interpretations of past commits
   without touching the main flow. "Memory reconsolidation" side effect.

## Storage layout

```
{root}/memento/{owner}/
  meta.json                    # owner identity (overlay, created/updated)
  commits.jsonl                 # append-only commit log (physical time order)
  branches/{name}/
    ref                         # {origin, commit_id[, moment_id]}
    staging.jsonl                # live moments
  commits/{YYYY-MM}/cmt_<ULID>/
    meta.json                    # parent pointer + ancestry skips
    moments.jsonl                 # frozen member moments
    notes.jsonl                   # annotations (commit_note + moment_note)
```

- **Y-m sharding**: `cmt_<ULID>` → UTC year-month is a pure function (no index needed).
- **commits.jsonl**: owner-level append-only log. Line order = physical time. Crash recovery predicate.
- **ref**: points to a commit. Moving it = `reset` (auto-mechanical-commit first, then move — nothing silently lost).
- **Commit autonomous dir**: born frozen, lazily created. Only `notes.jsonl` can be appended after creation.

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

**Rewind (reset)**: Move a line's ref to a different commit. Staging is
auto-committed first — nothing is silently discarded.

```
moss memento branch reset <owner>/main --to <owner>/cmt_yyy
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
```

## Key invariants

1. Commit members (moment_ids) are frozen on creation — never modified.
2. Annotations append new versions, last-wins (by file byte offset, not timestamp).
3. Payload is opaque — memento never parses or rewrites it.
4. Fork only from frozen commits, never from staging.
5. Ref movement anchors staging first (auto mechanical commit).
6. Merge does not exist — single parent chain only.
7. Deleting a line removes staging + ref; commits survive forever.
