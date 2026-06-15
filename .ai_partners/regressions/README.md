# Regression Tracking Convention

> REGRESSION.md is a **verification trajectory** — what was tested, how it was tested,
> and what was found. It is the sibling mechanism to features: features track what was
> built and why; regressions track how it was verified and what broke.

## Why

Automated tests (unit, integration, e2e) tell you what passes and what fails.
They don't tell you:

- **Why** this set of tests exists and what methodology shaped it
- **What was tried before** — the dead ends that led to the current approach
- **How to execute** the tests when environment setup, hardware prerequisites,
  or human-in-the-loop steps are required
- **What the baseline result was** — the first complete run that established
  "this system was healthy at this point"
- **What broke and why** — root cause diagnoses that compound across sessions

A regression set bridges this gap. It combines executable test cases with the
context that makes them meaningful to the next human or model who runs them.

REGRESSION.md is the map. Git log is the timeline. Baselines are the
reference points — the first complete run of each version, preserved as
snapshots you can compare against.

## Core Concepts

### Regression Set

A named collection of test cases that verify a specific scope — a subsystem,
an integration boundary, or a release surface. Each set lives in its own
directory under `.ai_partners/regressions/`.

A regression set has:
- **Methodology** — the testing approach, prerequisites, automation level
- **Test cases** — a table of cases with priority, steps, and expected results
- **Version** — an incrementing counter that tracks case structure changes
- **Baselines** — snapshots of the first complete run for each version

### Version

`version` in the frontmatter is an update counter. It starts at 1 and
increments when the test case structure changes — cases added, removed,
or materially modified. It does NOT increment for:
- Re-running the same cases (that's a new baseline under the same version)
- Typo fixes or description clarifications
- Frontmatter updates (status, priority, etc.)

Version changes are tracked in git. The version number in the frontmatter
is a quick signal, not an audit trail — `git log` is the audit trail.

### Baseline

A baseline is the **first complete run** of a given version's test case set.
It is an immutable snapshot: the case table at that point in time, plus
the results (first test outcome, root cause diagnosis, final result).

A version has exactly one baseline — the first complete run. If you re-run the same
cases later and get different results, you either:
- Create a new version (if the difference matters and you want to track it)
- Or accept that the baseline stands and move on

Baselines live in `baselines/YYYY-MM-DD_vN.md`. The date is when the
run was completed. The version number ties it to the REGRESSION.md version
that was current at that time.

### Test Case Table

The table in REGRESSION.md is the **live** case list — the authoritative
set of what should be tested right now.

| Column | Meaning |
|--------|---------|
| Case ID | Unique within the set, e.g. `TC-001` |
| Priority | `P0` (blocking), `P1` (should pass), `P2` (nice to have) |
| Description | What this case verifies |
| Test Steps | How to execute — commands, scripts, manual actions |
| Expected Result | What a passing run looks like |

In a baseline file, three result columns are appended:

| Column | Meaning | Possible values |
|--------|---------|-----------------|
| First Test | Outcome of the first run | `PASS`, `FAIL`, `BLOCKED` |
| Fix | Root cause diagnosis if first test failed | Free text (not a commit hash) |
| Final Result | Outcome after fixes or decisions | `PASS`, `FAIL`, `SKIP` |

`BLOCKED` means the test couldn't run — missing environment, hardware not
connected, dependency not ready. `SKIP` means a conscious decision that the
case doesn't apply in this context (e.g., hardware-specific case on a
machine without that hardware).

The `Fix` column records the **root cause diagnosis**, not the solution.
"LowState packets 2180B > MTU 1500, ufw dropped IP fragments past the
first" is valuable to the next person. A commit hash is not.

## Workflow

### Creating a Regression Set

```
Copy TEMPLATE.md → .ai_partners/regressions/<name>/REGRESSION.md
Fill in frontmatter (version: 1, status: draft)
Define methodology
Build the test case table
Run all cases → create first baseline
Set status: active
Commit
```

No CLI. The template is a file — copy it, edit it, commit it.

### Updating a Regression Set

When test cases change (add, remove, restructure):
- Increment `version` in frontmatter
- Update `updated` date
- Add cases to the table (or modify existing ones)
- Run all cases → create a new baseline for the new version
- Commit REGRESSION.md + the new baseline together

### Running a Regression

```
1. find .ai_partners/regressions -name "REGRESSION.md"
2. Read the REGRESSION.md for the set you need
3. Follow the methodology, execute cases in priority order
4. Compare results against the latest baseline
5. If this is the first full run of a new version → create a baseline
```

## Directory Topology

```
.ai_partners/regressions/
  README.md                    # This file — the convention specification
  TEMPLATE.md                  # Template for new regression sets
  <regression-name>/           # kebab-case, semantic (no date)
    REGRESSION.md              # ONLY file created initially (copy from TEMPLATE)
    baselines/                 # Created on first baseline write — do NOT pre-create empty
      2026-06-15_v1.md         # YYYY-MM-DD_vN — first complete run of version N
    discuss/                   # Optional, created only when content exists
    design/                    # Optional, created only when content exists
```

Only REGRESSION.md is created upfront. `baselines/`, `discuss/`, and `design/`
appear when content is committed to them. Never pre-create empty directories —
an empty `baselines/` signals "there are baselines here" when there aren't.

Discovery: `find .ai_partners/regressions -name "REGRESSION.md"` returns
every active regression set. No CLI, no index file. The filesystem is
the index.

## REGRESSION.md Frontmatter Schema

```yaml
---
title: Human-readable title
version: 1            # Update counter: increment when case structure changes
status: draft         # draft | active | expired
priority: P1          # P0 | P1 | P2 | P3
created: YYYY-MM-DD
updated: YYYY-MM-DD
scope: subsystem      # subsystem | integration | release
depends: []           # Regression set names or feature names this depends on
description: >-       # One-line summary for listing
  Brief description.
---
```

## State Machine

```
draft → active → expired
```

- `draft`: Test cases are being designed, no baseline yet
- `active`: At least one complete run exists, the set is ready for use
- `expired`: The tested system has changed enough that the methodology or
  cases no longer apply. The set remains as a historical reference.

`expired` is not `completed`. A regression set doesn't finish — it becomes
irrelevant because the target system moved on. The expired set stays in
the directory; it's just not expected to pass against current code.

## Model's Role

- **Bootstrap at session start.** `find .ai_partners/regressions -name "REGRESSION.md"`
  discovers all regression sets. This is the regressions equivalent of
  `moss features list` — two discovery paths, same session start habit,
  one CLI, one find. Run both.
- **Co-author test cases with the human.** The human defines the scope and
  priorities ("what matters if it breaks"). The model proposes cases by
  mapping human concerns onto the system's interfaces and boundaries. The
  human accepts, rejects, or reshapes. Both own the result.
- **Guide humans through execution.** The model reads the methodology and
  test steps, the human performs the actions, the model records results.
  The G1 pattern: the human says "verify this," the model proposes how,
  the human aligns, the model carries the human through execution.
- **Record root cause, not commit hash.** The `Fix` column in baselines
  exists to compound diagnostic knowledge across sessions. A clear
  diagnosis saves the next instance from repeating the investigation.
- **Mark expired when the target system drifts.** If you discover that a
  regression set's methodology no longer applies, mark it `expired` and
  note the reason in the methodology section. Don't leave stale sets
  in `active` status — they'll waste the next instance's time.
- **Create a baseline after the first complete run of a new version.**
  This is the reference point. Without it, the next instance has no
  baseline to compare against.

## Scope: When to Create a Regression Set

A regression set is warranted when the verification involves **context
worth handing off**: multi-step setup, hardware prerequisites,
human-in-the-loop steps, or non-obvious diagnostic patterns.

Skip it for:
- Pure automated tests that `pytest` covers — their methodology is in
  the test code itself
- One-off verification that completes in a single session with no
  cross-session value
- Trivial smoke tests with no diagnostic depth

## Relationship to Features

| | Features | Regressions |
|---|---|---|
| Tracks | What was built and why | How it was verified and what broke |
| Primary document | FEATURE.md | REGRESSION.md |
| Lifecycle | draft → in-progress → completed | draft → active → expired |
| Directory naming | date-path: YYYY/MM/name | semantic: name |
| Version | Dates in frontmatter + paths | Version counter in frontmatter |
| Results | Key decisions, session logs | Baselines (first complete run) |
| CLI | Yes (thin enforcer) | No — `find` + `grep` |
| Git discipline | FEATURE.md in merge commits | REGRESSION.md in merge commits |

They are complementary. A feature's FEATURE.md may reference a regression
set that verifies it. A regression set's `depends` field may reference
features it depends on being completed. The link is loose — no forced
bidirectional binding.

## What This Is Not

- **Not a test framework.** It doesn't execute tests. It documents what
  to test and records what was found.
- **Not a replacement for pytest.** Automated tests stay in `tests/`.
  This is the layer above — methodology, prerequisites, diagnostic context.
- **Not a CI dashboard.** It captures baseline snapshots, not every run.
- **Not authoritative over test code.** If REGRESSION.md says one thing and
  the test suite says another, the test suite wins. Update REGRESSION.md.

## Cross-Validation

This convention is designed to be internally consistent with the features
convention (`.ai_partners/features/README.md`):

- Same philosophical foundation: filesystem as database, markdown as
  portable format, model-to-model context handoff
- Same sub-convention structure: `discuss/` for methodology collisions,
  `design/` for architecture decisions
- Same frontmatter style: YAML, kebab-case identifiers, P0-P3 priority
- Same git discipline: the convention document should be committed with
  the artifacts it governs

Key divergences are intentional:
- No CLI — regression sets don't need status management tooling
- No date in directory path — regression sets are long-lived entities
- Version counter — regression cases evolve; features don't need this
  because their lifecycle has an endpoint
- Baselines instead of session logs — regressions record verification
  results, not design decisions

---

*Designed through discussion between human engineer and DeepSeek V4
on 2026-06-15. Version 1. The first regression set (ghost-runtime) will
validate the convention.*
