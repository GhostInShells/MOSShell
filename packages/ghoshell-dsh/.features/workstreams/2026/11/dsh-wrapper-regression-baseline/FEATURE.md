---
created: 2026-11-06
depends: []
description: 'The first regression baseline for the dsh wrapper: the core mechanisms
  stated as cases, and one runnable sample that produces a baseline anyone can compare
  against.'
milestone: dsh 0.2.0-rc.2
priority: P1
status: active
status_note: 'first baseline recorded: 2026-11-06_v1'
title: Dsh Wrapper Regression Baseline
updated: '2026-11-06'
---

# Dsh Wrapper Regression Baseline

## Motivation

The `v0_2_0` generation now carries thirteen capabilities — launch, connect, anchor, session
state, permission, off-record calls, tool calls, inject, fork, workspace attach, transcript,
model patches, provider listing. Each was built by measuring dsh, and each measurement lives only
in the commit message that happened to record it.

That is not a baseline. A baseline is a **set of cases anyone can run and compare against**, so
"is this still true?" costs one command instead of a day of re-measurement. Without it, every
future session re-derives the same facts, and the facts drift silently when dsh moves.

So this workstream builds two artifacts and keeps them in step:

1. a **regression set** — the core mechanisms as test cases with expected results, in
   `regressions/dsh-wrapper/REGRESSION.md`;
2. a **runnable sample** — `skills/verifying-against-a-live-dsh/`, one program that launches its
   own throwaway dsh, runs every case, and prints one `PASS`/`FAIL` line per case ID.

The first complete run of the sample is the baseline. Everything else (root causes, dead ends) is
recorded here, where a design decision belongs.

## Design Index

- `regressions/dsh-wrapper/REGRESSION.md` — the cases.
- `regressions/dsh-wrapper/baselines/` — first complete runs.
- `src/ghoshell_dsh/skills/verifying-against-a-live-dsh/` — the runnable sample.

## Key Decisions

**The baseline lists mechanisms, not implementation details.** A case says "an anchor reports the
log's last position and the turn containing it", not "the plugin reads `snapshotEvents()`". Tests
that pin internals break on every refactor and teach nobody anything; a case that pins *behaviour*
survives a rewrite and still means something.

**Verification scripts must be self-contained.** The probe scripts that produced these
measurements lived in the repository-level `.playground/`, hard-coded a port and a token, and
belonged to one machine. They could not be handed to anyone. The sample creates its own
`DSH_HOME`, installs the plugin, picks a free port, discovers its own token — so "run this" is
the whole instruction, and `.playground/` goes back to being a scratch area rather than an
asset.

**One sample, many cases.** A case-per-script split would mean one dsh launch per case — slow,
and it hides ordering effects (permission switches and forks interact). One process, cases in
dependency order, is both faster and more honest about state.

**Two kinds of case, marked in the table.** `P0` cases must pass for the baseline to be `PASS`;
`P2` cases document behaviour that is *known and accepted* — e.g. that a structured call may come
back with a tool the caller did not ask for. A baseline that hides known limitations is a lie with
a date on it.

## Exploration paths

**Where the numbers came from.** Every expected result in the table was measured on a live
dsh 0.2.0-rc.2 before it was written down: cache reads, prefix lengths, log shapes. Cases that
merely assert "no error" were not kept — they pass for the wrong reasons.

**What was dropped.** Three mechanisms were explored and abandoned before the baseline was
drawn: running a tool off-record through `ctx.tools` (couples the plugin to every agent preset),
stripping the session's tools for a call (measured: `cacheReadTokens` 9088 → 1024 on every call),
and per-scope tool restrictions (memory-only, so "disable" would be a weaker word than it sounds).
A baseline should carry what the layer *does*; decisions about what it refuses to do belong here.

## Methods

- **Throwaway home, real launch.** The sample always starts its own dsh through `DshLauncher`
  rather than attaching to a running one. Attaching made the earlier probes depend on a machine
  state that no one else had.
- **Cases print their own verdict.** Each case prints `PASS`/`FAIL` with the case id and the
  measurement it made, so a baselining run and a comparison run produce the same readable diff.
- **No hidden global setup between cases.** Where a case needs state (a forked child, a switched
  permission), it creates it and the later cases are written to tolerate it.

## Implementation Notes

- A case may pass while documenting a limitation — `chat_off_record` returning a tool the caller
  did not ask for is expected behaviour today, and the case asserts exactly that rather than
  pretending the feature is complete.
- Cases that need model credentials are marked `BLOCKED` when no key is present, not skipped
  silently: a baseline that quietly omits half its cases is worse than one that says so.