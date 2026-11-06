---
title: dsh wrapper — core mechanism baseline
version: 1
status: draft
priority: P1
created: 2026-11-06
updated: 2026-11-06
scope: integration
depends: []
description: >-
  What the v0_2_0 generation promises, one case per mechanism: launch, connect, anchor, state,
  permission, off-record calls, tool calls, inject, fork, workspace, transcript, resources.
---

# dsh wrapper — core mechanism baseline

The case set for one question: **can a caller drive a dsh through this package and get what the
package says it will get?** Every case pins a behaviour that was measured against a real dsh
0.2.0-rc.2 before it was written here — not an intention.

One runnable sample executes all of them: `src/ghoshell_dsh/skills/verifying-against-a-live-dsh/`.
Case ids in the sample's output are the ids below.

## Exploration Index

```
moss codex get-interface ghoshell_dsh.blueprint.agent:AgentSession
moss codex get-interface ghoshell_dsh.v0_2_0.session:DshSession
moss codex get-interface ghoshell_dsh.v0_2_0.launcher:DshConnection
git log --oneline -10 -- packages/ghoshell-dsh/src/ghoshell_dsh/v0_2_0/plugin
```

## Methodology

Fully automated, one process, no hardware. The sample creates a throwaway `DSH_HOME`, installs
this generation's plugin into it, launches its own dsh on a free port, and runs the cases in
dependency order against that one instance.

Two consequences worth knowing before reading the results:

- **Order matters and is intentional.** Permission switches, the fork and the workspace
  registration leave state behind; later cases are written to tolerate it rather than to assume a
  clean slate.
- **Cases may pass while documenting a limitation.** `TC-012` asserts that a call with an extra
  tool declaration *succeeds* — not that the model picks that tool. Today it often does not, and
  pretending otherwise would turn a known limitation into a false guarantee.

## Prerequisites

- `dsh` on `PATH`, version matching `DSH_VERSION` (`0.2.0-rc.2`).
- Python with `httpx`, `pydantic`, `websockets`.
- **Model credentials are a human step.** A profile must have credentials registered before a
  model can answer; the sample never looks for a key and never assumes one exists. Without
  `--with-model-cases`, every case that needs a model reports `BLOCKED`.

Run:

```bash
cd packages/ghoshell-dsh
PYTHONPATH=src:../ghoshell-processes/src \
    python src/ghoshell_dsh/skills/verifying-against-a-live-dsh/sample.py
```

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|-----------------|
| TC-001 | P0 | The launcher reaches readiness, not just spawn | run the sample | Token discovered from stdout and mux connected before the body runs |
| TC-002 | P0 | Entering a connection means it is usable | run the sample | A Remote verb answers immediately after `async with` — no 401, no manual wait |
| TC-003 | P0 | Model resources are listed, live apart from declared | run the sample | At least one `live` provider; totals include declared-but-unconfigured routes |
| TC-004 | P1 | A session can be created and its state read coherently | run the sample | `attached=True`, `turn=None` on a fresh session, `running`/`hasPending` both false |
| TC-005 | P0 | A session knows where it may act | run the sample | `sandbox` is `read-only` or `workspace-write`; `sandboxRoot` equals the session cwd |
| TC-006 | P0 | Permission switches, and full access is refused | run the sample | read-only → workspace-write round-trips; `danger-full-access` raises "not reachable" |
| TC-007 | P0 | Switching permission does not touch the conversation | run the sample | Transcript is unchanged and empty after three switches |
| TC-008 | P0 | An anchor is a coordinate read from the log | run the sample | `seq` ≥ the rename event's seq, `title` is the one just set, `turn` derived from the log |
| TC-009 | P1 | An anchor refuses a position that does not exist | run the sample | `seq=10**6` raises "outside the log"; `seq=0` is honoured |
| TC-010 | P1 | A reading excludes results, reasoning and injected context | run the sample | Transcript renders without `tool/result` content; counters describe what was rendered |
| TC-011 | P0 | An off-record call answers and records nothing | `--with-model-cases` | Text is non-empty; the conversation's text is unchanged; no session directory was created |
| TC-012 | P2 | Extra tool declarations are accepted, and tool calls are reported | `--with-model-cases` | The call succeeds; `tool_calls` may name a session tool rather than the declared one — **known limitation, not a failure** |
| TC-013 | P1 | An injected message queues without waking the agent | `--with-model-cases` | `woke=False`, `messageId` non-empty, no turn was started |
| TC-014 | P0 | A fork continues the history in a different directory | `--with-model-cases` | Child's `sandboxRoot` is the new directory; child's turn equals the parent's; `seededEvents` > 0 |
| TC-015 | P0 | A fork inherits its parent's permission | `--with-model-cases` | Child reports the parent's sandbox mode and approval policy |
| TC-016 | P1 | A session registers under the workspace that owns its directory | run the sample | `attach_workspace()` returns an id, and a second call returns the same one |

## Execution Notes

- **`BLOCKED` is not `FAIL`.** Model cases are blocked whenever no human has registered
  credentials; that is a property of the environment, and a baseline should say so plainly.
- **A fork needs a completed turn.** dsh refuses to cut at a `turn/start`, so `TC-014`/`TC-015`
  report `BLOCKED` in a run without model cases even though nothing about forking is broken.
- **Do not assert on `workspaces()` for `TC-016`.** A workspace created after the follow stream's
  baseline snapshot does not push itself into that list; the registry's own reuse of the id is the
  stable witness.
- **Cache numbers are not cases.** `cacheReadTokens` moves with model, prefix and patch; it was
  used to *decide* design (adding tools is free, removing them is not) but pinning an exact
  number here would make the baseline fail for reasons that are not regressions.
