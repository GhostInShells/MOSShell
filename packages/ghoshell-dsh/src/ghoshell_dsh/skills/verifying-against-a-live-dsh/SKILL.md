---
skill: verifying-against-a-live-dsh
verified: 2026-11-06 against dsh 0.2.0-rc.2 on macOS
---

# Verifying against a live dsh

## Scenario

You changed something in `v0_2_0` — the plugin, the protocol layer, the launcher — and you need
to know whether the *mechanisms* still behave. Not "does it import", not "do the unit tests pass":
those never started a dsh. The question is whether a caller can launch one, drive a session, and
get what this package promised.

This is the executable half of `regressions/dsh-wrapper/REGRESSION.md`: the case ids printed here
are the case ids in that table, and one full run **is** a baseline.

## Method

```
own DSH_HOME (throwaway) ──▶ install this generation's plugin ──▶ free port
        │
        ▼
DshLauncher(config)  ── async with ──▶ token discovered ──▶ mux connected ──▶ ready
        │
        ▼
sixteen cases, dependency order, one dsh:
  launch → providers → session → policy → switches → anchor → errors
  → transcript → [model cases | BLOCKED] → fork → workspace
        │
        ▼
one PASS / FAIL / BLOCKED line per case id, then a count
```

Three rules the earlier probe scripts broke, and this one keeps:

- **Self-contained.** No hard-coded port, no token argument, no dependence on a dsh someone else
  started. If it needs state, it makes that state and says so.
- **One process, many cases.** A fresh dsh per case would hide the ordering effects — the fork
  inherits a permission that an earlier case switched.
- **`BLOCKED` is a real verdict.** Model cases need credentials a *human* registered in a profile.
  The sample never looks for an API key; without `--with-model-cases` those cases report blocked,
  and a baseline that says "5 blocked" is worth more than one that silently ran 11.

## Exploration path

- `src/ghoshell_dsh/skills/verifying-against-a-live-dsh/sample.py` — the run itself, and the case
  ids.
- `regressions/dsh-wrapper/REGRESSION.md` — what each case is for and what a pass looks like.
- `src/ghoshell_dsh/v0_2_0/launcher.py` — `DshLauncher.__aenter__` is the readiness contract; if
  `TC-001` fails, everything after it is noise.
- `src/ghoshell_dsh/v0_2_0/plugin/index.js` — every verb the protocol layer calls lives here; a
  case failing almost always means this file and the Python side disagree about a field name.

## How to trust this

```bash
cd packages/ghoshell-dsh
# the baseline as it stands: 11 pass, 5 blocked (no credentials registered here)
PYTHONPATH=src:../ghoshell-processes/src \
    python src/ghoshell_dsh/skills/verifying-against-a-live-dsh/sample.py

# a full run, in a profile where a human has registered credentials
... sample.py --home <that home> --keep-home --with-model-cases
```

Exit code is 0 only when no case failed; `BLOCKED` and `PASS` both exit 0, so the program can be
used as a pre-commit gate without demanding credentials from everyone.

**Verified 2026-11-06**, dsh `0.2.0-rc.2`, macOS: `11/16 passed, 5 blocked` — the five blocked
being the model cases plus the two fork cases, which need a completed turn that only a model can
produce.
