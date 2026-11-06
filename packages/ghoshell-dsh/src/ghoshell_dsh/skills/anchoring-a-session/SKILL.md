---
name: anchoring-a-session
description: Write down where a dsh session is — a coordinate that survives the process, read from the session's own log and costing nothing to take.
dsh_version: 0.2.0-rc.2
level: L1
verified: 2026-11-06 against dsh 0.2.0-rc.2 on macOS
---

# Anchoring a session

## Scenario

You are about to do something with a conversation that outlives the process holding it: hand a
job to another runtime, write a note against "what we knew here", come back tomorrow and read
the stretch that followed. What you need first is not a transcript — a transcript rots, and it
is wrong the moment it is copied. What you need is a **coordinate**: which session, which log
position. Everything else can be re-read from the thing that keeps being true.

`session.anchor()` produces exactly that. It is the dsh side of it; the `DshSessionAnchor` type
on the Python side is the same value written down.

## Method

```
session.anchor(seq=None, note="…")      ← None means "now"; a seq means "that earlier point"
        │
        ▼
   plugin verb ghoshell/anchor      (POST /api/ghoshell/anchor — an ordinary Remote envelope)
        │   ctx.sessions.get(sessionId)      → a live session, or an explicit failure
        │   session.snapshotEvents()         → the coordinate: the last event's seq
        │   session.requestHeader().config   → model / reasoningEffort
        │   session.header                   → agentPreset / cwd
        │   scan turn/start · session/title  → turn / title
        ▼
   DshSessionAnchor(session_id, seq, turn, model, ..., note, created)
        │
        ▼
   anchor.dump(".../x.anchor.yml")  ← flat YAML; from_file() gives back the same type
```

Four decisions behind the shape:

- **A coordinate, not a snapshot.** No message text is copied. `anchor()` does not call a model
  and does not touch `ctx.llm`: taking one costs a log read, so there is nothing that can half
  fail.
- **`seq=None` means "the session's start", not "unknown".** An empty log answers `null`. That
  is why the payload keeps `null` instead of inventing `0`.
- **The extra fields are for the reader.** `model`, `agent_preset`, `cwd`, `title` are read from
  the session and never used to reconstruct anything — an anchor whose metadata is wrong still
  points correctly.
- **Refusals are loud.** An unknown session and an out-of-range `seq` both raise a
  `DshRpcException` carrying the plugin's own message (`no live session …` / `seq N is outside
  the log (0..M)`), rather than clamping the seq and answering something plausible.

## Exploration path

- `src/ghoshell_dsh/v0_2_0/plugin/index.js` — the `anchor` route, and what it reads off the
  session (`snapshotEvents`, `requestHeader`, `header`).
- `src/ghoshell_dsh/v0_2_0/session.py` — `DshSession.anchor()`: the facade side, `seq`/`note` in.
- `src/ghoshell_dsh/v0_2_0/client.py` — `DshClient.remote()`: the bare Remote mouth this verb
  goes through, for verbs that do not have a pydantic parameter model yet.
- `src/ghoshell_dsh/blueprint/anchor.py` — `DshSessionAnchor` itself, including `dump` /
  `from_file` / `from_ref` (when you have a *span* and need one end of it).

## How to trust this

```bash
cd packages/ghoshell-dsh
PYTHONPATH=src:../ghoshell-processes/src \
    ../../.venv/bin/python src/ghoshell_dsh/skills/anchoring-a-session/sample.py
```

It boots a throwaway dsh in its own home with the plugin installed, creates one session, gives
it a durable event with `rename`, and then checks the coordinate — one `PASS` line per claim,
`ALL CHECKS PASSED` and exit 0 only if all of them hold. No model credentials are needed: an
anchor does not require a turn to have happened, and the sample proves exactly that by asserting
`turn is None` rather than inventing one.

**Verified 2026-11-06**, dsh `0.2.0-rc.2` on macOS: 14/14 checks passed, `seq` = the last log
position, `cwd`/`agent_preset` straight from the session header, no extra session directory
created by anchoring.
