---
name: launching-a-dsh
description: Start a dsh server from Python, land on a port, and know when it is actually ready — the first step of anything else that uses this package.
dsh_version: 0.2.0-rc.2
level: L0
verified: 2026-11-06 against dsh 0.2.0-rc.2 on macOS
---

# Launching a dsh

## Scenario

You are writing something — a ghost, a tool, a test — that needs a dsh server of its own. Not a
dsh that somebody else started: a child process this program owns, on a port this program
picked, in a home directory this program can throw away afterwards. The question you are about
to get wrong is *when it is ready*: `dsh` printing a URL is not the same moment as the mux
websocket accepting connections, and the gap is where "connection refused" comes from.

## Method

```
pick an isolated DSH_HOME ──▶ pick a free port ──▶ set DSH_WEB_NO_OPEN
        │
        ▼
DshLauncher(config) — constructing it touches nothing: no PATH lookup, no spawn
        │
        ▼
async with launcher:            ← this line is the whole procedure
        │   spawn dsh ──▶ read stdout ──▶ find `token=…` ──▶ authorize ──▶ connect mux
        │                                                                     │
        │                                                        ready event fires
        ▼
    (body runs only once both are true; a timeout raises TimeoutError instead of hanging)
        │
        ▼
exit the block ──▶ SIGINT ──▶ grace ──▶ SIGKILL the process group
```

Three facts behind the shape:

- **Constructing does not launch.** `DshLauncher(DshLauncherConfig())` resolves no binary and
  starts no process; binary resolution is deferred to `__aenter__`, this sample trusts `PATH`
  on the machine that wrote it.
- **Readiness is two conditions**, not one: the web token was discovered *and* the mux
  websocket connected. `__aenter__` waits for both, with `readiness_timeout` (default 30s) as
  the ceiling.
- **The token is learned from stdout**, unless you pass it. That is why the launcher owns the
  process — an externally started dsh gives you no pipe to read it from.

## Exploration path

- `src/ghoshell_dsh/v0_2_0/launcher.py` — `DshLauncherConfig` (what you can steer),
  `DshLauncher.__aenter__` (the readiness wait), `_spawn_dsh` (the argv actually used).
- `src/ghoshell_dsh/v0_2_0/__init__.py` — the public names. `DSH_VERSION` states the dsh
  version this generation was written against; it is a declaration, not a runtime gate.
- The argv is `dsh --profile <profile> --port <port> [--no-open] <extra args>`; `--profile`
  is the *process* profile, not a dsh config file.

## How to trust this

```bash
cd packages/ghoshell-dsh
# The package not installed in the current interpreter? Point at the sources instead of
# touching the environment (never `uv sync` inside this directory — it would prune the
# workspace environment down to this package's needs):
PYTHONPATH=src:../ghoshell-processes/src \
    python src/ghoshell_dsh/skills/launching-a-dsh/sample.py              # isolated home, free port
PYTHONPATH=src:../ghoshell-processes/src \
    python src/ghoshell_dsh/skills/launching-a-dsh/sample.py --keep-home  # keep the home to inspect
```

It prints one line per landmark — port, token discovered (redacted), mux connected — and exits
0 only if the readiness wait returned; a timeout exits non-zero with the reason. If it exits 0,
the launch path works. If it hangs, readiness never happened: check `dsh --version` against
`DSH_VERSION` first, then the profile.

**Expected noise.** Before `READY` the reconnect loop may refuse the port a couple of times
while the server binds. That is the mechanism working, and it logs below WARNING; what you see
above WARNING is real. If a connection that had already worked drops, *that* is a warning.

**Verified 2026-11-06**, dsh `0.2.0-rc.2` on macOS: `READY`, `stopped cleanly`, exit 0. The
same run confirmed the side-effect claims — a fresh *empty* `DSH_HOME` is enough (`dsh` does not
need a pre-seeded profile), `~/.dsh` was untouched, no orphan process, the throwaway home was
removed.
