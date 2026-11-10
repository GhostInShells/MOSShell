# Install

This node owns a dsh process and a case store. The environment below is the requirement;
`check.py` is the probe that refuses to launch when it is not met.

## 1. The engine in the moss environment

The node runs on the interpreter that spawns it (`sys.executable`) — in this repo, the
project's `.venv`. That environment must have the `[dsh]` extra:

    uv sync --all-extras

Use `--all-extras`, not `--extra dsh`: a selective sync prunes the host layer this ghost
runs on. (A consumer install `pip install ghoshell-moss[dsh]` becomes available once
`ghoshell-dsh` is published; in this workspace the extra resolves from
`packages/ghoshell-dsh`.)

Verify:

    .venv/bin/python -c "import ghoshell_dsh; print(ghoshell_dsh.v0_2_0.DSH_VERSION)"

## 2. The dsh executable

    dsh --version      # 0.2.0-rc.2

`PATH` is enough. To pin a specific install, set `DSH_BINARY` to its absolute path.
`check.py` resolves it the same way, so a broken pin fails before spawn.

## 3. Initialize the hive

Run from this directory (or pass `--home`):

    .venv/bin/python -m ghoshell_moss.deepseek_harness.v0_2_0.node --install

It creates one directory, the hive:

    .hive/.dsh/    DSH_HOME — dsh profiles, sessions, storages
    .hive/cases/   the memento store; every case is one branch under it
    .gitignore     runtime ignores (written only when absent)

The whole `.hive/` is gitignored on purpose: a case embeds absolute paths (its workspace,
its scratch directory, the session that produced each step), so a case is machine-local by
construction and must never be committed.

A missing `.dsh` is provisioned by dsh itself (`--dump-config`, no boot). An existing one
is left alone except for the plugin bundle: settings / storages / workspace.json are never
rewritten. The command is idempotent.

## 4. Mark it installed

    moss nodes install nodes/dsh-hives/v0_2_0

The marker is `.installed` in this directory (gitignored). The node will not spawn until
it exists; `check.py` then verifies the environment and the hive before every launch.

## Starting

The node is started through the matrix node channel, not from this file. The node manager
runs `check.py` first; a failure there is the reason it did not start.
