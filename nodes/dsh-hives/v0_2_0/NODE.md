---
name: dsh_0_2_0_rc_2
description: dsh 0.2.0 hive — own a dsh process, run stateful cases on it, and read or talk to any step they took
category: hive
singleton: true
exec:
  command: python
  args: main.py --port 23080
check:
  command: python
  args: check.py
---

A dsh hive. It runs its own dsh web process, shows that page as a window on the screen, and
holds **cases**: stateful tasks written into a memento store, each driven by a dsh session
and readable afterwards.

## One directory = one hive

    .hive/.dsh/     the dsh home — profiles, sessions, storages
    .hive/cases/    the case store — one branch per case

The store is the roster: a case exists as long as its branch does, and after a restart the
cases are already there, idle. The whole `.hive/` is gitignored — a case carries absolute
paths, so none of it is portable.

## The channel

- `cases` — **always on**. The case control surface: `draft`, `bind`, `autopilot`, `pause`,
  `resolve`, `close`, `read`, `chat`, and the protocol library.
- `agents` — a **gated** child: the plain carrier collection, opened with `mount_child` when
  you need a session outside a case.
- `ui` — a **gated** child: the page itself, driven through dsh's own client API.
- `status` / `web_url` / `workspaces` / `create_workspace` — the hive's own facts and places.

## A case in four acts

1. **Draft.** `draft(definition)` creates the branch and writes its documents; it returns the
   path of `CASE.md`. Authoring the case means editing that file — the body is the brief.
   The definition carries a name, a description, a protocol (a template name or a full op
   table) and an owner (`ghost/<name>`).
2. **Bind.** `bind({case_name})` gives the case a carrier: it reads the documents, primes the
   session (protocol, brief, op table, the scratch directory's file list) and records an
   `init` commit. With no fields filled in it rebuilds what the last `init` recorded — the
   way back after a restart.
3. **Autopilot.** `autopilot({case_name, budget})` dispatches the loop and returns at once.
   Every finished turn is one step: its first line is an op word from the case's protocol,
   and it lands as one commit. A `continue` word drives the next turn; a `break` word stops
   the loop; a budget that runs out stops it too; a reply with no readable op line is a
   violation and stops it.
4. **Read.** `read(case)` lists the latest steps; `chat(case, coord, question)` asks one step
   a question with the context that produced it. `resolve` declares the case finished;
   `close` abandons it and archives its branch; `pause` stops a running loop and leaves the
   carrier bound.

Budgets are per run: the grant is written into a `run.start` commit and every step records
what its turn cost, so what is left is derived from the trace, never kept on the side.

## Running it

Started through the matrix node channel, on port 23080; pass `extra_args="--port N"` there to
move it. Launch arguments: `--home DIR`, `--profile NAME`, `--title NAME`. Install first —
see `INSTALL.md`.
