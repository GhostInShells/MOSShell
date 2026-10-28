# ghoshell-agent-swarm

A prototype distribution in the MOSS monorepo: a **programmable swarm of ghosts**.

It sits above `ghoshell-moss`, never under it — the dependency runs one way, so
nothing in this package can quietly become part of the shell.

```
src/
  ghoshell_moss/            # the shell: channels, CTML, mindflow, matrix
  ghoshell-agent-swarm/     # this: tasks, swarm, collaboration as facts
    ghoshell_agent_swarm/
```

## Install (workspace member)

```bash
uv sync                      # from the repo root; installs this editable too
python -c "import ghoshell_agent_swarm as s; print(s.__version__)"
```

## Status

Skeleton. The abstraction is being written by hand, so that the code — not the
notes — decides what a task, a swarm and a member actually are.

Design notes live with the project, not here:
`.ai_partners/features/workstreams/2026/10/dsh-agent-swarm/` (`05-design-principles.md`
is the current spine).
