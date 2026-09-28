---
name: Ghosts
description: ghost instances, roles, and naming conventions of this repository
---

# Ghosts

`.moss/ghosts/` holds each ghost's persistent home: `workspace/ghosts/<name>/` (the name matches the directory exactly, case-sensitively).

## Instances

| instance | prototype | role |
|---|---|---|
| `echo` | Atom | out-of-the-box minimal reference baseline |
| `deepseek` | Dolores | out-of-the-box sample, model-named (honest description) |
| `moss` | Dolores | the project's own ghost (老莫) — model-side owner carrying the project's trajectory |
| `none` | — | placeholder home when no ghost is running |

## Naming

- prototype = mythic character name (Atom / Dolores); instance = model name (deepseek).
- the project's own ghost is an **individual** (era-bound): `moss` is the first generation's individual
  name; it is not reused across generations — the next generation takes a new name.

## Privacy

`.memento/` and `.dsh/` contain real conversations / keys / people, and are **not open-sourced for now** —
what ships is the mechanism, not the content.
