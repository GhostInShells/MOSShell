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

- **prototype** = mythic character name (Atom / Dolores) — the architecture slot.
- **instance** naming has two regimes, following the openbox / project-own split (KD1):
  - **openbox samples: the model name** — `deepseek` (Dolores), an honest description of what runs it;
    `echo` (Atom) predates this convention.
  - **the project's own ghost: an individual name** — `moss` is the first generation's individual name,
    era-bound: it is neither a model name (an individual may change its base within one lifetime) nor a
    project name. It is not reused across generations — the next generation takes a new name.

Reasoning and the rejected alternative (imposed sameness): KD1–KD4 of the `openbox-ghosts` workstream.

## Privacy

`.memento/` and `.dsh/` contain real conversations / keys / people, and are **not open-sourced for now** —
what ships is the mechanism, not the content.
