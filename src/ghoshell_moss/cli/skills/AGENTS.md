---
name: moss-skills
description: Governance for MOSS skills — action-oriented, deliverable skills for composite tasks. Read before authoring a skill.
---

# MOSS Skills

Skills are **action-oriented, deliverable knowledge for composite tasks** — not documents.
An agent working with MOSS picks one up at the start of a task via `moss skills list` / `read`.

## Role — skills serve agents *using* MOSS

A skill answers: *"how do I use MOSS to accomplish a composite task?"* — drive the CLI, wire
channels, route audio, operate a shell/ghost, compose nodes.

It does **not** cover how to *develop* MOSS (kernel internals, adding channels/contracts,
architecture). Development guidance lives in `moss docs`, `moss codex`, and the `CLAUDE.md`
family — not here.

Corollary: a skill must be **generic and durable** — not an artifact of one feature iteration.

## Where skills sit

MOSS self-explanation is layered; skills fill the **delivery** layer:

| Layer | Carries | Example |
|---|---|---|
| L0 code | first self-explanation principle | the source itself |
| L1 CLI-flow | cognitive map | `moss codex` / `moss start` / `moss --ai all-commands` |
| L2 dir doc | what a directory carries | `<subdir>/AGENTS.md` |
| L3 docs | systematic exposition | `moss docs read` |
| **skills** | **action entry for composite tasks** | **this directory** |

A skill is legitimate only if the layers above cannot cover it, **and** it is a real
composite-task entry an agent needs at task start.

## Entry criteria (answer all three before writing)

1. **Composite action?** Coordinates several components/systems with an executable procedure —
   not a single command or a single interface.
2. **Needed at task start?** If it is only a lookup need, `moss docs` already covers it.
3. **Stable within six months?** The abstractions it depends on must be battle-tested, not under
   active evolution. An unstable domain written as a skill is a self-made stale source.

If you cannot answer **yes** to all three, do not write it.

## Anti-patterns (historically validated)

- ✗ A skill tacked onto a component — the component's interface is already the prompt.
- ✗ Step-level operational skills — CLI help + `moss --ai all-commands` already self-explain.
- ✗ Decision / architecture discussion — that belongs to `features/` / `docs`.
- ✗ Interface usage notes — `moss codex get-interface` is the authoritative, non-rotting source.
- ✗ Copying docs into a skill — skills point at executable procedures; pure reading stays in docs.

## Writing discipline

- **Language: English.**
- **YAML frontmatter is required** — see Meta format below.
- **Reference interfaces, do not copy** — `moss codex get-interface <module>` is authoritative.
- **Do not hardcode specifics** — enum members, CLI literals, CTML command literals all drift;
  point readers to reflection (`moss codex`) for current values.
- **Describe the shape of the composite behavior** — the unique value is orchestration knowledge
  (what first, what next, how components cooperate), not an interface catalogue.
- **Scripts live beside the skill** — `<name>/SKILL.md` is the entry; helper scripts may sit in
  the same directory (see Scripts).

## Meta format

Every `SKILL.md` carries frontmatter:

```yaml
---
name: <leaf-name>                               # skill identifier; matches the directory name
description: <when an agent needs this skill>   # discovery/display signal; be specific
moss_version: <version>                         # the MOSS version verified against, e.g. beta2
platform: [<platform>]                          # optional; environment constraint, e.g. [macos]
verified: <YYYY-MM-DD>                          # last date the content was confirmed correct
---
```

The version fields exist so content can be **stale-checked and pruned** later. A skill whose
`moss_version` no longer matches the current MOSS version is a candidate for review.

## Categories (second level)

Skills are organized one level deep by category:

```
skills/<category>/<name>/SKILL.md
```

The category is **curated, not free-form**. Current whitelist:

| Category | Scope |
|---|---|
| `audio` | audio device routing, playback, capture, voice |

**Creating a new category requires agreeing with the architect first** — do not invent one.
The category directory name is the category; it is not repeated in frontmatter.

## Scripts

A skill may ship helper scripts in its own directory:

```
skills/<category>/<name>/
  SKILL.md
  <script>            # optional
```

Scripts assume the **MOSS python environment** drives them — invoke via `uv run` or the
project's `.venv` python, never a bare system python.

## Delivery

- `moss skills list` — discover skills (glob + frontmatter index).
- `moss skills read <path>` — read a skill (short name or full path).
- `moss skills list --root <path>` — explore skills in any directory.

Skills live at a standard location and are discovered by the agent's own harness; MOSS does
**not** inject them into the agent context.

> `moss skills recall` was removed. It required `LLMFuncs` (an LLM config), which dead-ended
> cold-start / unconfigured workspaces. Skills must not depend on a second config mechanism.
