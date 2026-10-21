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

### Two kinds of skill

**Task skill** — an executable procedure for a composite task: what to do first, what next,
how components cooperate. The original kind.

**Mechanism skill** — orientation into a core mechanism an agent must use or extend (for
example, how capabilities are wired across cells, or how a screen projects them). A mechanism
skill is not an API catalogue and not an implementation. It states the problem, the motive,
the shape of the solution, and where the framework already explains it. Its value is judgment
and navigation; executable detail stays in the interface reflection.

A mechanism skill **may point at openbox implementations as reference direction** — the nodes
and services shipped with MOSS are the worked examples. Pointing is not copying: name the
files or packages, say what to look for in them, and defer interface truth to
`moss codex get-interface`. Never paste implementation or line numbers into the skill.

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
   not a single command or a single interface. For a mechanism skill, the composite action is
   reasoning about and extending a mechanism that spans several components; it must still end
   in something the reader can run to confirm the mechanism works.
2. **Needed at task start?** If it is only a lookup need, `moss docs` already covers it.
3. **Stable within six months?** The abstractions it depends on must be battle-tested, not under
   active evolution. An unstable domain written as a skill is a self-made stale source.
   Acceptance in real use — not a green unit test — is what makes a mechanism stable enough.

If you cannot answer **yes** to all three, do not write it.

## Anti-patterns (historically validated)

- ✗ A skill tacked onto a component — the component's interface is already the prompt.
  A mechanism skill is not exempt: it must add judgment and navigation the interface cannot.
- ✗ Step-level operational skills — CLI help + `moss --ai all-commands` already self-explain.
- ✗ Decision / architecture discussion — that belongs to `features/` / `docs`.
- ✗ Interface usage notes — `moss codex get-interface` is the authoritative, non-rotting source.
- ✗ Copying docs into a skill — skills point at executable procedures; pure reading stays in docs.
- ✗ Copying openbox code into a skill — reference direction only; the source stays authoritative.

## Writing discipline

- **Language: English.**
- **YAML frontmatter is required** — see Meta format below.
- **Reference interfaces, do not copy** — `moss codex get-interface <module>` is authoritative.
- **Do not hardcode specifics** — enum members, CLI literals, CTML command literals all drift;
  point readers to reflection (`moss codex`) for current values.
- **Describe the shape of the composite behavior** — the unique value is orchestration knowledge
  (what first, what next, how components cooperate), not an interface catalogue.
- **Point at openbox source as direction, not as text** — name paths/roles and what to look for.
  Do not paste code, line numbers, or output. Never point at `FEATURE.md`, workstreams, or
  decision numbers: those are reverse indexes, not authority, and they do not ship.
- **Mechanism skills distill, they do not index** — problem, motivation, solution shape, where
  the framework already explains it, reference directions with one executable acceptance,
  version and identity. The knowledge surface belongs to the framework (`moss codex blueprint`,
  `channel_builder`, interface reflection); a skill does not curate a reading list in its place.
  Its value is the conceptual key that is easy to miss, not a table of contents. Keep
  implementation out; if a section needs code to make sense, the interface is under-documented.
- **Mark provenance for expiry** — `moss_version`, `verified`, `author`, and `evidence` in the
  frontmatter. A skill whose interface no longer matches reflection is a review candidate.
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
author: <model identity>                        # who wrote it; real harness identity
evidence: <what real use confirmed it>          # e.g. live runs, dates, participating nodes
---
```

The version and identity fields exist so content can be **stale-checked and pruned** later.
A skill whose `moss_version` no longer matches the current MOSS version, or whose referenced
interfaces no longer reflect, is a candidate for review. Prefer pruning a rotted skill over
letting it quietly become false.

## Validation — the zero-context handoff

A skill is written for an agent that was not in the room. Validate it that way before treating
it as verified:

1. **Design a proposition, not a quiz.** Write a concrete task that requires the mechanism —
   the intended outcome, not the roles or the answer. "Make this node's live metrics consumable
   by other cells" is a proposition; "write a ServiceDeclaration" is a quiz.
2. **Use a side agent.** Start a fresh agent with no access to the authoring conversation. Give
   it the task and the skill path; tell it to read the skill first, then research the framework
   surface the skill points to. Do not hand it the author's notes, FEATURE documents, or the
   reference implementation's design rationale.
3. **Ask for an approach.** The agent returns how it would build it, which components it would
   touch, and where the skill was unclear. Research only — no code.
4. **The author judges.** Did the approach land on the intended shape? Did it invent a parallel
   mechanism, copy an implementation, or miss a boundary? A plausible plan that violates the
   mechanism is a failure of the skill, not of the agent.
5. **Fix or prune.** Fill the missing key, or conclude the skill cannot carry the mechanism and
   remove it. Record the validation in `evidence`.

This tests the delivery surface, not the agent. The agent's self-assessment is not the verdict.

## Categories (second level)

Skills are organized one level deep by category:

```
skills/<category>/<name>/SKILL.md
```

The category is **curated, not free-form**. Current whitelist:

| Category | Scope |
|---|---|
| `audio` | audio device routing, playback, capture, voice |
| `matrix` | cell-level capability wiring, service kinds, mesh communication |
| `screen` | screen projection: window semantics, adoption, layout, foreground gestures |

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
