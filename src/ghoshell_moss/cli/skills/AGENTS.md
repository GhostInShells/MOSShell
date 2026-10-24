---
name: moss-skills
description: Governance for MOSS skills — distilled experience for *using* MOSShell. Read before authoring, pruning or reviewing a skill.
---

# MOSS Skills

A skill is **distilled experience**, written for somebody who is *using* MOSShell — a person or a
model — not for someone developing the framework itself. That change of identity is the whole premise:
the reader arrived with a task, not with the source tree.

The scope is not the manual. It starts there (install it, wire a channel, drive a screen) and keeps
going: how to talk with a child, how to talk with an elder, how to build a robot out of what is here.
A skill exists because somebody needed the same advice twice and the second time cost the same.

## Topology — three levels

Skills are graded the way `features specification` grades declarations. Every skill states its level
in the frontmatter; the level decides how much a skill is allowed to assume.

| Level | Answers | Example |
|---|---|---|
| **L0** | the direct method for one concrete task | audio etiquette; installing moshell; how to distribute a build |
| **L1** | a general method for a class of domain problems, from which N L0s can be grown | how to create a ghost for a user |
| **L2** | how to create new L1s — building a mechanism of your own | how to build your works system |

An L0 may be looked up mid-task. An L1 is read before starting a domain. An L2 is read when someone
is about to invent a governance of their own — it is the most expensive to write and the most
dangerous to write badly.

## Where skills sit among the other surfaces

| Surface | Carries |
|---|---|
| **code** | facts — the first self-explanation, and the only authority |
| **cli-flow** (`moss start`, `moss codex`, `moss --ai all-commands`) | the cognitive map: where things are |
| **features** | the evolution trail: why the project is shaped this way |
| **docs** | architecture philosophy (may later be retired, or migrate into works) |
| **skills** | distilled experience: the path that is not visible from the map |

> Building a product with moshell: the code is the components on the desk, the CLI is the bookshelf,
> and skills are the tool books on it.

A skill is legitimate only when the surfaces above it cannot carry the same thing — the map tells you
where to look, a skill tells you what you would not have guessed after looking.

## What goes in, what does not

Before writing one, answer six questions. Two of them are answered by somebody who was not in the room.

1. **Which scenario?** For a zero-context agent, what problem situation calls for this skill? Hand the
   scenario to a fresh agent and watch what it does without the skill.
2. **Which category?** Does it belong to an existing one? If not, that is a governance act — agree on
   it with the architect before creating it.
3. **Does it pay?** Does it markedly cut exploration cost, decision cost, or trial-and-error cost — or
   teach something the reader could not have known beforehand?
4. **Does it attract?** Given the problem, would an agent notice this skill fast enough to want to open
   it? A skill nobody opens is noise with a filename.
5. **Does it keep?** Will it still be useful one iteration stage later?
6. **What does it cost the pool?** Every skill adds to what must be read past. Does the pool stay
   legible, or does this tip it into a governance pass?

If any answer is no, do not write it.

## Philosophy — the fewest skills for the same body of knowledge

What a skills pool should contain, and nothing else:

1. the **L0** skills a zero-context agent cannot work without — the ones that visibly cut exploration
   and usage cost;
2. **L1** skills as few-shots — enough that a new domain can be reasoned by analogy, not so many that
   they become a catalogue;
3. **all** the **L2** skills — sharing the experience of building whole mechanisms on moshell.

And the negative rules, which matter more:

- ✗ **A skill never replaces facts.** Code is the truth; a skill that restates it will be wrong first.
- ✗ **A skill never replaces exploration.** It points at the path. A skill so complete that nothing has
  to be looked up adds entropy: it buries the path under its own description.
- ✗ **No domain-path knowledge.** Where things live in a particular project is `GROUND.md` /
  `AGENTS.md` business, not a skill's.
- ✗ **If one always-visible piece of code self-explains it** (the channel builder, the matrix), write no
  skill — in a composite situation, give the path to that code instead.
- ✗ **Skills are not liabilities, but they do rot.** Stale: delete. Noisy: delete. Duplicated: merge.
  Each iteration stage's surface review is the occasion to do it (`moss features surface`).

## What a SKILL.md carries

Four sections, in this order. Everything longer than this lives beside the skill, and `SKILL.md` links
to it — drawings, code, tools, long walkthroughs. No large binaries ride with a skill.

1. **Scenario** — one paragraph: the situation that brings a reader here. It is what an agent checks
   against its own intent before trusting the rest.
2. **Method** — the procedure as a state machine or a behaviour tree first, prose second. What comes
   first, what must be true before each step, what the forks are. Not a narrative.
3. **Exploration path** — where the resources are: which packages, which commands, what to look for
   when you get there. Point; never paste.
4. **How to trust this** — the verification strategy. What a reader can run or observe to confirm the
   skill still describes reality. This section is why a skill can be reviewed rather than believed.

## Frontmatter

The official `SKILL.md` protocol's own fields apply — do not restate its format here, follow it. Beyond
them a MOSS skill adds what a reader needs to decide whether it is still current:

```yaml
---
name: <leaf-name>
description: <the situation that needs this skill>
moss_version: <the version it was written against>
stage: <the iteration stage, e.g. stage3>
level: <L0 | L1 | L2>
---
```

Every other field follows the general `skill.md` convention.

## Validation — the skill must be found, then it must work

Two checks, both with an agent that was not in the room. A skill is unverified until both pass.

1. **Attraction.** Give a fresh agent a domain problem, without naming the skill. Watch whether
   `moss skills` leads it to the right one. If it never arrives, the description or the category is
   wrong — the content may still be fine.
2. **Delivery.** Give it the problem again, let it read the skill, and see whether it can walk the
   exploration path and produce a plan in the intended shape. Research only — no code, no hint from the
   author.

Fix or prune. Record in the body what case confirmed it.

## Governance

A skills pool is curated the way categories are: somebody has to be responsible for its size.

- **On creation** — run the six questions; the pool's entropy is part of the cost.
- **When stale** — a skill is a claim about one MOSS version. Checked and still true, the developer
  bumps `moss_version`; no longer true, it is fixed or pruned.
- **When poor** — a validation fails twice, or two skills teach the same thing differently.
- **At every stage end** — `moss features surface` opens the pass; prune, merge, re-level.

Categories are not a fixed whitelist. A category is a first-class governance object: propose one when
the pool has outgrown its current shape, and agree it with the architect.

 
