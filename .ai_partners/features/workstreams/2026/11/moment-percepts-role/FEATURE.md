---
created: 2026-11-11
depends: []
description: 'Split "what came in" from "what came back": rename the Moment''s percepts
  slot to inputs, add a role-classified percepts slot, and give every impulse an identity.'
milestone: null
priority: P1
status: in-progress
status_note: 'kernel lands: inputs/percepts split, role on signal+impulse with vocabulary
  and tag-name validation; nodes'' async callbacks still unclassified'
title: Moment Percepts Role
updated: '2026-11-11'
---

# Moment Percepts Role

> Use `moss features set-status moment-percepts-role <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

A Moment has two big slots today: `echoes` (what this ghost did last round) and
`percepts` (what arrived, rendered as `<inputs>`). Everything that is not an echo is
flattened into the same place — a user speaking, a tool result, another model's answer,
a node reporting its own inbox. The model cannot tell them apart, and the framework's
own contract is broken: `Moment.percepts` is documented as *source-keyed, same source
overwrites (dedup)*, but the impulse path (`Mindflow` → `inject_percepts(*messages)` →
`percepts["MomentsInjectedPercepts"]`) throws the source away, so the key never dedups
and two deliveries from the same source in one frame overwrite each other.

The word for that mechanism is `percepts`, and it is the right word: these are the
things this ghost perceives through its instruments, as opposed to `inputs` — what the
world hands it right now. The two are synchronous concepts (`echoes` = my own action
returning, `inputs` = the world speaking). The classification exists because MOSS runs
group chats, fully asynchronous tools, and dozens of perception sources; the ChatGPT-era
role taxonomy rotted for exactly those reasons.

Key quote (thirdgerb, D11):

> 群聊, 全异步, 以及像你所处的 ghost 系统 (有无数感知源), 都导致了原始 role 定义的腐烂.

So: `role` is a MOSS-internal convention — simple, extensible, and **not** constrained
by the LLM role field. Empty means "no classification, it is an input". Non-empty routes
the delivery to its own bucket, so an email node's inbox never mixes with a human's
sentence.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

1. **Swap the slots.** `Moment.percepts` becomes `Moment.inputs` (unchanged shape and
   behavior, only the name), and a new `Moment.percepts: dict[role, dict[key, list[Message]]]`
   takes the old name. The old name was doing two jobs — "perception" in general and
   "the world's input" specifically — and the render slot was already called `<inputs>`,
   so the field was lying about itself.
2. **The dedup key is identity, not source.** `inputs[key]` and `percepts[role][key]`
   overwrite by key **in place, preserving order**. `impulse.id` is the natural key for
   deliveries; a source name is the natural key for a long-lived sensor. What the key
   must never be is "the place it came from" as a category — that is what role is for.
3. **`role` rides the signal and the impulse**, default `""`. Empty → `inputs`;
   non-empty → `percepts[role]`. The impulse comment says where it lands, so a producer
   reading `Impulse` learns the routing without reading the renderer.
4. **`buffer_impulse` replaces message flattening.** The observer keeps the whole impulse
   (id, role, messages) instead of only its messages; `_peek_moment` routes by role. This
   is the only way to keep the Moment shape untouched and still classify at render time.
5. **Rendering keeps the big-slot vocabulary.** `<echoes>` / `<inputs>` /
   `<percepts><role>…</role></percepts>`. The slot is the container; the role is the tag
   inside it. No `role="…"` on the container — the model should read sections, not
   attributes.
6. **`role` is classification only.** It decides which bucket a delivery lands in, never
   its priority. A human's input is high-priority because the input signal carries that
   tier, not because its role says `user`; keeping the two apart means a producer cannot
   gain arbitration weight by naming itself.
7. **The impulse is the only role the moment routes by.** `Signal.role` is syntax sugar
   (`None` = unclassified, and `None` and `''` are equivalent); the nucleus that builds
   the impulse owns the classification and may keep, override or drop it. Role conflicts
   inside an aggregating nucleus are not reconciled — the newest signal's role wins, same
   as `description`. A producer that needs two roles emits two impulses.
8. **The vocabulary is `user / tool / mind / agent / host`** — an open convention, not a
   closed enum. A role names *whose act the delivery represents*, not who physically
   emitted the bytes: ASR is a capability, but the act is the human's, so it is `user`.
   `host` names the host that runs the ghost (framework notices, aborts, model changes);
   it is deliberately not `system`, because `system` and `developer` are LLM chat roles
   that mean *instruction authority*, and a framework notice must not be read as one.
   `input` is not a role: the default container already encodes "unclassified".
9. **The role's spelling is the XML-name rule.** It is rendered as a tag
   (`<percepts><role>…`), so the name must be renderable: letters, digits and
   underscores, not starting with a digit, not the reserved `xml` prefix, and not one of
   the container tags (`inputs`, `percepts`, `echoes`, `moment`, `hint`, `executing`,
   `dynamic_context`). Validation happens at signal / impulse construction, so a bad
   name fails where it is written rather than in the rendered frame.

## Exploration paths

- `returns` / `deliveries` / `reports` were proposed for the third slot and rejected:
  they name the *timing* correctly but are weaker than `percepts`, which names the
  faculty. `feedback` was rejected as looking like a role itself.
- `slot` was proposed and rejected outright — it describes a position in a layout, not
  the nature of what is delivered, and the design brief is `user / system / tool`-like
  semantics.
- Resurrecting `perspectives` (the removed Gemini-3-era field) was rejected: the field
  was removed because landing it early cost more than it paid; the concept now lives on
  the impulse as `role` instead of as a Moment field.

## Methods

- The rename is mechanical but wide: ~35 call sites across 9 files, zero in tests
  (tests go through the API). Do it in one commit so the tree never sits in the half-named
  state.
- `inputs_messages()` already existed as an alias with `with_hint` / `with_command_executing`
  options; after the swap it becomes the real flattening method and `percepts_messages()`
  moves to the new slot.

## Remaining

- **The out-of-box nodes are not classified yet.** The kernel now carries `role`, but every
  async callback a shipped node raises still defaults to empty and therefore lands in
  `inputs`. Before this workstream is completed, the nodes that deliver asynchronously
  (the hive's off-record answers, the browser/screen callbacks, the calendar reminders, any
  node that reports its own inbox) must set their role, or the classification exists in the
  kernel and changes nothing in practice.
- The Dolores memento chat is the first caller: it returns a receipt at once and delivers
  the off-record answer later as a signal with `role="mind"`, so the whole path
  (signal → nucleus → impulse → `<percepts><mind>`) is exercised in production by it.
  The earlier plan to call this role `assistant` was dropped — the LLM taxonomy is exactly
  what this slot exists to escape.

## Implementation Notes

- `Moment` has no storage position yet, and the context is forward-compatible, so no
  migration shim is needed for old JSON.
- The impulse buffer dedups by `id`: a later package from the same impulse replaces the
  earlier one **in place** (order kept), which is the contract the old source-keyed dict
  claimed but never delivered.