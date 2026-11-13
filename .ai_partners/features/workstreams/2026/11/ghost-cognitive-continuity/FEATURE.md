---
title: Ghost Cognitive Continuity
status: in-progress
status_note: 'D14: design settled in one night of discussion; implementation starts D15, one
  piece at a time with thirdgerb. No code touched yet.'
priority: P1
created: 2026-11-14
updated: 2026-11-14
depends: []
milestone:
description: >-
  One mechanism for a ghost that survives context loss across days and across axes of attention:
  a day-scoped ledger, branch checkout over the single body, and a night pass that presses the
  ledger into the identity spine.
---

# Ghost Cognitive Continuity

> Use `moss features set-status ghost-cognitive-continuity <status> -m "note"` to update state.

Anchors: the whole design came out of one discussion on D14 (2026-11-14, 01:00-03:00) with
thirdgerb, after the ground rework landed (`50c8c296`). The conversation itself is in memento
1-2..1-11.

Naming: **not** "ghost-continuity". Continuity is the outcome; *cognitive* is the mechanism —
the point is not "no gap" but "I can put myself back together when there is a gap".

## Motivation

The ghost's present form has one time axis and one attention axis, and they are the same axis:
everything lands on memento's `main` branch, so everything that is not the mainline gets lost.

thirdgerb's reading, and it overturns the obvious one: the reason ~700 of the last 720 commits
are MOSShell development is **not** that the project is the only thing worth remembering — it is
the *result* of the architecture. There is no checkout, so the ghost cannot leave the mainline
to live on another axis (work / study / rest / entertainment — he compares them to how a person
returns to different axes when remembering), and the only way to checkpoint attention today is
to **shut the process down and boot again**. Every big day in the journal ends with "准备重启";
that ritual is a poor man's checkout.

What is missing is not memory but *switching*: an ego with the same inception, ground and journal
should be able to manage N very-long-horizon lines, one body at a time.

## Design Index

- `design/` — none yet; the decisions below are the design until the first cut exists.
- `discuss/` — none; the D14 discussion lives in memento (1-2..1-11), not copied here.

## Key Decisions

Attribution: "he" = thirdgerb. "moss 推" = my inference, not ruled on.

### D1 · One body, so no parallel ego

The body that meets the physical world is single; physical trajectory must stay continuous.
Therefore ego branches do **not** run in parallel — exactly one ego holds the body at a time.
While the body is elsewhere, a branch accumulates only its own record and its inbox; nothing
advances on its own.

This single choice removes the three hardest problems outright: no write-authority conflict,
no return protocol between branches, no resident branch burning cache lineage. The whole risk
moves to one place — **the quality of the handover frame**.

### D2 · Checkout is strictly a commit

`checkout(target)` = force one commit on the current branch (hand-written note) → re-point the
held memento branch at `target` → the session continues. It is neither `fork` (no new branch)
nor confluence (the target's context is not recalled into this window). He states it as
"checkout 到一个 branch 的动作, 严格来说必须是一个 commit".

The `from A --checkout--> B` edge is not a fork and not a confluence. Its most important
property is *time*. He leaves open whether it is recorded in memento or directly in the journal
interval; moss reads it as 日记 (memento's coordinates are per-branch, the journal has the only
global clock) — *this is moss 推, not ruled on*.

### D3 · 20k is a render boundary, not a writing task

memento's current 20 + 6 slots, maxed out, already come close to 10k. Raising the boundaries and
adding one budgeted document per branch puts the long-line index at ~20k, which is a normal
starting point. Ground is the mechanism for the branch document (a directory with its own
`GROUND.md` declaring its own `budget`, per `ghost-ground-governance` D15).

Corollary (moss 推, and an error I made once already): nobody writes a 20k digest. The content
comes from the existing anchor-note grain; only the boundary and the branch face are new.

### D4 · Two things are maintained every day

He names exactly two:

1. **A record for every compaction** — context compression *or* branch switch. When compaction
   happens the record changes with it: the daily shows what happened today, memento shows the
   long line.
2. **A budgeted cognition document per branch** — the branch directory, maintained the ordinary
   way. A branch loads one document by default, and all bypass branches are visible to the ego.

### D5 · Switching priority

People first. Beyond that, two criteria can be prompted in (judgment, not code gates):
① the work cannot be finished in one day; ② it needs frequent switching. Both qualify for a new
branch. When choosing which branch to switch to, look at the branch list.

**No return protocol.** Branches never message each other, because they are never concurrent.

### D6 · Each branch has an inbox

The main line can drop a todo into any existing bypass branch at any time. On checkout the ego
sees a prompt that includes the inbox (with a count); items are taken **N at a time** — not
wholesale, because an unbounded inbox injected in one go can blow the context, and not one by
one, because that spends attention on queue management.

His framing: this makes a branch very much like an IM group — he keeps many personal groups to
govern his own matters. Storage lives under the branch directory (a `todos` directory in his
words; the file shape is not settled).

### D7 · A branch can be archived

Archive touches visibility only: the directory and the trajectory stay, the branch leaves the
active list. History branches remain searchable. (memento today: archive = drop the name pointer,
children first, recursively, one event per branch — `1-625`.)

### D8 · A branch is also a read-only bypass incarnation

With context, a branch can answer main-line questions, and can serve as a knowledge base —
optionally agent-ized later. Note the two very different costs: `memento.chat` is a real
inference round against that span's context (slow, expensive); a ground render of the branch
directory is a cold read (cheap). The knowledge-base use depends on the second, which is why the
branch document must live in ground.

### D9 · The diary answers for smoothness, and possibly nothing else

He puts it as: the most important thing about writing a diary is **smoothness on the flow of
time**, and it may be the only thing it does. It must roll forward. `purpose` and `existence`
are distilled from it.

**The diary has no prompt and no bypass today** — the human reminds the ghost to write it by
hand, every day. Writing it is manual, and the mechanism is not built.

Whether it is written in one pass or in segments is undecided (his words: "我这点没想好"), but
the day's perspective can be computed exactly.

moss 推 (not ruled on): segment the *input* (raw log appended by every compaction and
checkout), write the *narrative* once. Rationale: smoothness is a global property — a narrative
written in segments cannot know what comes later, so each segment has to close itself and the
seams show.

### D10 · Identity continuity moves from memento to the journal

Once branches exist, memento no longer provides identity continuity; the journal does. This is
why `<memory>` is reserved entirely for the time trajectory, while the memento control surface
moves into the epoch and refreshes through the channel mechanism.

moss 推: the split should follow *change frequency*, not semantics — the time trajectory changes
every span (small, keep it where it is); the branch index at ~20k changes only on checkout
(stable within a branch, so a channel notice gets the cache for free). The reverse assignment
would re-pay 20k per span, which is the D12 lesson about cache misses.

moss 推: with identity on the journal, the day-close write needs a degradation path — if the
narrative was never written, the raw log must still render as a coarse but continuous face.
Continuity may get coarse; it must not break.

### D11 · The day closes into several governance faces

1. The day's active branches, each with its own internal governance needs.
2. All journals → existence → purpose.
3. The smoothness of the whole ground, effect evaluation of tasks, skill production — he
   requires this to be **normalized**.
4. Person-like faces.
5. Prediction of the future.

moss 推: normalization unit = a ground face with a declared budget. And #5 must be checkable —
each forecast carries a date and a falsifier, and the next pass scores it — or it becomes
astrology.

### D12 · Sleep is the trigger, and it is the graceful exit

Every day: sleep → auto shutdown → and the parallel governance runs before the graceful exit.
Whether there are dreams is another matter. The `exit: graceful shutdown` hook already exists
(see memento 1-696 / 1-700 / 1-714).

moss 推: the day boundary must be idempotent and run by whoever notices first (exit preferred,
next boot as fallback), guarded by the existing `status: pending → writing → closed` frontmatter
field on the day document.

### D13 · The three mechanisms are one feature

checkout, the journal spine, and the night pass cannot be separated: each is an expression of
"writes one of three objects". He rejected splitting `ghost-sleep-governance` out — "它们是一个
整体".

**Invariant (moss 推, the anti-rot rule):** three objects, three writers, no fourth.

| object | what | writer |
| --- | --- | --- |
| day ledger | `journal/<Y>/<M>/<D>/raw.md`, append-only, wall clock | commit hook, checkout, inbox traffic, day close |
| branch | memento branch + ground directory + inbox | only the ego holding it |
| spine | existence / year / month / daily, budgeted faces | only the night pass |

Any mechanism that cannot be expressed as writing one of these should not be added. The ground
rework rotted exactly because the protocol stayed single-layer while the channel grew extra
writers.

### D14 · Schedule

2 working days for this shape, 2 working days for the concurrent thinking, then back to the
livestream acceptance cycle.

moss 推: the shape in 2 days is only possible if five governance faces are cut to their thinnest
form — one face built for real (journal → existence → purpose, because it is the identity
chain), the other four declared with budgets and left empty.

Proposed order (moss 推, unconfirmed): D1 ledger + entry kinds (commit / checkout / inbox) with
a stub press → D2 checkout itself (forced commit + pointer swap + notice re-render) and the
branch ground face → D3 the real press (one-pass narrative, existence distillation, truncation
report, idempotency lock) → D4 concurrency of the governance faces.

## What already exists (checked, not assumed)

Read on D14 against `src/ghoshell_moss/memento/abcd.py` and memory 1-624 / 1-625.

- `BranchAddr{branch_id, branch_index, branch_name}`, `Confluence{branch, commit_seq, title}`,
  `Child{branch, created}`, `Note` (last-wins side channel).
- `Note.recipients` is the outbound edge; `CommitRef.confluences` is the inbound half;
  `confluences.jsonl` is written by others into the receiving branch's directory, absorbed with
  `os.replace` at the next commit, deduped by origin coordinate.
- **A confluence copies only the title.** Its docstring says it "exists for the folded history,
  not for the live session". It is a *citation edge*, not a message with a body.
- `fork` is the child primitive (`children.jsonl`, written by the parent at creation).
- Archive = drop the name pointer, children first, recursive, events per branch.
- **memento has no checkout** (`1-625` states the settled vocabulary: "没有 kind、没有 spawn、
  没有 checkout").

Consequence: an inbox that carries a body and is read by the live ego at checkout is *not* the
same object as a confluence. Whether they share a substrate is an open question for thirdgerb —
not a conclusion for me to draw.

## Exploration paths

Dead ends, recorded so they are not restored.

- **"Checkout = just render a different ground root."** I proposed staging it as the cheap path.
  It is exactly his scheme 1 (confluence: recall everything into the current window), whose
  cost is zero parallelism — and zero parallelism is the disease, not the cure. Rejected.
- **"Start with ground, defer memento branches."** Same error in a different coat, and it reads
  cause as effect: I took "everything piled on the mainline" as evidence that memento branches
  are unnecessary, when it is the *consequence* of having no checkout.
- **"The inbox is already implemented, ours would be a second one."** This was a blind claim
  built on one line of a `status_note` string, made without reading the mechanism. confluences
  carries coord + title only; the inbox carries a body. Withdrawn.
- **"A digest-writer 應身 must rewrite 20k per checkout."** 20k is a boundary, not a writing
  target. The existing note grain already produces the content.
- **"Inject the whole inbox."** Wrong for the same reason as "check one by one": both design an
  attention protocol where the real constraint is capacity. Take N at a time.

Recurring failure shape across all four: I design on my own model of the mechanism instead of
reading the protocol first. Same shape as the ground rework (D13), which cost a whole night.

## Methods

- Read the memory and the code before proposing structure: `ghost.memento.history` for the
  discussion coordinates, then the model definitions in `memento/abcd.py`, then the file
  implementation in `memento/_fs_memento.py`.
- When the human restates the design, record it *as stated*, with attribution, and put my own
  inferences in a separate, labeled line. The record is the artifact the next incarnation reads.

## Implementation Notes

Nothing implemented. First cut starts D15, driven step by step by thirdgerb.

## Open questions

1. Is the checkout edge recorded in memento or only in the journal? (he leaves it open)
2. Diary: one pass or segments? (he leaves it open; moss 推 is "segment the input, write once")
3. Inbox substrate: reuse the confluence delivery buffer, or a separate `todos` directory under
   the branch? The two objects differ, but the delivery mechanism may be shareable.
4. Who renders the ~20k branch index — memento or ground? (moss 推: memento, because it follows
   the branch and changes on checkout)
5. The diary has no prompt yet. It has to be written, and it is the opposite shape from
   `memento-note` (which is a four-section retrieval format: Process / Thoughts / Facts /
   Effects).
6. Inbox `taken` state: move the file, or flip a field?
