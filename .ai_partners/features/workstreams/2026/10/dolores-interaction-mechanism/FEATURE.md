---
title: Dolores Interaction Mechanism
status: in-progress
priority: P1
created: 2026-10-21
updated: 2026-10-21
depends: []
milestone:
description: >-
  How a ghost stays legible to the human while it works: every behaviour in
  this family lands as an intent in the tool surface, not as a line in a
  prompt.
---

# Dolores Interaction Mechanism

## Motivation

DSH's default shape is: work first, speak last. A ghost that thinks for two
minutes and then answers is invisible — with no screen open, the human cannot
tell whether it has started, is stuck, or has gone quiet. The natural fix is a
sentence in the prompt ("say something before you start"), and it does not
survive contact with a tool call: the moment a tool is within reach, that
instruction is one line among thousands, and the tool wins.

thirdgerb (D21): "这 DSH 机制的一个倾向就是干完活才做第一次交互" — and the fix
is not more instruction but a new intent in the tool surface:
"所有给你的 MOSS tool 本质上都是意图识别 ... 加一个叫做 reply first".

This is the same judgement the discuss redesign landed on earlier the same day:
a behaviour that has to be remembered is a behaviour that will be skipped.
Prompt layers decay; mechanisms hold.

## Principle

A MOSS tool is a word in the ghost's vocabulary. Adding a tool adds an intent —
it rides in the choice set at every step, present at the moment of decision,
whereas a prompt sentence has to be recalled. So a behaviour worth having
reliably is expressed as a tool named after the behaviour.

Two consequences worth keeping:

- the name must be the intent (`reply_first`), not the mechanism
  (`interpret_with_a_short_line`);
- a tool that only renames an existing one is still worth its place when the
  name is what makes the intent findable — but then the two must not drift, so
  `reply_first` shares `moss_interpret`'s implementation exactly.

## Case 1 — reply_first

The first move of any turn that will take more than a few seconds is one short
line to the human — "先拉一下屏幕" / "我在读代码" — so the ghost is audible from
the first second, before any result exists.

- Named for the intent: `moss_reply_first`.
- Behaves as `moss_interpret` + `moss_observe`: the CTML streams and executes,
  the call returns once its observed commands settle, and the turn continues.
- It does **not** end the turn. That is `moss_react`'s job, and the distinction
  is the whole point of having both: react ends, reply_first precedes.
- It carries one line, not a plan. The work itself follows in ordinary tools.

## Family — same shape, a different moment

Candidates, added one at a time as the need shows up:

- `reply_first` — before the work starts.
- close-out — one line saying what happened, instead of a silent stop.
- progress tick — a point mid-way through a long stretch, so a long job is not
  a black box with a single edge.

## Open questions

- Enforced or taught? Should `reply_first` reject a long CTML, or only exist
  with a description that says "one short line"?
- Scope: this mechanism exists to serve DSH. Another harness might not need a
  tool layer at all — so the tool belongs to the Dolores prototype, not to MOSS
  core, and the name follows: `dolores-interaction-mechanism`, not "ghost
  interaction mechanism".
- Division of labour: the tool description and the underscore prompt
  (`_prompts.py`) both speak about the same behaviour. Which one carries the
  rule, and which one stays quiet?
- How does the ghost learn whether it is being used at all — a usage count, or
  does it work simply because the affordance is there?