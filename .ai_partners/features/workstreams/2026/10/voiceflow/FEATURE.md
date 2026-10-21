---
title: Voiceflow
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P0
created: 2026-10-21
updated: 2026-10-21
depends: []
milestone:
description: >-
  Name and cut the voice-interaction layer: etiquette is only a preset of
  three orthogonal dimensions, the live state is one `current`, and the model
  reads / tunes / keeps it at runtime.
---

# Voiceflow

> Use `moss features set-status voiceflow <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Thirteen built-in etiquettes accumulated during testing. Reading them, most are
not different *kinds* of conversation — they are the same knobs set to different
values. The model cannot see that, so it treats `murmur` / `cue` / `duplex` as
three behaviours to choose between, when the honest model is one live state with
three dimensions.

The cost shows up in real conversation: `duplex` had "出声即让路" on, so the
moment the human started a sentence the reply being spoken was cut off — a
behaviour nobody chose that round, inherited from a preset picked for testing.
Conversely there was no way to ask "how much of my sentence is left before he
takes the floor", so the model could not decide to stop by itself.

So this workstream does three things:

1. names the layer — **voiceflow** (by analogy with `mindflow`: mindflow decides
   where *my attention* goes, voiceflow decides *when either of us speaks*);
2. splits the state into three orthogonal dimensions and makes it a single live
   object (`current`);
3. gives the model three verbs on it — read, tune, keep — and shrinks the
   built-in presets to the handful that say something distinct.

## Design Index

- Key design documents: `design/`
- The drawing made during the session: `matrix.mesh.svg` work `voiceflow`
  (42 ops incl. the discussion that produced it) and the earlier `etiquette` work.

## Key Decisions

**D1 — etiquette is a preset, not a behaviour.** An etiquette is a named point
in the product space of three layers; it is data, not a code branch. Switching
presets resets `current` to that point.

**D2 — three orthogonal dimensions.** Everything the old thirteen expressed is a
value on one of:

- **A · 判停** (turn end) — `stop.silence`, `stop.keywords`, `stop.classifier`,
  `stop.detect`. *"他这一句什么时候算说完"*
- **B · 交付** (delivery) — `deliver.emit`, `per_clause`, `mode`, `priority`,
  `history`, `low_confidence`. *"说完的话怎么到我这儿"*
- **C · 打断** (interruption) — `onset.emit/interrupt/priority`,
  `deliver.interrupt`, `deliver.barge_in_while_speaking` + boost.
  *"谁能拿走我正在做的事"*

**D3 — there is always exactly one `current`.** It remembers which preset it was
loaded from (`origin`), so a read can say *which knobs have moved away from the
preset*. Tuning mutates `current` in place; it does not create a preset.

**D4 — 出声即让路 moves from `duplex` to `cue`.** The human's decision, verbatim:
> 我说话的那个最后一个语音包，如果落到你正在说话的时候，就会打断你说话，
> 但是不打断你的动作，OK，我现在不需要这一个。

So the baseline (`duplex`) never cuts the model's voice; the model finishes its
sentence and the human takes the next turn. `cue` is the one that yields: there,
the human's voice immediately wins over the model's voice.

**D5 — `cue` gains an instruction, not a name binding.** `cue` means "speak only
when addressed". The name must **not** be hard-coded — this is a system default,
and a ghost adds its own name later:
> 名字那个地方就不要绑定了，因为这个不是说给你做的，而是说我们系统的默认的。

The rule is written into the etiquette's own `description`, which the listener
surfaces to the model each turn. `always` (兜底) is the opposite: the human's
voice has absolute priority, even over thinking.

**D6 — five presets survive; `knock` is not a preset.** Kept: `once`, `always`,
`aside`, `cue`, `duplex`. `knock` is the idle gate — a mechanism the listener
already owns after idle degradation, not a conversation style. Everything else
(`notify` / `scribe` / `observer` / `keyword_end` / `scored` / `murmur`) is a
combination of the three dimensions and can be reached at runtime by tuning.

**D7 — three verbs, and `tune` must confirm by returning the new state.**
> 改：只收我要动的旋钮，返回改完的现值 —— 这就是确认生效，不是回一个 ok 就算。

**D8 — presets live in the ghost home, not in the kernel.** The kernel ships the
three dimensions and their defaults; names and combinations are grown where they
are used:
> 礼仪本身应该是在 ghost home 里面根据实际情况一个个建出来的。

**D9 — `say(replace=True)` is "reply".** Its docstring now says finishing the
current sentence is not an obligation: when the human's new words change what he
means, answer now rather than queue behind an unfinished line. (`speech_module.py`)

## Exploration paths

- Two-column board (my proposals vs. his) was the *discussion* surface. It was
  replaced by the model drawing once the discussion converged — the columns are
  still readable in the `etiquette` work's ops, but the deliverable is `voiceflow`.
- Draw a knob as a *sentence* ("出声即让路 —— 连我正在说的语音一起掐"), never as a
  field name: the human reads meaning, and each of these knobs is a judgement
  about who yields, not a parameter.

## Methods

- Read the live state from `controller._active_etiquette` — it is the single
  truth; onset and deliver paths read it at signal time, so **A-live knobs take
  effect immediately**.
- `stop` knobs are copied into `StopJudge` at session start (`_make_stop_judge`),
  so **tuning a 判停 knob requires a session restart**; `tune` must say so in its
  return value instead of pretending it applied.

## Implementation Notes

- `run_etiquette` deep-copies the spec per session; the copy *is* the object
  `_active_etiquette` points at, so in-place mutation reaches the live session
  without a restart for onset/deliver.
- The config store (`.moss/configs/listener_etiquette.default.yml`) is a dump of
  `EtiquetteConfig`; shrinking the built-in list does not shrink an existing
  store — both have to be updated for a clean thirteen → five cut.
