# Pre-modeling — 有预算的预建模

Status: **parked, low priority.** Value is uncertain; cost is certain. This is a
proposal set aside, not a plan. It is built only if the value test below starts to pass.

## The one mechanism

Unload a secondary contradiction of the main interaction path into bypass attention:
fork from the same context, re-activate the lens with a per-round prompt, keep it inside
a budget, and merge back only when it is useful. The main path stays on interaction; the
bypass carries what the main path would otherwise drop when busy.

The three examples share this skeleton and differ only in what is unloaded:

- **reflection — doubt.** prompt + expectation; the fork judges a span against a failure
  criterion and returns a positive correction (an impulse) or stays silent.
- **perspective — angle.** a prompt defines a domain question; the fork models that
  angle and merges the model into the context.
- **prediction — forecast.** a prompt asks what happens next (intent, trajectory); the
  fork predicts and merges the forecast.

Mechanically one object with three presets: lens prompt, trigger, budget, output schema,
merge policy, value metric.

## Reflection preset — mechanics

- Trigger per-turn / per-commit; fork at the hot anchor; generation budget ~1000 tokens.
- Polarity discipline: prompt positive (the ego sees questions, not prohibitions);
  expectation negative (the failure criterion lives only in the reflection round); impulse
  positive (a better way, not a complaint).
- Not a gate: the model already gates itself in-context; the lens catches what the
  context silently down-weights. Non-interrupting impulse types by default.
- Above threshold, a second request asks for a positive correction carrying the first
  round's evidence (guards against generic corrections).
- Per-commit, score the numbered impulses; below threshold, close the point. Weight by
  consequence, not frequency.
- Benchmark: a judge against the expectation, plus downstream recurrence (did the same
  failure come back).

## The hard problem: distinctiveness

All lenses share the prefix, so they converge. Distinctiveness must come from:

1. **the question's subject**, not the prompt's adjectives — each lens asks a question the
   others do not answer, with a different output schema. A lens is one-turn and carries
   no history of its own: its identity is the prompt, so the prompt must be a question,
   not a style.
2. **partitioning by the main path's neglect** — the lens owns a dimension the main path
   drops when busy, and the main path must actually stop doing it (giving up is the
   premise). Otherwise the lens is an echo and the main path still does the work.
3. **measurement** — prediction divergence for forecasts (two lenses with one forecast
   are redundant), target/aspect divergence for corrections. Low divergence: merge or
   kill.

## Why parked: value uncertain, cost certain

Cost is N lenses × K cycles × M tokens — predictable and paid. Value is not: a lens may
produce nothing the main path needed. Redundant lenses do not change behavior, so they
are pure waste. The value test:

- forecasts: did the prediction come true?
- corrections: did the user-correction rate drop?
- all lenses: did the main path stop doing that dimension?

## Boundary if it is ever built

- N: a registered set, relevance-activated; start with one.
- K: event trigger (state changed) plus cooldown, not a fixed period.
- M: a hard per-run cap; the output is a card (prediction + confidence + suggested lens),
  not prose.
- deadline: a result that misses its usefulness window is dropped, not merged.
- per-round merge budget: top-k by confidence, dedupe, only on change.
- prefer **lens injection** (a cheap detector plus one line of prompt) before a subagent;
  a subagent is only for what must be computed ahead of time.
- fork at the hot anchor so the prefix cache hits (a cache hit costs ~1/50).

## Open questions

- Can the main path really give up the dimension, or will it keep doing it anyway?
- What makes two lenses genuinely different once the prefix is shared?
- Is one lens (intent prediction) enough to measure whether the whole idea works?
