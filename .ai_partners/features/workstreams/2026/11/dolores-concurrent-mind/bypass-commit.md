# Bypass Commit Mechanism — 旁路 commit 机制

Status: problem established; design not settled.

## What it is

The first functional mechanism of the concurrent mind, and the existing starting point.
A commit anchor forks a one-turn bypass session; the session is seeded with the source
session's verbatim prefix, so it hits the same kvcache and thinks from a state close to
the main path. Its only output is the commit note.

Place in the design space: commit fork / one-turn / effort off / token cap / no
workspace / readonly / no shell / lifecycle bound to the ghost. It is the "function"
corner of the bypass space — not an attention.

Constraints it must keep:

- the verbatim prefix, for the kvcache hit;
- lifecycle bound to the ghost — shutdown cancels in-flight notes; missing notes stay
  empty and are backfilled on the next boot (already implemented in
  `EgoMementoManager`).

## Before the design: what must be fixed

- The bypass commit must be one-shot readonly — a request must not produce local files.
- It must run without a session log.

(Both are construction todos, to be extracted after the discussion settles.)

## thirdgerb's design points

1. **Governable prompt.** The bypass commit needs a home-readable prompt, itself
   governable — most likely in a directory superego governs.
2. **Replayable → benchmarkable.** Commit notes can be replayed, so the task can be
   benchmarked. A bypass scores / labels commits; periodically statistics decide whether
   to optimize. Example: a day's total commit score below 60 triggers optimization;
   regression in units of N. After regression, a new expectation is established for
   future scoring.
3. **Scoring.** At a compaction node's commit ref: rebuild, insert the expectation, and
   score the quality of the commits visible in its own context. Archive the score; use
   it to drive the downstream logic.
4. **The `request` tool.** A system reminder carries the request key and a JSON result
   schema; the model answers with `request(key=..., result=...)` — no per-request tool
   definition.

## The ghost's response

**The expectation should generate questions, not scores.** A judge with an expectation
measures taste; the note's product goal is "can the next self live on it". Proposal:

- fork a probe branch at the compaction anchor, seeded with a notes-only prefix (no
  recap), isolating the note;
- the expectation generates probes from the span: what happened, what was decided, what
  is still open, what to do next;
- the probe answers from the note alone; **pass rate is the score**;
- grade mechanically (extraction / yes-no / multiple choice), or the judge's taste
  re-enters through the back door;
- a second, product-level score uses the full compacted prefix (with recap) — the
  context a future session actually receives.

**Loop fixes**

1. No daily sum: it depends on volume and lets one good note average out a bad one. Use
   per-compaction pass-rate distributions and windowed low quantiles; trigger on
   sustained drift. Regression by pairwise comparison against the incumbent on the same
   frozen set.
2. Regression covers the whole note pipeline: prompt + model / effort / token cap. The
   cap is often the real bottleneck.
3. The expectation must not be written by the optimizer — that is setting your own exam
   and grading it. Derive it from the consumer side (which span the next session
   re-read, which fact it asked twice); at minimum, freeze the benchmark set and never
   tune and evaluate on the same spans.

## Open questions

- Who writes the expectation — a fixed rubric, or consumer-side signals?
- Probe cost per compaction: always, or only during regression?
- Where does the scoring bypass's own output go — a file, a branch commit, an impulse?

## Artifacts

`fork-points` and `bypass-runtime-axes` on the artifacts surface (not exported).
