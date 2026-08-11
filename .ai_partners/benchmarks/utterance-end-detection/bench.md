---
title: utterance-end-detection
version: v2.0.0
result_type: models:UtteranceEndScore
instruction: >
  You are a conversation end detector. Output a single 0-9 completeness
  score for the utterance. Digit only.
cases_file: cases.jsonl
---

# Utterance End Detection Benchmark (single-token)

Tests whether a fast model can output a SINGLE 0-9 token score for utterance
completeness — the scoring rubric (the hint) is a strategy variable: it can
be placed in the instruction, in a thinking block (内观), or omitted.

## Strategies (A/B via run_bench.py)

- baseline: minimal instruction, no hint.
- hint-in-instruction: `--instruction rubric.txt`.
- hint-in-thinking: `--thinking rubric.txt`.
- both.

## Objective

- Flash model, single structured call, single-token 0-9 score
- Compare strategy accuracy / latency

## Findings (2026-08-11, anthropic/deepseek-v4-flash, single runs)

- **Single-token 0-9 scoring works reliably** — every strategy produced a
  valid digit; the structured output carries only `score`, no reason.
- **Strategy comparison (u1-u7, 7 cases, 1 run each)**:
  - baseline (no hint): `[9,9,0,2,2,3,9]` — mid-word cut nailed (0).
  - hint-in-instruction: `[9,9,2,2,2,3,9]` — mid-word overscored (2).
  - hint-in-thinking (内观): `[9,9,1,1,0,2,9]` — closer on mid-word (1)
    but over-penalizes ambiguous (u5=0, expected ~5).
- **Ambiguous middle band (u5) is the hard case** across strategies —
  both 0/9 ends are reliable; the model's real cost-earning zone is the
  ambiguous middle.
- **First-person narrative thinking hint** (rubric-thinking.txt) still
  over-penalizes ambiguous (u5=1, u18=2) and introduces new errors
  (u17 narrative lead-in=8, u19 complete=5) — style of the 内观 hint
  matters but doesn't fix the middle band.
- **Per-case config works** — u23 (thinking) vs u24 (instruction) on the
  same prompt scored 2/2 in one run; strategy is a per-case variable.
- **Prompt cache: negative.** Two identical cases (u21/u22) took 1.03s /
  1.11s — no observable cache-hit speedup at this prompt size on this
  provider (deepseek-v4-flash via anthropic protocol).
- **Latency ~1s = model call** (`agent.run` round trip, `cast` field) —
  inference + network dominated, not a cacheable prefix. Caveat: this ran
  over the human's phone-wifi hotspot, slower than good wifi; the absolute
  number is network-sensitive. The time-sensitive point stands: clause-level
  single-token judgment is fast enough to sit in the ASR-tail + VAD +
  link + first-packet budget, and network is part of that budget.

