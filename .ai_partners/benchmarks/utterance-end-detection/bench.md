---
title: utterance-end-detection
version: v1.0.0
result_type: models:UtteranceEndScore
instruction: >
  You are a conversation end detector. Rate how complete each utterance is on
  a 0-9 scale. 0 = clearly incomplete (mid-word, cut off mid-thought).
  5 = ambiguous (could go either way). 9 = clearly a complete thought,
  question, or statement. Return the score and a one-sentence reason.
cases_file: cases.jsonl
---

# Utterance End Detection Benchmark

Tests whether a fast model can reliably detect utterance completeness in a
single token / structured output call — a core building block for voice
interaction turn-taking.

## Objective

- Flash model, single structured call
- Rating 0-9 per utterance
- Target: accuracy > 90%, latency < 300ms

## Response Model

`UtteranceEndScore` — score (0-9 int) + reason (str).
