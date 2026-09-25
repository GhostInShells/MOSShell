# Benchmark Convention

> bench.md + case.jsonl — model evaluation assets, model-agnostic, re-runnable across providers.

## Why

Automated tests verify code correctness. Benchmarks verify **model capability** —
can a model (any model) perform this task reliably, fast, and accurately?

A benchmark answers: "If I swap the model, does accuracy/latency change?"
It is model-agnostic by design — the same case set runs against any provider.

## Files

```
.ai_partners/benchmarks/
  README.md               # This file
  TEMPLATE.md             # Copy to create a new benchmark
  <name>/                 # kebab-case
    bench.md              # YAML frontmatter (BenchmarkMeta) + markdown description
    cases.jsonl           # One JSON object per line ({label, prompt, instruction?, expected?, times?})
    models.py             # Response model (BaseModel subclass)
```

**bench.md frontmatter** — maps directly to `BenchmarkMeta`:

| Field | Required | Purpose |
|-------|----------|---------|
| title | yes | benchmark name |
| result_type | yes | `module:attr` of response BaseModel |
| instruction | no | default system prompt (case can override) |
| cases_file | no | path to cases jsonl (default: `cases.jsonl`) |
| version | no | semver for result tracing (default: v1.0.0) |

**cases.jsonl** — one JSON object per line:

| Field | Required | Purpose |
|-------|----------|---------|
| label | yes | unique case identifier |
| prompt | yes | the prompt string (or file path relative to benchmark dir) |
| expected | no | scoring reference (natural language — scorer interprets) |
| instruction | no | case-level override of default instruction |
| times | no | repeat count (default: 1) |

## How to Run

Two paths, same engine:

1. **CLI** — single case, quick check:  
   `PYTHONPATH=<bench-dir> moss llms call "<prompt>" -r models:ModelName -j`

2. **Python** — full benchmark, all cases:  
   `PYTHONPATH=<bench-dir> python run_bench.py <bench-dir>`

The engine is `PydanticAIFuncs.run_benchmark()` — reads bench.md frontmatter
(via BenchmarkMeta), loads cases, calls each, produces results.

## Scoring

Scoring is a model func too — `scorer(prompt, expected, response) -> Score`.
A loop feeds results into the scorer, producing accuracy/latency summaries.
The scorer is declared in bench.md (per-benchmark).

## Discovery

```
find .ai_partners/benchmarks -name "bench.md"
```

No CLI. The filesystem is the index. Each bench.md is one benchmark.

## Relationship to Regressions

| | Benchmarks | Regressions |
|---|---|---|
| Unit | Model capability | System correctness |
| Case format | jsonl (scriptable) | Markdown table (human-first) |
| Run | Automated (engine loop) | Human-in-the-loop |
| Model | Different models, compare | Model-irrelevant |
| Product | result.jsonl + score | baseline pass/fail |
| Scoring | Model func (self-bootstrapping) | Human diagnosis |

## State Machine

```
draft → active → expired
```

- `draft`: cases being designed, no complete run yet
- `active`: at least one full run, producing comparable results
- `expired`: task no longer relevant; stays as historical reference

## Model's Role

- **Propose benchmarks at session start.** `find .ai_partners/benchmarks -name "bench.md"`
  discovers all benchmarks — run alongside `moss features list` and regression find.
- **Design cases with the human.** Human defines the task and what matters;
  model proposes cases, edge conditions, scoring criteria.
- **Run and compare.** Same benchmark, different `--provider/--model/--tag`;
  produce result sets for cross-model comparison.
- **Score via model func.** The scorer is another model func — benchmarks that
  evaluate model outputs are themselves model-driven, closing the loop.
