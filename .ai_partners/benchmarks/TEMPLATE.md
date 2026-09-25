---
title: $BENCH_TITLE
version: v1.0.0
result_type: module:QualifiedModelName
instruction: >
  $INSTRUCTION
cases_file: cases.jsonl
---

# $BENCH_TITLE

<!--
Frontmatter fields map to BenchmarkMeta. Fields not listed below are
optional — see README.md for the full schema.

  title        — benchmark name (kebab-case recommended for directory)
  version      — semver for result tracing
  result_type  — module:attr of the response BaseModel (runtime import)
  instruction  — default system prompt; individual cases can override
  cases_file   — path to jsonl, relative to bench.md (usually cases.jsonl)

The markdown body is the human-readable description (motivation, objective,
scorer design, etc.) — it becomes BenchmarkMeta.description.
-->

## Objective

<!-- What is this benchmark measuring? What question does it answer? -->

## Response Model

<!-- Describe the structured output — what fields, what they mean. -->

## Cases

<!-- How are cases constructed? What does expected mean for scoring? -->

## Scoring

<!-- How to interpret results? What scorer model func is used? -->
