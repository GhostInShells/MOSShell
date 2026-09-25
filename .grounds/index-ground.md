---
name: index-ground
description: 认知场索引 — 扫描项目中所有 GROUND.md, 展示每个场的 description.
ignore:
  - ".moss/"
pins:
- label: fields
  verb: frontmatter
  arguments:
    path: "**/GROUND.md"
    keys: ["description"]
    max_depth: 3
    limit: 50
  description: 项目中的认知场索引 — 每加一个 GROUND.md 自动出现
---

# Ground Index

This directory is the root of a project that contains multiple cognitive
fields (grounds). Each subdirectory with a GROUND.md is a field —
a directory with a declared identity and a set of pins that describe
what's inside.

The pin below lists every ground in the project. The path shows where;
the description shows what you'll find there. Enter a directory to read
that ground's own body and pins — each one teaches its own domain.
