---
label: skill-ground
description: Skills 集合目录 ground — 标记认知场类型, 索引子目录 SKILL.md, 不改 skill 文件.
pins:
- label: skills
  verb: frontmatter
  arguments:
    path: "*/SKILL.md"
    keys: ["description"]
    limit: 20
    max_depth: 1
  description: 子目录 SKILL.md 索引 — 只看 description, 一层, 上限 20
---

# Skills

This directory is a skills collection. Each subdirectory with a `SKILL.md`
is a skill — a reusable prompt or procedure that a coding agent can invoke.

## How to use

The pin below lists all available skills and their descriptions. To invoke
a skill in Claude Code, type `/<directory-name>` (the subdirectory name,
not the `name` field in frontmatter).

To understand a skill in full — its detailed instructions, constraints,
and examples — read its `SKILL.md` directly.

## How to add

Create a subdirectory with a `SKILL.md`. The frontmatter must have:

- `name`: display name for the skill
- `description`: what the skill does (this is what the pin indexes)

## Discovery

This ground is discovered through the parent ground's directory view
(`ls` pin or equivalent). A model entering here reads this body to
understand the field type, scans the skill index, and decides which
skill to invoke or explore.
