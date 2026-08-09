---
label: claude-memory
description: Claude Code memory 系统 ground — MEMORY.md 索引 + 磁盘真实状态对账.
pins:
- label: memories
  verb: frontmatter
  arguments:
    path: "*.md"
    keys: ["name", "description", "type"]
    budget: 3000
  description: 磁盘上的记忆文件 — name/description/type, 反映真实状态, 可与 MEMORY.md 对账发现腐烂
---

# Claude Memory

@MEMORY.md

This directory is Claude Code's persistent memory system. MEMORY.md is
the authoritative index; each linked `.md` file is a typed memory record.

The pin below shows what's actually on disk — the ground truth. Compare
it with MEMORY.md:

- File in pin but not in MEMORY.md → forgotten to be indexed (decay)
- File in MEMORY.md but not in pin → deleted but still referenced (decay)

Memory files carry frontmatter: `name`, `description`, `type` (user /
feedback / project / reference). The body has the full context; the
frontmatter is the indexable surface.
