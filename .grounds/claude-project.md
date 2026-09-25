---
name: claude-project
description: Claude Code 项目 ground — 兼容 CLAUDE.md 约定, 不修改项目文件.
pins:
- label: claude
  verb: law
  arguments:
    filename: CLAUDE.md
    budget: 20000
  description: CLAUDE.md 法链 — 从 cwd 向上收集到 ground root, 20k 上限
---

# Claude Project

Compatibility ground for Claude Code projects. CLAUDE.md is loaded as
law via the `law` pin — the project's own files stay untouched. At the
root this renders the root CLAUDE.md; walking into subdirectories shows
the ancestor CLAUDE.md chain from where you stand.

Add more pins above (edit frontmatter), then `moss ground validate`.
