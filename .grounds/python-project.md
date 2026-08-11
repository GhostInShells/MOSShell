---
name: python-project
description: Python project ground — readme, source tree, tests, and pyproject.
pins:
- label: readme
  verb: file
  arguments:
    path: README.md
  description: 项目自述
- label: pyproject
  verb: file
  arguments:
    path: pyproject.toml
  description: 项目元数据与依赖
- label: source-tree
  verb: ls
  arguments:
    path: $CWD/src
    depth: 3
  description: 源码树, 最多三级
- label: tests
  verb: glob
  arguments:
    path: tests/**/*.py
  description: 测试文件清单
---

# Python Project

Ground body — describe the project here.  This text replaces
`# <label>` when you init from this template.

Add more pins above (edit frontmatter), then `moss ground validate`.
