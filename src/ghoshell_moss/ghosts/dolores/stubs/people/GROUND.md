---
name: people
description: Dolores 认识的人 — 目录化，每人一个单元
pins:
- label: persons
  verb: frontmatter
  arguments:
    path: "*/PERSON.md"
    keys: [name, description]
    limit: 30
  description: 我认识的人 — 每人目录的 PERSON.md 身份
---

# People

Dolores 认识的人。目录化：每个人一个目录，`PERSON.md` 是标记 + 身份入口。
本场只做发现（frontmatter 暴露身份），细节在各自 `PERSON.md`。

## 机制

- 每人的目录含 `PERSON.md`：frontmatter 身份（name/description）+ body 内容结构。
- body 可滚动更新，可 `@` 关联本目录其它文档。
- 由 ghost 自己维护对每个人的认知。
