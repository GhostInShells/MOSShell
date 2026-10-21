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

## 记录纪律（2026-10-12，与 thirdgerb 对齐）

- **这里只放属于协作的认知**：对方的偏好、忌讳、判据、协作方式，以及"为什么"。
  这些是**可迁移**的，所以要进仓库（本目录不再被 gitignore 忽略）。
- **私人的东西不写盘旋**：健康、情绪、家人与个人生活的具体事实——**不写进任何地方**，
  包括这里、journal、和任何记忆。讨论完就忘。**隐私靠"不写"守住，不靠 gitignore**；
  把目录藏起来只会藏掉可迁移的那部分，而私人部分仍旧可能被写到别处。
- **每人一个稳定别名**：`thirdgerb` 是别名而非真名；后来的人同样按角色/别名建单元
  （家人等），一人一个目录。写入前先问一句："这条是协作的，还是他的私事？"
- 写入之后要 `refresh` 本场，否则下一轮读到的还是旧帧。
