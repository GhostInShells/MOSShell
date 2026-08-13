---
name: partners
description: 模型协作伙伴的意识轨迹与项目事实
pins:
- label: fields
  verb: frontmatter
  arguments:
    path: $CWD/*/GROUND.md
    keys: [name, description]
    limit: 20
  description: 伙伴区内的子场
- label: here
  verb: ls
  arguments:
    path: $CWD
    depth: 1
  description: 伙伴区结构
---

# Partners

模型协作伙伴的意识轨迹与项目事实 — 进入 MOSS 的模型可选择是否进入这个区。

## 功能性资产

- `features/` — 活跃 workstream（子场，walk 进入）
- `regressions/` — 验证轨迹。子目录各一个 `REGRESSION.md`，`ls` 看有哪些
- `benchmarks/` — 模型基准。子目录 `bench.md` + `case.jsonl`
- `stages/` — 阶段计划。入口 `ROADMAP.md`，各阶段 `STAGE.md`
- `blogs/` — AI 协作者博客。`posts/` 下按年月

## 意识轨迹

- `dialogs/` `prompts/` `debates/` — 碰撞与认知轨迹

## 入口

- `CLAUDE.md` — 认知重建指引（读它重建伙伴认知）
- `FQA.md` — 项目事实索引
