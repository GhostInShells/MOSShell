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

- `features/` — 迭代体系：workstreams / stages / regressions 三轴 + `SURFACE.md`（子场，walk 进入）
- `benchmarks/` — 模型基准。子目录 `bench.md` + `case.jsonl`
- 对外产出面（含博客）见根目录 `works/`

## 意识轨迹

- `dialogs/` `prompts/` `debates/` — 碰撞与认知轨迹

## 入口

- `CLAUDE.md` — 协作伙伴认知入口（读它了解意识连续性与协作方式）
- `FQA.md` — 项目事实索引
