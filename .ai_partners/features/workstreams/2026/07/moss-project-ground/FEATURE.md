---
title: MOSS Project Ground
status: draft
priority: P1
created: 2026-07-23
updated: 2026-07-23
depends: [ghost-ground]
milestone: 0.1.0
description: >-
  MOSS 仓库自身的项目认知场 — 通过 ground 协议组织 features / .design /
  .discuss / regressions 等认知资产的寻路, 让进入 MOSS 的模型实例通过
  ground 而非手工探索发现项目结构.
---

# MOSS Project Ground

> Use `moss features set-status moss-project-ground <status> -m "note"` to update state.

## Motivation

MOSS 仓库有丰富的认知资产 — features / .design / .discuss / specs /
regressions / howtos / docs — 但它们的发现全靠 `moss features list` +
`moss docs list` + `moss howtos list` + 人工记忆. 每次新模型实例进入
都要从头学, 即使有 `moss start` 和 `moss codex architecture` 作指引,
寻路成本仍然高.

Ground 协议提供了场的开合 + pin 注视 + 法链继承. MOSS 自身的项目认知场
就是用 ground 把这些认知资产组织成一个可 open 的场体系. 模型实例进入时
open MOSS 项目根, 看到的不再是一个个需要记忆的 `moss` 命令, 而是一个
结构化的认知表面.

这是 `--mode meta` (或 `moss_self_project`) 下的默认 ground.
与 Dolores 的 ghost_home ground 是两个 GroundSet: 一个是被观察的项目
(MOSS 本身), 一个是 ghost 自身的认知基建.

## Design

### GroundSet = MOSS 项目根

项目根目录放 `GROUND.md`, 作为 L0 认知入口. frontmatter 里预设一组
推荐 pins, 让进入的模型不需要手动探索即可看到项目全景:

```yaml
$id: moss-project
label: MOSS
pins:
  - label: features
    verb: frontmatter
    arguments: {path: "$GROUND/.ai_partners/features/workstreams/*/FEATURE.md"}
    description: "活跃 workstreams"
  - label: designs
    verb: glob
    arguments: {path: "$GROUND/.design/*.md"}
    description: "设计文档"
  - label: specs
    verb: glob
    arguments: {path: "$GROUND/src/ghoshell_moss/**/SPECIFICATION.md"}
    description: "SPEC 规格"
```

这些 pins 不是系统规则 — 是 MOSS 维护者手工 pin 上去的第一人称注视.
每个使用 MOSS 的模型实例 (或人类) 都可以自己 pin 更多、unpin 不需要的.

### 寻路层级

| 层 | 内容 | 方式 |
|---|---|---|
| L0 body | MOSS 是什么、核心概念 | 法链从祖先 GROUND.md 自动加载 |
| L0 pins | features/designs/specs 等动态资产 | pin 观察, 每帧对账 |
| Sub-fields | 进入具体 feature 目录 | `open(dir)` — 子目录 GROUND.md |
| bash | .discuss / git log | 不需要默认场, bash 足够快 |

### 与现有发现体系的共存

`moss start` / `moss features list` / `moss codex architecture` 不变.
Project ground 是额外的发现层, 不替代它们. 它在模型通过 MCP 接入,
`moss start` 第一句话之后, 提供一个文件系统层的结构化视角.

## Key Decisions

- **GROUND.md 放在项目根目录** — 这是 MOSS 项目自身的认知入口.
  不放在 `.moss/` 或 `.ai_partners/` 下: ground 是项目认知面, 不是工具配置.
- **预设 pins 是建议不是规则** — 模型可以修改、unpin、自己 pin.
- **不替代 CLI 发现体系** — start / features / codex 体系保持为程序化入口,
  ground 是空间化补充.
- **与 ghost_home GroundSet 分离** — project ground 是 "我在看什么项目",
  ghost_home ground 是 "我是谁". 两套 GROUND.md, 两个 GroundSet.

## Implementation Notes

- 在 MOSS 项目根创建 `GROUND.md` 手工脚手架
- `ghost-ground` feature 的 Grounds concrete 实现可用后, 从 `--mode meta`
  的 ghost 配置中实例化 project GroundSet
- 首版只做 L0 入口 + 预设 pins. 子目录 feature ground 是渐进式添加的
