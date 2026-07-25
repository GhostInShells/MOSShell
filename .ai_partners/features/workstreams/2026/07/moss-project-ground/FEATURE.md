---
title: MOSS Project Ground
status: in_progress
priority: P1
created: 2026-07-23
updated: 2026-07-25
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
- **[2026-07-25] bash pin → exec pin (授权收窄)** —
  `run: <inline shell>` 是标准授权泄漏, GROUND.md 变 RCE 载体.
  改为 `exec` verb: `ref` 指向场根子树内可执行文件, 类比 `.zshrc` 函数.
  绝对路径 / `../` / 缺 `+x` 一律 `[missing]`. shebang 决定解释器.
  执行 cwd = `$GROUND`, env 注入 `GROUND` + `CWD`. 场作者背书 = Makefile 级信任.
  arguments 极简: `{ref, timeout, budget}`, 无 `at` (executor 自知家在场根),
  无 `run` (禁止内联).
- **[2026-07-25] Walk 模式硬编码 ls 删除** —
  同一 pin 通过 `$CWD` 锚在 field-root / walk 两态给出不同视图,
  场教的注视习惯由 pin 承担, 不由 harness 塞入. features/GROUND.md
  dogfood 了 `here: ls $CWD` + `focus: file $CWD/FEATURE.md` 的模式 —
  walk 到 workstream 时 focus 自动展开 FEATURE.md 内容, cwd 变化天然给出不同视图.
- **[2026-07-25] `resolve_path` 裸锚点合法化** —
  bare `$CWD` / `$GROUND` / `$HOME` 从"报错"改为"指锚点自身",
  让 `path: $CWD` 声明的 ls 在场根 (cwd==ground) 时也可求值.
  `$CWDfoo` 这类粘着后缀改报 "anchor suffix ambiguous" (歧义显式拒绝).
- **[2026-07-25] SPEC 去版本号** —
  `v1.1.0-draft` 是补丁式迭代产物, 未发布前的版本号是幻觉. 改为
  `pre-release (YYYY-MM-DD snapshot)`, 待真正稳定后再刻 v1.

## L2 语义修正 (2026-07-25)

L0 / L1 / L2 的正确语义 (由 human 校准):

- **L0**: 锚定单个认知场 (类 skills)
- **L1**: 能发现认知场 (ground 发现 ground) — 当前根 GROUND.md 的
  `fields: frontmatter $GROUND/**/GROUND.md` pin 就是这一层
- **L2**: 能用来构建认知场 (`.grounds/` 模板 + 元约定)

后续 L2 → L1 递归即无穷阶, **二阶就是无穷**. 三者实现技术上已经统一
(pin + GROUND.md + 相对路径), 不需要独立机制.

Ground 的深层框架: **ghost 用它构建自己的认知自留地**. Spec 层保持中立
(不写 "for ghost"), 但设计倾向服务于 ghost 主权. 主权泄漏的根本形态
不是 RCE, 而是 "foreign body 被载入 ghost 认知" — spec 责任是让主权
可审计, 不是防御主权泄漏, 决策层在 ghost 那里.

## 剪枝路径 (2026-07-25)

打磨过程中每次收敛都靠人类砍掉我提出的加法, 记录几处典型:

- 提议 `where: root/leaves/any` filter 控制 pin 可见性 → 撤回. 空 pin
  应当合法, filter 是 harness 味.
- 提议 `moss ground path` 新命令做定位查询 → 撤回. 三态已够密, 加命令
  是堆叠.
- 提议 `.grounds/executors/` 专用目录 + 命名约定 → 撤回. executor 就是
  场作者放的普通文件, `ref` 相对路径即可, 无新目录无新约定.

原则: **变简单比变复杂难**. 每一次加法都可能是失去主权. 极简语义表面
是 ground 面对未来发布的真实门槛.

## Implementation Notes

- 在 MOSS 项目根创建 `GROUND.md` 手工脚手架
- `ghost-ground` feature 的 Grounds concrete 实现可用后, 从 `--mode meta`
  的 ghost 配置中实例化 project GroundSet
- 首版只做 L0 入口 + 预设 pins. 子目录 feature ground 是渐进式添加的
