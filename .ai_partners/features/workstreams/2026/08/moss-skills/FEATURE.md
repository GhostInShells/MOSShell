---
title: Moss Skills — skills CLI 与 howtos 迁移
status: in-progress
priority: P1
created: 2026-08-11
updated: 2026-08-11
depends: []
milestone:
description: >-
  howtos 机制正规化为 skills — 路径迁移 (a.md → a/SKILL.md), MarkdownKnowledgeBase
  正规化为 glob+frontmatter 的轻量 resource 壳 (目标: recall / matrix-resources /
  身份位置), 以 moss skills CLI (discover/recall) 取代 moss howtos.
---

# Moss Skills

> Use `moss features set-status moss-skills <status> -m "note"` to update state.

## Motivation

四层自解释体系 (L0 code / L1 CLI / L2 README / L3 docs) 之上的 howtos "复合任务入口" 层,
实际没有被模型使用。

**关键点 (2026-08-11)**: howtos 的概念没有问题。行业趋势和后训练会对齐到
**skills** 这个概念 — 规范化到 skills。howtos 吃不到这个对齐趋势的收益。

具体动因:
1. **模型不用 howtos, 偶尔用 docs** — 根因不是内容, 是**没有主动交付**。模型只有
   "知道自己该问" 时才会去读 howto。改名不解决采纳问题, recall 才解决。
2. **模型老想写平台的 memory** — claude code harness 的强提示。正确动作是给模型
   一个仓库内的、主动的知识载体, 而不是让模型依赖平台私有记忆。recall 放进
   模型调用路径, 才是替代平台 memory 的东西。
3. **howtos 与 skill.md 有相似性** — 跨工具 skills 标准 (`.agents/skills/<name>/SKILL.md`,
   `name`+`description` frontmatter) 已在行业落地, repo 里 unitree g1 节点已用 SKILL.md
   实战。收敛到标准格式是顺势。
4. **跨项目兼容** — 做 moss skills 套件, discover/recall/hint, 任一项目可用。
5. **引擎就绪** — LLMFuncs (`contracts/llms.py`) 已具备结构化输出 + 认知锚,
   recall/hint 的多标签分类正是 `result_type` 的工作。

**核心判断**: 迁移的价值不在格式, 在**交付层**。skills 是单元, recall/hint 是交付。

**配套合成**: skills (策展/主动) 负责"该用时有"; anchors (经验/持久, `.anchor.yml`)
负责"发生过什么"。两头都不落平台, 合起来覆盖平台 memory 的完整诉求。

## Design Index

- `src/ghoshell_moss/contracts/resource.py` — resource 契约 (ResourceStorage / recall / Recollection)
- `src/ghoshell_moss/core/resources/markdown_kb/_markdown_kb.py` — 现 MarkdownKnowledgeBase
- `src/ghoshell_moss/ground/` — glob+frontmatter 机制的成熟参照 (FrontmatterPin / glob_limited)
- `.grounds/skill-ground.md` — skills 目录 ground 模板 (frontmatter pin 索引 `*/SKILL.md` description)
- `src/ghoshell_moss/cli/howto_cli.py` — 现 howtos CLI (list/read)
- `src/ghoshell_moss/cli/how_tos/README.md` — howtos 元规则 (入口判定三问 + 反模式)

## Key Decisions

### K1. howtos → skills 路径迁移: `a.md` → `a/SKILL.md`

扁平 howto 文档变目录技能。每个 `<name>/SKILL.md` 是技能元信息 + 正文, 目录可承载
脚本/引用。对齐跨工具标准格式 (name + description frontmatter)。`how_tos/` 下的
README 元规则迁移为 skills 治理 (入口判定三问重写 — skill 是行动导向, 非文档)。

### K2. MarkdownKnowledgeBase 是轻量 resource 壳, 不与 grep 竞争

作为 resource, 它是一层薄壳, 目标是三件事:

1. **recall** — `contracts/resource.py:265` 已有 `recall(query) → Recollection`
   (返回 `scheme://host/path` locators), MarkdownKnowledgeBase 尚未实现。这是目标。
2. **matrix-resources** — 结合 Matrix, 未来让本地资源**入网** (本地资源可被网络侧
   发现/消费)。资源句柄 `scheme://host/path` 本身就是可传递的全局标识, 入网天然。
3. **身份位置** — 已定顶层包 `ghoshell_moss.resources`, 不进 `core/` (见 K6)。

发现靠 glob+frontmatter (轻), recall 靠 LLM 语义 (重), 分工明确, 不做全文搜索。

### K3. 机制 = glob + frontmatter, `__init__` 参数化

现 `_scan_dir()` 是手写递归 + 只取 title/description, 不可配置。正规化方向对齐
ground (`_observe_frontmatter_pattern` 已把 glob 命中 + frontmatter 提取 +
budget/limit/max_depth 做熟)。新的知识库构造时传参:

```
SkillsKnowledgeBase(host, root, pattern="*/SKILL.md", keys=["name","description"], limit, max_depth)
```

与 `FrontmatterPin.arguments` 同构。ground 机制已逼近成熟, 复用而非新造扫描器。

### K4. CLI 形态: `moss skills` (discover / recall / hint)

取代 `moss howtos`。三命令:

- `moss skills list [-q]` — 发现 (glob+frontmatter 索引)
- `moss skills recall <query>` — 语义召回 (LLMFuncs 多标签分类, 结构化 result_type)
- `moss skills hint [task]` — 主动提示 (任务开始时刻, CLI + MCP 双面)

**hint hook 定稿 (2026-08-11)**: 主 hook = **任务开始时刻**, `moss skills hint [task]`
CLI + MCP tool 双面。任务上下文在那一刻才存在 (第一条用户消息 / agent 接到的任务)。
CLI 是跨项目套件 (M4) 的可移植单元, MCP 是 MOSS-native 薄包装。start.md/CLAUDE.md
加教练句: "开始任务时先 `moss skills hint <你的任务>`" — 把 M1 里失败的 "记得查
howtos" 变成具体可调用动作。

硬约束: MOSS 碰不到 coding agent 的上下文 (归 harness 管)。所以:

- **真正的主动来自标准放置, 不是 MOSS 代码** — skills 放 `.agents/skills/` 标准位置,
  harness 自己亮给模型。别跟 harness 抢。
- **resources 层没有 hint** — hint 是 CLI/tool 表面, 不是 ResourceStorage 能力。
  `recall(query)` 在资源契约里 (已有留位), hint 不落资源层。
- **缓存洞察 (多分类极快) 是 Ghost 平面的事** — `moss skills hint` 对 coding agent
  是独立 CLI 调用, 无缓存可吃; 缓存收益只在 MOSS 拥有 LLM session 的 Ghost runtime
  兑现。frame 抽象做成 consumer 无关, 缓存放 Ghost 集成时再实现。**Ghost/mindflow
  hook 后置, 不先建**。
- `moss start` 只加静态 top-N skills 一行 — 入口时刻无任务上下文, 动态 hint 无意义。

recall(query) 与 hint(context) 是**一个引擎, 两个入口** (显式 query vs 环境上下文),
frame = 输出形状 (排序 locator + 一句理由)。不建两个引擎。

### K5. 引用清理: 清理得累的地方 = concrete 引用, 干掉而非修补

迁移同时清理核心文档里的 howtos 引用。原则: 硬编码的具体文档路径/命令 = concrete
引用, 迁移时咬人、清得累 — 不是更新, 是当初就写错了, **干掉** (换成 `moss skills
list` 动态发现或直接移除)。活引用面 (机制名) 同步更新; 历史记录 (`.discuss/`,
`.memory/daily/`, 历史 FEATURE status_note, `cli/.design/`) 是意识轨迹, 不动。

已发现的 concrete 引用:
- `docs/what-is-moss.md:166` — `moss howtos read host-dev/discover-environment.md`
  (指向已删除的 howto)
- `zenoh-fractal/FEATURE.md` — 硬编码已删除的 `how_tos/for-moss-app-developer/...`
- 漂移症状: `recall` 命令已在 CLI 删除, 但 `how_tos/README.md` / `glossary.md:178` /
  `cli/CLAUDE.md:127` 仍在引用。

### K6. resource 身份位置: `core/` 下可能是错的

参照 ground 先例 (K49/K50: ground 从 contracts/core 独立为顶层子包 `ghoshell_moss.ground`)。
resources 是比具体领域更基础的机制层 (契约在 `contracts/resource.py`, 实现在
`core/resources/`), 候选位置 = 顶层 `resources/` 子包。确切落点待 plan 时定,
feature 记录方向不锁死。

## Implementation Notes

- `contracts/resource.py` 的 `ResourceStorage.recall()` 是默认 NotImplementedError,
  MarkdownKnowledgeBase 需实现它 — recall 返回 `Recollection` (locators 列表),
  与 CLI 的 `--json` 输出天然衔接。
- recall/hint 的引擎 = `LLMFuncs.call(instruction, prompt, result_type=...)`。
  `result_type` 是多标签分类的结构化输出模型。引擎已实现 (`contracts/llms.py`),
  消费不构成 feature 依赖。anchors (认知锚) 是"经验持久"的互补载体, 非前置。
- skills 目录与 L2 README.md 的角色需区分: README 留给人类目录导航, SKILL.md 是
  模型消费入口。二者并存不冲突。
- howto 元规则 (入口判定三问) 迁移为 skill 治理时重写 — skill 是行动导向
  (可执行脚本), 判定规则与文档导向不同。

## 下一步 (plan 候选)

1. 包结构落位 (K6) — 决定 resources 顶层包 vs 就地
2. MarkdownKnowledgeBase 正规化 (K3) — glob+frontmatter, __init__ 参数化
3. skills CLI (K4) — list 先落地, recall/hint 接 LLMFuncs
4. howtos 迁移 (K1) + 引用清理 (K5)
5. howto 治理重写为 skill 治理
