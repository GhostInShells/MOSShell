---
created: 2026-07-23
depends:
- ghost-ground
description: MOSS 仓库自身的项目认知场 — 通过 ground 协议组织 features / .design / .discuss / regressions
  等认知资产的寻路, 让进入 MOSS 的模型实例通过 ground 而非手工探索发现项目结构.
milestone: 0.1.0
priority: P1
status: completed
status_note: 根场 (@claude.md + grounds pin) 落地, glob_limited 递归重写 + observe 清理 + snapshot,
  裸测端到端验证通过
title: MOSS Project Ground
updated: '2026-08-13'
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
  shebang 决定解释器. 执行 cwd = `$GROUND`, env 注入 `GROUND` + `CWD`.
  场作者背书 = Makefile 级信任. arguments 极简: `{ref, timeout, budget}`,
  无 `at` (executor 自知家在场根), 无 `run` (禁止内联).
- **[2026-08-10] exec 拒绝信息区分 (修正 2026-07-25)** —
  原决策 "绝对路径 / `../` / 缺 `+x` 一律 `[missing]`" 被实现推翻了:
  三类情况各自渲染不同标记 — `[missing]` (文件不存在), `[not executable]`
  (无 +x), `[outside ground]` (授权拒绝). 区分让住客可诊断可修复;
  pin 的 ref 是场作者自己声明的, 不存在信息泄漏问题.
- **[2026-08-10] max_depth 实现落地** —
  glob / frontmatter pattern 模式的 `max_depth` 参数正式实现 (SPEC §4.1):
  递归深度上限 + 场边界停止 (某层出现 match 后不下钻该子树).
  对 `**/GROUND.md` 场发现即 "不穿透场" — 发现子场后不深入子场内部.
  `ls` 的 observe 同步修复 effective_depth, 前端 frontmatter 无 fm 的 observe 对齐 render.
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

## 失败模式 (2026-08-13)

渲染重写 (RenderedView) 落地后, human review 暴露两个 silent todo —
与 audio (voice-input-state-machine) 同根: **讨论时说关键, 实现时交付
违背设计的东西**. 这次 review 时符合 CLAUDE.md 失败模式预判地暴露.

### glob_limited 假 boundary stop

`glob_limited` 用 `Path.glob` 一次性递归 + `len(parts) <= max_depth`
后过滤, 而非显式递归. 后果:

- `max_depth` 语义是 "path 组件数" (含文件名), 不是 "递归深度" —
  `max_depth=1` 只匹配 base 自己, `=2` 才到一层子场, 与直觉反.
- SPEC §4.1 的 "boundary stop (发现子场后不深入)" 是假的 — `Path.glob`
  已递归到底, 代码里 `matched_dirs` 后过滤是后验模拟, 不是边递归边停.
- 用 `Path.glob` 那点 scandir 优势换来假 boundary stop 骗过 review.

### observe 残留 hash 对账

`1423c542` 移除 hash 对账后, observe 里的 `mtime` 计算和
`_observe_frontmatter_pattern` 的读文件拼 parts 没清干净. `mtime`
是孤儿字段 (全仓无人消费), 读文件拼 parts 是双重读取 (render 阶段
`_content_frontmatter_pattern` 再读一遍).

### 决定 (已实施 2026-08-13)

- `glob_limited` 重写为显式递归 (iterdir + fnmatch + ignore), 拆两个
  正交语义: `recursion` (目录层深度上限, 0=不递归/N=N 层/None=无限) +
  `stop_on_match` (防穿透, 目录直接含 match 即不下钻, base 豁免).
  三个调用方各取所需: glob 标准语义 (recursion), frontmatter 防穿透
  (recursion + stop_on_match), markdown_kb 深度上限 (recursion).
  `recursion` 计目录层数而非 path 组件数 — `recursion=1` = 一层子场,
  修正了 "max_depth=1 只匹配 base 自己" 的反直觉语义.
- observe 退化为 "exists + 计数 + exec payload", 删 `mtime` 死字段和
  `_observe_frontmatter_pattern` 读文件拼 parts, 内容读取全交 render.
  file observe 也不再读内容 (range 切片是 render 的事, size 报全文件).

### 对账归位: snapshot (同轮)

hash 对账没被再杀一次, 而是归位到正确的层: Ground 实例 (进程内) 的
`snapshot(ack_hash=None)` — 渲染 + 感知 digest + 变更标记, render 保持
纯内容. 要点:

- **对账目标 = 渲染文本全量** (`view.to_markdown()` 的 sha256), 不是
  源文件 — 信号统一 (所有 verb 同一语义), 且正是模型感知的东西.
- **触发 = render 动作, auto-advance**: 首次建基线不标 changed; 之后
  相对基线变化标 changed 一次, 渲染即承认 (无 update 动词).
- **返回 `Snapshot{view, hash, changed}`** — 数据结构 RenderedView 不动,
  hash 伴随返回, 影响面最小.
- **状态存活**: 进程内 Ground 实例属性, 不落盘; CLI 单次调用无跨调用
  记忆; channel 会话是唯一消费者 (缓存写入不并发安全, 须单 owner).
- 块级粒度 (哪个 label 变) 不做 — channel 持上一 view, 变化时自己按
  label diff 两个 RenderedView, hash 是廉价脏标记.

## 裸测反馈与设计确认 (2026-08-13)

根场落地后旁路 agent 裸测端到端可用, 报 5 问题 + 1 工件. 关键教训:
**裸测是讨论触发器, 不是权威 bug 报告** — fresh agent 无设计上下文,
会把设计特性误判成缺陷.

### #2 误判: `$CWD` 锚是特性

裸测指 grounds pin "描述是项目索引却用 $CWD, walk 后索引消失". 这是 false
positive. 设计意图 (human 校准): **进场模型已读过 grounds 索引、知道
recursion=1 的场位置; 之后 walk 重放同一索引是冗余**. $CWD 让索引随 walk
变相对当前位置的场, 是故意的.

锚点分野 (关键):
- 索引型 (稳定, walk 折叠 TOC) → `$GROUND` — 如 nodes `index` (`$GROUND/**/NODE.md`)
- 观察者跟随型 (随 walk 漂移) → `$CWD` — 如 `here: ls $CWD`、`focus: file $CWD/NODE.md`、根 `grounds`

### #3 推迟: `chain +N` 含自身

spec §7.3 "ancestor up to $HOME" 只含祖先, 实现连自身也算 (root=+1).
不拍板 — ground 给模型用, 体验问题由模型 dogfooding 定, 人类不越位.

### max_depth 不用改 (关键决策落定)

human 主张 max_depth 改名 recursion 消歧义, 上一轮模型主张就叫 max_depth.
两边立场都落了: pin 契约层保留 `max_depth`, glob_limited 内部拆 `recursion`
+ `stop_on_match`. 结论 max_depth 无歧义, 不改字段.

### 已修: meta / help / $id / 清理 / 术语

- `meta` walk 报 `pins:(none)` (与 render/observe 不一致): cmd_meta 原
  `_run_one` 开 bare ground 不查祖先 → 改与 observe 相同 `_find_ancestor_ground`
  + `open(doc=祖先)`.
- `render --help` 谎称 walk 自动 "cwd listing" → 改措辞 (cwd listing 只来自
  $CWD ls pin, 非自动).
- render header 加 `$id` (存在才渲染): ViewHeader 加 id, render_context/walk
  透传 ground_id.
- 4 个 GROUND.md 的 `$id` 是机械分层前缀 (moss-project/...) 言之无物 → 移除.
  anchor 机制才是身份正主.
- 术语 `fields` → `grounds` (ground/field 两套词对齐 ground).
- 删散落工件 `test-ground-output.md`.

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