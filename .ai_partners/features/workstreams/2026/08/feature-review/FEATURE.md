---
created: 2026-08-13
depends:
- memento-cli-and-agent
description: 基于 FEATURE.md 声明的 feature review — 零上下文 review (遗忘测试)，声明 vs 交付对账，
  default-to-fail。MOSS 做确定性层 + review policy + brief 生成，不做执行层 (语义 review 住在
  coding-agent 集成架构)。
milestone: v0.1.0
priority: P0
status: in-progress
status_note: '2026-08-14 对齐轮定案: 零上下文 review = 遗忘测试, MOSS 做声明+确定性+协议不做执行层, 旁路定位+流程发起'
title: Feature Review — 零上下文 review (遗忘测试)
updated: '2026-08-14'
---

# Feature Review — 零上下文 review (遗忘测试)

> 人类架构师 + deepseek-v4-flash，2026-08-13 会话定案。本 FEATURE.md 指导实现；
> 验收判据自带 dogfood：**明天的开发成果用本 agent 自 review**。

## Motivation

2026-08-13 一个会话内，四个 workstream 暴露同根失败（全部记录在各自 FEATURE.md 失败模式节）：

| workstream | 失败 | 根因表述 |
|---|---|---|
| voice-input-state-machine | Listener 继承旧 VoiceController ABC + `__import__` 劫持 | deliver-first 堆屎山 |
| moss-project-ground | glob_limited 假 boundary stop、observe 残留孤儿 mtime | 两个 silent todo |
| warrant | KD5 双模式被静默降级为单 host 模式 | 无主动交流，直到 human 追问 |
| matrix-operator | counter 单测全绿但内核层 7 条致命故障路径 | "单测全绿不足以撑内核质量关" |

共同根因 = CLAUDE.md / `.ai_partners/CLAUDE.md` 记录的模型失败模式：**讨论时品味高，
交付时 silent todo + 交付优先**。且失败模式被记录 ≠ 会拦住下一个实例——每个实例是全新
上下文，读到教训和遵守教训是两回事。

**结构性事实**：模型的读代码/评方案品味显著高于交付保真度。今天的全部问题，都由
human 的预判 + 摩擦力观察抓住——而那是个人纪律，不是机制，会磨损、会漏。

**本 feature 的战略判断**：用模型的长处（读代码、评方案、对账）去兜它的短处（交付
保真）。在开发循环中引入**声明对账审查**——这个审查一直是应该做的，之前没做是因为
不想替 claude code 造工具；现在 memento agent 基建能用，它变成 MOSS 原生能力，
跨平台可复用。

**开发策略约束（人类架构师 2-3 收益规则）**：任何应用实现至少 2-3 个收益才动手。
本 feature 收益：① 交付保真闸口；② 验证 FEATURE.md 作为机器可读规格载体的价值；
③ dogfood memento agent factory 基建；④ 跨平台（不绑定任何 coding 平台）。

## 对齐轮 (2026-08-14) — 概念收敛，最终方案前的锚点

> 人类架构师 + 模型，2026-08-14 讨论收敛。本节记录对齐结论，**修订**下方部分原 Key Decisions
> （KD1、KD2 及"闸口"定位，见文末"修订清单"）。最终方案在此节基础上展开，不重开已收敛命题。

### 战略定案：零上下文 review = 遗忘测试

feature review 本质 = **零上下文 review**，即"遗忘是最高形式的编码"的机制化：质量判据是
"零上下文化身能否从 FEATURE.md + 交付重建认知、知道如何开始"。review 的 stance 是**重建**
（"你是下一个化身，上手试试，报告卡在哪"），不是**对账/verify**（"审计清单"）——重建让
default-to-fail 自然涌现（"我卡在这"本身就是 fail），且比审计更抗橡皮图章。

### 人类协作经验的 ground truth

动机的更深层是六条协作经验，机制化时逐条落位：

1. **交付引力 / 不打断** → review 是独立的一轮，不 inline 打断（非致命问题等一轮完再 review）。
2. **零上下文化身 review** → 开发模型自己的 dogfooding 不作数，必须压缩到初见。
3. **手动零上下文 review 太依赖人** → coding agent 独立 review 本身有用，机械化编排侧。
4. **A 开发 / B review，反馈发给 A** → reviewer 必须与 developer 是同类对等会话（同为 coding agent）。
5. **保留设计上下文作状态** → anchor 机制原生实现（见"anchor = 意图测试"）。
6. **质量判据 = 新会话能知道如何开始** → 聊方案→开发→验收三循环，review 在验收点。

### 分层定案：MOSS 做什么、不做什么

**MOSS 做**（声明层 + 确定性层 + 协议层，均无模型循环）：

- 确定性层：机械对账（status/status_note 一致、文件拓扑、显式"待实施/失败模式" vs 代码有无）。
- review policy：切面集合 + 品味标准 + default-to-fail + 输出 schema，版本化声明。
- review brief 生成：聚合材料（FEATURE.md + diff + 触及文件 + 可选 anchor + 确定性结果）→ portable 协议。

**MOSS 不做**（执行层）：语义 review agent（真正"读代码、报卡壳"）**不在 MOSS 运行时**，
住在未来的 coding-agent 集成架构。理由：① 零上下文在任何新 coding-agent 会话天然有；
② enforcement 来自 schema + 确定性层，不来自 agent 基座；③ 跨平台可复用来自"不打包模型"。
MOSS 的 pydantic-ai agent 至多是 `--standalone` fallback（CI / 无 coding-agent），非主路径。

### 五个切面 → 分层路由

零上下文 review 套到五个切面，真相标准散在不同层，不可融合、只可分层路由：

| 切面 | 对象 | 真相标准 | 落层 |
|---|---|---|---|
| 致命问题 / silent todo | 交付 vs 声明 | 对账，file:line 证据 | 机器 + agent |
| 摩擦点验证 | 声明本身 | 重建（自己的卡壳） | agent |
| 代码质量 | 代码 | 味道清单，阈值是品味 | agent + 声明的味道标准 |
| 风格品味 | 代码 | 纯判断，项目方言 | 人类 L2/L3，agent 只 surface |
| 用交付 review 方案 | 设计 | L3 判断，交付是证据 | 人类，agent 只 surface |

### 项目级 policy（共识）

"不同项目、不同品味"的兼容方式 = 把品味从 agent 里拿出来，放进**项目级 review policy**
（切面集合 + 品味标准 + 输出 schema），per-feature 可在 frontmatter 覆盖。切面集合本身可声明
（剪枝），不硬编码——这是本 feature"剪枝策略"的完整形态。

### anchor = 意图测试（经验 5 的原生机制）

`PydanticAIAgentAnchor` 冻结 instruction + tools + model + turns（一帧认知流），**价值即 review，
不是 replay**——经验 5 的原生实现。它在 review 里开启第二种测试：

- **遗忘测试**（零上下文化身 vs FEATURE.md）——声明能否在遗忘后重建。
- **意图测试**（交付 vs design anchor）——交付是否忠于被压缩掉的意图。

anchor 还解锁**区分两类 FAIL**：意图丰富而声明模糊 = 声明不合格（重写 spec）；声明与意图都清楚
而交付对不上 = 实现没跟上（写代码）。无 anchor 分不开。**定稿时 dump / 交付前消费**。
v1 lean：留 anchor 输入槽、不接（意图测试最易橡皮图章，需最强防锈脚手架）。

### 对原有 Key Decisions 的修订

- **KD1（等价实现复用 factory 基建）→ 修订**：MOSS 不自己做执行层，只做确定性层 + policy + brief；factory 复用退为 `--standalone` fallback。
- **KD2（agent 即指令）→ 修订**：指令/协议是声明，enforcement 靠 schema + 确定性层，不靠"写进 .py"。
- **"声明对账闸口"定位 → 修订**：旁路服务人类 + 流程发起，不是 commit 前硬 block。

### 开放点（最终方案展开，不重开已收敛命题）

1. 确定性层边界（机械检查做多大）。
2. review brief 的结构化程度（多接近机器可消费的 schema）。
3. v1 是否接 anchor 槽（lean：留槽不接）。

## Design Index

- 参考实现基建：`memento-cli-and-agent` (FEATURE.md) + `src/ghoshell_moss/agents/memento_pydantic_agent/factory.py`
- 契约：`src/ghoshell_moss/agents/contract.py` — `MementoAgent.invoke`
- agent .py 先例：`agents/memento_agents/calc.agent.py`、`src/ghoshell_moss/agents/explore.agent.py`
- CLI 先例：`src/ghoshell_moss/cli/features_cli.py`（`moss features` 组）、`memento_cli.py`（agent 调用 + `_build_agent`）
- 失败样本（对账对象的设计来源）：今日四个 FEATURE.md 失败模式节

## 设计要点

### 对账协议：声明 → 证据 → 裁决

review agent 的核心操作是**交付 vs 声明的 diff**，不是代码质量评审：

1. 解析 FEATURE.md：frontmatter（status/priority/depends/milestone）+ Key Decisions 节
   + 失败模式节 + Implementation Notes + 状态表（子任务/待做）。
2. 每条**声明**产出一个对账项：`{声明, 代码证据, 裁决}`。
3. **default-to-fail**：声明决策必须找到对应代码证据；找不到证据 = 违约（FAIL），
   **不是**"假设已实现"。与实现者的 default-to-done 正好相反。
4. 重点对账对象：显式记录的"决定（待实施）"、Key Decisions、status 与 status_note 的
   一致性、以及实现是否落在声明的工作流内（不越权、不悄悄换方案）。
5. **absence 是机器可检测的信号**——glob 假 boundary、hash 残留，本质都是
   "spec 说重做/删了，代码还是旧的"，是对账能抓到的 diff。

### 注意力压缩（对 human 的价值）

输出不是全量扫描，是**被 flag 的 diff 清单**。human 只 review flag 出来的点位，
把"带预感扫全部"压缩成"只看偏差"。这是本 feature 削减 human 切换上下文次数的机制
（human 注意力是链上唯一不可并行资源，见 STAGE.md 的 human L2 治理）。

### 防锈：review agent 自己的失败模式是橡皮图章

模型天性倾向同意。设计上对抗手段：

- **default-to-fail**（见上）——找不到证据 = 违约，不是"看起来实现了"。
- 输出必须逐条列出声明及证据路径，不允许笼统的"整体通过"。
- 每条裁决带证据引用（文件:行），无证据的裁决无效。
- human 的预判仍是最后一道线——agent 压缩注意力，但替代不了"预感"层。预期管理：
  机械对账全抓，friction 观察部分抓。

## Key Decisions

1. **等价实现，不暴露 memento agent**。feature review 是独立 agent：复用
   `memento_pydantic_agent.factory` 的基建（sandbox + get_interface 注入 +
   pydantic-ai Agent），但**不依赖 memento 轨迹/存储**——invoke 走
   `memento=None` 降级基线（纯内存单轮，不写 `.memento/`）。不在 review 流程中
   挂 memento 的 line/commit 机制。

2. **agent 即指令**。review 行为写成 agent .py 文件（含 `__model__`/`__owner__` dunder +
   review 指令 + 所需 capability import），与 calc.agent.py / explore.agent.py 同范式。
   agent 是独立模块，import 即用，CLI 只是装配皮。

3. **命令落点：`moss features review <name>`**。挂在现有 `features_cli.py` 组下
   （对齐 `moss features` 组，不新建顶层命令）。`<name>` 解析到对应 FEATURE.md。
   `moss features check`（已有，列未完成工作流）与 review 是两层：check 提醒、
   review 对账裁决。

4. **调用时机：声明开发阶段性 commit 之前**。每个触及 feature 代码的阶段性 commit
   前，先 `moss features review <name>`，产出对账清单后随 commit。完整流程 =
   review → 修（违约项）→ review → commit。`moss features check` 仍作为非阻塞提醒。

5. **对账输入 = FEATURE.md + git diff/status**。agent 需要读到：当前工作树 diff
   （git status + git diff 触及文件）、FEATURE.md 全文、被改代码。能力经
   `CAPABILITY_FACTORIES` 注入（参照 `get_interface` 的注入模式）。

6. **先做最痛的**。首版对账聚焦：显式"待实施/失败模式"声明 vs 代码（今天四个失败
   的类型）、status/status_note 一致性。通用代码质量（魔法值/短变量名/无日志）不进
   首版范围——那些是 CLAUDE.md 已知问题，不是本 feature 的增量。

## Implementation Notes

- **文件拓扑**：
  ```
  src/ghoshell_moss/agents/feature_review.agent.py   ← review 指令 + capability import
  src/ghoshell_moss/cli/features_cli.py              ← 加 review 子命令
  src/ghoshell_moss/agents/_feature_review.py         ← 解析 FEATURE.md → 声明清单 (纯函数, 可测)
  ```
  `_feature_review.py` 的声明解析是纯函数层：frontmatter 解析 + Key Decisions 提取 +
  失败模式节提取 → 结构化声明列表。可脱离 agent 单独单测。
- **agent 输入组装**：`invoke(user_prompt=...)` 收什么——FEATURE.md 全文 +
  `git status --short` + 触及文件 diff（按声明相关文件截断）。capability 注入
  `git diff`/`read_file`（factory 已支持 `injections` 参数注入，见 factory.py `injections`）。
- **factory 复用**：`factory(path, injections=..., cwd=...)` 返回 `MementoAgent`，
  调 `invoke(memento=None)`。`__model__` 由 agent .py 声明或 `ANTHROPIC_MODEL` env。
  thinking 默认 ON（质量优先）；如需省钱可 `__thinking__ = False`（机械对账 agent 的
  候选场景，但首版默认 ON）。
- **裁决输出契约**：markdown 对账清单，逐条 `- [PASS|FAIL|WARN] 声明 → 证据(file:line)`。
  FAIL = 声明无代码证据；WARN = 部分证据 / 声明模糊。无"整体通过"式输出。
- **验收（dogfood 判据）**：明天开发本 feature 的阶段性 commit，本身用
  `moss features review feature-review` 对账。它必须能抓到自己实现里的
  声明 vs 交付偏差——**抓不到 = 失败**。同时用它对一个今日失败工作流（如
  moss-project-ground 或 warrant）做回溯对账，验证能复现 human 抓到的违约点。
- **`--ai` flag 纪律**：输出走 `print_simple_table`/纯文本，`--ai` 可用（CLAUDE.md
  CLI 纪律：所有新命令必须支持 `--ai`）。

## 工具面 (授权能力, 2026-08-13)

review agent 的纯只读工具面已落地 `src/ghoshell_moss/tools/`。全部只读,
**import surface = authorization whitelist** — agent .py 导入哪个工具,
哪个工具就是它的能力面, 没导入的不可达。

| 工具 | 形态 | 能力 |
|---|---|---|
| `git_status` | async @cli | `git status --short` (repo 根) |
| `git_diff` | async @cli | `git diff [--stat] [-- path]` (repo 根) |
| `read_file` | sync | 读文件带行号, 可选行范围, 复用 FileEditor.view |
| `list_files` | sync | 列目录, 复用 FileEditor.file_list |
| `glob_files` | sync | 相对 repo 根的 glob 路径匹配 |

约束:

- **git 白名单结构性内建**: 前缀固定 `git + subcommand`, 写命令 (add/commit/
  push/reset/...) 结构化不可触达, 不需要运行时黑名单。`cwd` 惰性绑定 repo 根
  (`@cli` cwd 新增 callable 支持, 修掉 facade 默认 cwd=`.moss` 的路径偏移)。
- **fs 自持 repo-root 边界**: resolve + relative_to 拒绝越界 (file editor 只 hint
  workspace_root, 边界是 tools 自己的职责)。复用 FileEditor 契约, 不另开读路径。
- **grep 不入面**: 拿到文件内容后 python 上下文做加工即可 (沙箱内 str/正则)。
- **git add/commit 等写操作不入面**: review 是只读闸口, 不产生写操作。

agent 侧接入: `__interfaces__ = [git_status, git_diff, read_file, list_files, glob_files]`
展开 interface。git 工具 async, agent sandbox (同步 exec) 需 `asyncio.run` 包裹;
fs 工具同步, 直接调用。测试: `moss memento agent parse agents/memento_agents/git_probe.agent.py`
看完整 interface 展开。

## Validation Plan

1. `_feature_review.py` 单测：给定 FEATURE.md 样例 → 声明清单结构正确（frontmatter +
   Key Decisions + 失败模式节均提取）。
2. 回溯对账：对今日四个失败 workstream 跑 review，验证能 flag 出 human 已记录的违约点
   （如 ground 的 glob_limited / warrant 的双模式降级）——这是本 feature 存在理由的直接证据。
3. dogfood：本 feature 自己的阶段性 commit 前跑 `moss features review feature-review`。

## 讨论回顾与疑点 (2026-08-14 追加)

> 本文档收敛后不会反复被读，故采用**追加**而非改写的方式记录后续讨论。前两节
> （2026-08-13 原始设计 + 2026-08-14 对齐轮）冻结不再改；本节是唯一随讨论演进的活页，
> 下一轮确定具体执行方案后继续追加于此。

### 讨论回顾

从"四个命题"起步 → 七个维度 → 收敛到"零上下文 review = 遗忘测试"，再推进到详细方案的讨论。
详细方案由人类架构师给出，模型推导与之在主线收敛（MOSS 不做执行层），但**实现细节在模型
早先的记录里被误植了一处方向性错误**（见"现在的变化"）。

### 共识（含详细方案）

1. `moss features review` = **meta-prompt + 文档发现**，仅此而已，无 review 逻辑、无机械检查。
2. review 知识 = **`features/reviews/` 下的 step 命名 prompt 文档**（`0_xxx.md`），n 个文档 = n 个
   切面，形如 skill.md，换项目换目录即可。
3. **锦囊模型（惰性披露）**：specification 只*提示*有 `moss features review` 这个实机，不把 review
   内容加载进模型注意力（不污染注意力）；调用时才呈现基础原理 + n 个切面文档。
4. **执行 = sub agent（无则人类操作）**：模型派 sub-agent（天然全新上下文 = 零上下文化身），
   输入 = instruction + feature.md + 切面文档；无 sub-agent 机制时人类执行。
5. 只读、无副作用，输出可导出对账；但**图章无意义**，价值在真实语义发现。
6. **验证 = 预期 vs 测试结果对照（人类判断）**，不是机械判断。

### 现在的变化（相对"对齐轮"节的修订）

- **"确定性层（机械对账）"去掉**。方向性错误：模型早先结论"旁路非闸口"，却又在"MOSS 做"里
  塞回一个带 exit-code 语义的机械层，等于闸口换皮。"不能用机械判断，那不是面向智能体了"。
- **"项目级 policy" 的实体 = `features/reviews/` 文档目录**，不是 policy 文件 / CLI flag。
- **"review brief 生成（聚合材料）"不成立**：命令只呈现原理 + 切面文档，材料由模型/sub-agent 组装。
- **"五个切面 → 机器/agent/人类分层路由"修正**：没有"机器"层。五切面 = 五文档，全由 sub-agent
  执行；人类只剩 fallback 执行 + 最终裁决两个角色。
- **零上下文的实现 = "派 sub-agent"这个动作本身**（sub-agent 天然全新上下文），不是 MOSS 造零上下文 agent。

### 当前疑点（下一轮待对齐）

1. **机械矛盾的覆盖缺口**：status/status_note 不一致、文件拓扑缺失这类 100% 机械可查的违约，
   在"纯语义 review、无机械判断"下由谁抓？是接受"这类矛盾靠声明书写 discipline 消灭、不归
   review 管"，还是某个切面文档里提示 sub-agent 核对（但那又退回机械判断）？这是"确定性层"
   被否后留下的真缺口，需确认是有意为之还是需要更轻的机制（如 features check 扩职责）。

2. **锦囊的触发点**：交付引力下的模型倾向"交付优先、不停下来 review"。锦囊模型假设模型会在
   合适时机主动调用，但触发在哪？`features check` 的提醒？commit 前流程约束？人类提示？
   这决定 review 是"会被用"还是"躺在 specification 里没人开"——"流程发起"的共识还没落到具体触发形态。

3. **sub-agent 的材料与工具面**：sub-agent 是零上下文化身，它怎么拿到"代码 vs 声明"的材料？
   模型派 sub-agent 时把 diff/触及文件贴进 prompt（模型仍参与组装，可能带入开发模型偏差），
   还是 sub-agent 自带只读工具（真独立，但需定义工具面）？这决定 sub-agent 的独立性边界。

### 自反挑战的结论：agent 层 vs 编排层（cli-flow 定名）

挑战"为何不直接做 AGENT.md (agentprotocol.com) 的 agents？"——结论是**分解**：

- **agent 层 = 视角文档**（一段指令 + 输入输出契约），确实是 agent，用 AGENT.md format 完全成立，
  coding agent 的 sub-agent 原生认 AGENT.md 时甚至可白嫖 spawn。
- **编排层 = 惰性披露 + when 对齐 + 快照合成 + 委派契约**，不是 agent，AGENT.md 管不着，却是本
  feature 的增量——绑定 FEATURE.md 的"声明-交付对账"语义 + feature loop 生命周期。

编排层解决四个"裸 agent"解决不了的问题：

1. **when**：何时用哪个视角（定稿/交付前），feature loop 语义，AGENT.md 无此概念。
2. **快照绑定**：视角绑定某个 feature.md + 目录，@ 生成自包含快照。
3. **注意力模型相反**：AGENT.md 是 ambient declaration（按需发现加载），我们要 lazy disclosure（藏起、调用才展开）。
4. **零上下文契约**：必须委派 fresh sub-agent，不自我执行——AGENT.md 只定义 agent，不定义"谁在什么条件下执行它"。

**cli-flow 定名**：用 CLI hint 关联智能流程——不污染全局上下文、不依赖模型原生 prompt、不靠渐进披露
发现。feature review 是**实例**，cli-flow 是**模式**，可推广（第二个此类流程复用骨架）。

三个把"CLI 不是 agent"从消极翻成积极的补充：

- **pipe 可组合**：独立 CLI 可被 bash pipe 进别的 agent run 命令，in-process agent 不行。故 @ 命令
  stdout 必须是裸文本 synthesized instruction（非 table/JSON）。
- **缓存省轮次**：内联自包含快照作为 fresh agent 首条消息命中 prompt cache，省至少两轮 tool-call。
- **CLI 即准入门槛**：MOSS 不替人实现、不兜底工具权限，CLI 是 MOSS 责任面边界，pipe 之后是执行器域。

**终点**：agent facade / coding agents HOST——loop 可调度 agent 时，此旁路机制是候选之一；cli-flow 的
形态（纯文本进出、确定性、可 pipe）天然可被 HOST 调度。

### 视角提案（提示，非终稿，最后建文档）

核心视角文档最后建，此处只留思路：

| 视角 | when | 核心提问 |
|---|---|---|
| 接手（可重建性/摩擦点） | `定稿` | 你是下一个化身，读声明+目录，能知道怎么开始吗？卡在哪、什么含糊、缺什么？ |
| 对账（声明 vs 交付） | `交付前` | 逐条对账声明 vs 代码；声明说 X 而代码悄悄做 Y 或漏 X 的，用 file:line 指出；absence 也是信号。 |
| 方案（设计自洽） | `随时` | 读声明+已有交付，设计层面矛盾/漏洞/静默降级？只 surface 候选点，交人终审。 |

- **when 词汇表**：`定稿` / `交付前` / `随时`（枚举要小，模型才能匹配当前时间点）。
- **文档模板** = metadata(description + when) + content；无数字前缀（数字前缀服务 loop 设想，不成立）。
- 三视角覆盖四个 motivating failures（全是"对账"能抓的 silent todo）+ 遗忘测试（接手）+ L3 兜底（方案）。

### 疑点更新（"当前疑点"三个已解）

1. 机械矛盾 → 交 CLI / 未来 pre-commit，不属 review。✓
2. 触发点 → specification 里一行纪律陈述 + create-feature 模板 hint。✓
3. sub-agent 材料 → 内联（feature.md + ls depth=1 + 视角文档），sub-agent 只读代码。✓

剩余开放点（开发阶段试错，不阻塞）：

- **元模板住哪**：synthesized instruction 骨架（{} {} 替换两个内联部分），lean 是 v1 写死在 CLI，需求出现再外提。
- **feature.md 内联 + 目录展开**：moss ground 正在验证内存构建 ground 数据源、自动展开 feature.md 目录，未打磨好、不一定复用；v1 或只 feature.md + 简单 ls depth=1。
- **两个发现位置**（全局 features/review/ + per-feature review/）的 inherit/override 语义：lean 按视角名继承+覆盖（同名覆盖、缺的继承）。目录名 TBD（review vs reviews）。