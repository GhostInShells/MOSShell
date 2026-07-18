---
title: Ghost Filesystem Desktop — Ghost 的认知桌面
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-07-19
renamed_from: Project Manager
depends:
  - momento-mori
milestone: 0.1.0
description: >-
  Ground 是 Ghost 的认知场 — context_messages 上可 pin 的表面 + 场 (目录)
  的开合管理. 以 frontmatter+body+pins 的 L0 文件 (GROUND.md) 为载体;
  frontmatter 是运行时生命周期的合法发明域, body 永远开放集, pins 是第一
  人称声明. 三层落地按预训练迁移量排序: CTML channel (主), bash CLI
  (`moss ground`), module_eval (小概率). 与 subprocesses/job_supervisor
  (执行), Memento (过去) 共构反身性基建.
status_note: >-
  2026-07-19 Claude Opus 4.7 [1m]. 抽象锁定, 待人工重命名 desktop→ground
  后按 SPEC 重写 concrete 与 CLI. 契约文档: core/desktop/SPECIFICATION.md
  (将随目录重命名迁至 core/ground/). 本轮沉淀 K44~K49:
  K44 pin 传参走 argv (`pin: [verb, arg1, arg2, ...]`), 命令名多态; 拒绝
  kwargs 与强类型结构 (前者跨语言协议难定, 后者预训练迁移量差).
  K45 pin 类型最小集 = file / glob / frontmatter / ls. bash 不做, 未来同族
  加入; frontmatter 字段 / markdown 段落 (K41) 同族语法延伸而非新机制.
  K46 label + description 手动指定, 撤自动派生 (拒绝理由: 未来 pin_bash
  等类型使派生规则膨胀成一门未申明的语言). label = pin 声明第一字段, ground
  内唯一, 承担 unpin 定位与 fenced block 语言标签双职. description = 边注
  (合并原 note); 长解说走 body.
  K47 拓扑分区 = ground 头 + body + 声明区 (每行 `label:verb(args) #
  description`) + 结果区 (每 pin 一个 fenced block, label 作 lang tag).
  声明区不允许自由文本. 展示格式细节 dogfooding 中调整.
  K48 CLI 极小面 = spec / init / frame / observe 四动词. 撤 pin / unpin /
  update / status / pins / instruction. 理由: SPEC 就绪后直接编辑 GROUND.md
  的 YAML 是最快路径 (K16 去发明纪律); pin 动词的真正落点是 CTML channel
  层 (K14), CLI 不必对应实现. 每次 CLI 调用是一次性 open→render→close,
  无跨调用 opened 状态.
  K49 全库 desktop→ground 重命名 (contracts/desktop.py → contracts/ground.py,
  core/desktop/ → core/ground/, cli/desktop_cli.py → cli/ground_cli.py,
  `moss desktop *` → `moss ground *`). Grounds/Ground/GroundConvention 类名
  不变. feature 目录名 ghost-filesystem-desktop 保留 (git 历史语义). 07-12
  K16 表格里 "channel 名 desktop 隐喻仍成立" 被本轮覆盖 — desktop 是"表面"
  隐喻, ground 是"目录"实体, 混用两个词模型会崩. 全库 ground.
  同时修正/收敛:
  K30 tree_ignore 轻方案改为完整 .gitignore (`pathspec` 依赖) — pin_glob /
  pin_ls 都需要完整语义, 硬编码兜底集不够用. 新增依赖代价可接受.
  K33 更正 — channels/desktop_channel.py 从未存在于任何提交 (git log 验证).
  07-13 记录的"已写未测未装配"是模型对不存在文件的凭空断言. 真实状况:
  CTML channel 层从未落地. K14 装配等 SPEC 就绪 + K49 重命名完成后启动.
  K34 完成 — 旧 Stage 1 代码 (12+1 原语 1548 行) 已由 d75a0112 (07-14)
  删除. 从"已知未决"移出.
  未决 (给下一实例): K23 (L2 引导地址) / K24 (侧影载体位置) / K25 (向下场
  声明的形状) / K40 (K35 合成语义与 K28 幂等的冲突) / K41 (pin 类型扩展)
  / K43 (.grands/ 回退路线); 多认知方法开放; Ghost 认知场初始化等 GROUND.md
  闭环后启动.
  --
  历史轨迹 (07-11~07-15 讨论 / 06-28~06-29 Stage 1 落地): 摘要保留在
  §Key Decisions, 完整细节见 `git log --all -- .ai_partners/features/
  workstreams/2026/06/ghost-filesystem-desktop/FEATURE.md` 与
  `src/ghoshell_moss/contracts/.discuss/2026-07-{11,12}_*.md`.
---

# Ground — Ghost 的认知场

## 2026-07 收敛方向

Ground 的定位: **认知场 (context 表面 + 场开合)**, 以 `GROUND.md` (frontmatter
+ body + `## ground:pins`) 为 L0 载体. feature 名保留 `ghost-filesystem-desktop`
的历史语义, 但代码层 / CLI 层 / 文档表面全部用 ground.

- **场** = 打开的目录, 挂到父 ground channel 上作 command-less virtual channel;
  `instruction` = 法链 (向上 CLAUDE.md + 本地 GROUND.md body), `context_messages`
  = 帧渲染 (声明区 + 结果区), `startup`/`close` = load/sediment.
- **贴纸** = pin. argv 契约 (`pin: [verb, arg1, ...]`), 命令名多态; 每帧重读,
  mtime + 内容 hash 对账.
- **L0** = 场的 `frontmatter + body + pins`. frontmatter 是 MOSS 唯一发明域;
  body 与 pins 永远开放集.

**契约文档**: `src/ghoshell_moss/core/desktop/SPECIFICATION.md` (待随 K49 重命名
迁至 `contracts/ground/` 或 `core/ground/`).

**三层落地** (按预训练迁移量排序): CTML channel (主) > bash CLI (`moss ground`,
极小面) > module_eval (小概率, 不做).

**自举验收**: 下一个模型实例能通过 ground 桌面 (open 本 feature 目录 + pin
FEATURE.md 与 2026-07 两份 discuss + SPECIFICATION.md) 重建对本设计的认知,
而不靠人类手工指路.

**详细讨论轨迹** (优先读):

- `src/ghoshell_moss/contracts/.discuss/2026-07-12_desktop_channel_landing_and_fractal_grounds.md`
  — channel 落点, 分形 L 体系, features 迭代史打磨定律, 双寄存器判据
- `src/ghoshell_moss/contracts/.discuss/2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`
  — 拆捆绑, 认知场概念, 收敛形状

## Motivation

Ghost 需要在文件系统上有 "自己的工作面":

- 模型 context 没有可控的注意力结构 — 系统推什么就看什么, compact 之后关键
  信息丢失, 无法自主构建工作记忆
- 行业方案 (CLAUDE.md, MEMORY.md, rules, skills) 是 **静态** 的 — 没人定义
  "进入一个目录在运行时意味着什么", 也没人做 **场所局部** (place-local) 的
  认知加 enter/exit 生命周期
- MOSS 自己的 features / .discuss / .design 三级验证体系有相同的结构但没有
  运行时载体 — 每次会话都要靠模型记得去 `features list` 才存在

Ground 把这些统一为 **运行时的可 pin 的 context 表面 + 场的开合**. frontmatter
是唯一的机器边界; body 永远开放; 什么文件 "有认知价值" 由场里的 GROUND.md
文件自己声明. 与静态方案的本质差异: **场是活的** — 每帧重绘, 变更标记, pin
是第一人称动作而非系统自动规则.

## Design Index

- **契约文档 (SPEC)**:
  - `src/ghoshell_moss/core/desktop/SPECIFICATION.md` — GROUND.md 格式规范,
    pin 类型契约, frame 拓扑, CLI 极小面, 语言无关性要求
- **2026-07 重绘讨论**:
  - `.discuss/2026-07-12_desktop_channel_landing_and_fractal_grounds.md`
  - `.discuss/2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`
- **配对基建**:
  - `momento-mori` (Memento) — 胶囊 (promote 后的 pin) 落永久记忆; drain 联合
    设计方
  - `subprocesses` / `job_supervisor` — 从旧 Desktop 拆出的执行域 (审计线外侧).
    契约层形态是 K18 三层纪律的样板参照
  - `file-editor-contract` — 写路径 + 结构化 view; K6 撤守卫留下的空档承接方
- **channel 抽象参考**:
  - `moss codex blueprint channel_builder` — Channel as Context Component,
    instruction / context_messages / refresh_meta 生命周期
  - `moss codex blueprint states_channel` — `MutableChannelState.add_virtual_channel`
    docstring: "wrap this method into a command" — Ground 的原生解
  - `src/ghoshell_moss/channels/module_eval_channel.py` — abc → concrete →
    channel 三层装配的标杆参照
- **CTML 视角**: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- **历史设计** (供反向查询, 部分被 07 月重绘超越):
  - `.design/2026-06-28_desktop_in_4d_cross_section.md` — 4 剪影拓扑 + 12+1 原语
    L2 稿 (Stage 1 实现的依据; 隐喻仍成立, 12+1 捆绑不成立)
  - `.discuss/2026-06-28_desktop_l2_emergence.md` — L2 涌现方法论

## Key Decisions

保留 07-12 起的核心方向 (K14~K21) 与 07-14/15 对齐 (K35~K43) 的摘要, 加本轮
07-19 (K44~K49). 更早的 K1~K13 (原捆绑设计) 大部分已随 Stage 1 代码删除
(d75a0112), 需要考古走 `git log --all -- FEATURE.md`. K23/K24/K25 收在
"已知未决".

### 07-12 核心方向 (K14~K21)

- **K14. 落 channel** — 父 ground channel + 每场 command-less virtual channel;
  context_messages 是"桌子" (帧尺度 O(1) 覆写表面, compact 免疫).
- **K15. 分形 L 体系** — L-1 (领域数据 CLAUDE.md/MOSS.md) / L0 (frontmatter +
  body + pins) / L1 (L0 模板) / L2 (L1 库). frontmatter/body 边界即机器/模型
  边界. features 体系是实物证明.
- **K16. 双寄存器词汇** — 理论词汇住文档, 表面词汇骑预训练先验.
  **本轮 (K48) 修正**: "channel 名 desktop 隐喻仍成立" 被覆盖 — 全库统一 ground.
- **K17. 桌子即工作记忆** — 不与世界自动同步. 变更 hash 对账标记 stale,
  显式 update 推进认知. 原生 drain 协议独立立项 (K19).
- **K18. 三层抽象纪律** — contracts (ABC) → core (concrete) → channels
  (手写装配). 不做过早规则糖.
- **K19. 原生 drain 独立立项** — 与 Memento "产生时入记忆" 合并设计,
  非本 feature 关切.
- **K20. 目光落运行时侧影, 不自动 sediment 到 body** — auto-sediment 到治理
  文件违反 "沉淀是主动动作". 侧影载体位置见 K24 (未决).
- **K21. open/update 边界** — `Grounds.open(dir, label=None) → Ground`.
  CTML 表面在父, core 层薄转发到 Ground; 同 dir 幂等 (K28).

详细讨论见 `.discuss/2026-07-12_*` 与 `git log --all -- FEATURE.md` 中
2026-07-12 前后 commit.

### 07-14 认知场对齐 (K35~K39)

- **K35. 携带/属地合成** — 场 (pins + body) 天然属地; convention 与 body
  可携带 (L1). 分层合成: convention per-field merge, 法链式合成, pins 永
  不携带. 默认属地胜; 强制覆盖须 open 显式参数. 与 K28 幂等的冲突待
  dogfooding (K40).
- **K37. L1/L2 纪律** — 冷启动 fewshot + 可验证. 修改记录即"未被囚禁"证据.
- **K38. L2 = 追认 .ai_partners/** — 不新建 templates/ 目录. moss 自建 L2
  dogfood.
- **K39. 五问信息传递 + 三级发现链** — where/what/which/how/why 劈机器域
  /文体域. L2→L1→L0 发现链走 glob + marker.

详细行业对照与 K36 前身讨论见 `git log --all -- FEATURE.md` 中 2026-07-14
commit.

### 07-15 marker + $id + pin 类型扩展 (K22 / K36 / K41~K43)

- **K22 (07-15 重决)** — L0 marker: DESKTOP.md → GROUND.md. GROUND = 电学
  零电位参考点 + common ground 双料先验, AI 行业未占. 本轮 K48 把 rename
  从 marker 层升级为全库层.
- **K36. pin 落盘瘦身** — 只留 argv `pin` + `label` + optional `description`.
  seen_* 观察态是运行时侧影, 不入 GROUND.md. 首次进入无上帧故不标 stale.
  本轮 SPEC 落定.
- **K41. pin 类型扩展 (path#field, path## heading)** — 语法候选, 本轮 K44
  argv 契约 (`pin: [type, arg1, ...]`) 天然容纳同族扩展.
- **K42. $id 身份体系** — frontmatter `$id: <URI>` 字符串, MOSS 不校验, 解析
  交上层. 撤 `$template` (血统记录与身份声明混淆).
- **K43. .grands/ 设计分支** — 回退方案. 反悔判据: 多认知方法成为真实痛感.

详细讨论见 `git log --all -- FEATURE.md` 中 2026-07-15 commit.

### 07-19 本轮决策 (K44~K49)

本轮为 "抽象锁死, 待人工重命名后按 SPEC 落代码" 的收敛点. 契约文档独立到
`src/ghoshell_moss/core/desktop/SPECIFICATION.md`.

- **K44. pin 传参 = list[str] 位置参数, 命令名多态** — 三方案对照:
  - 方案 1 (kwargs `pin_file(path=..., range=...)`) — 参数名要跨语言协议化,
    模板压力大 (可选参数空值处理麻烦).
  - 方案 2 (强类型结构 `PinFile(path=..., range=...)`) — 跨语言协议难定,
    模型预训练迁移量差.
  - 方案 3 (`pin: [verb, arg1, arg2, ...]`) — 采用. 理由: bash argv 是
    预训练最深处 (K16 双寄存器); SPEC 表达最简 (每 type 一行 argv 契约);
    落盘 shape 最简; 未知类型前向兼容天然.

- **K45. pin 类型最小集** — `file / glob / frontmatter / ls` 四个,
  pin_bash 本轮不做 (07-11 T1 不变量决定). K41 语法扩展 (path#field /
  path## heading) 在本 argv 契约下是同族扩展, 不需新机制.
  展开约束 (SPEC 强判): glob / ls 只出结构 (paths + mtime + size), **不出
  内容** — 一个 pin 展成 50k tokens 是 dogfood 死亡场景.

- **K46. label + description 手动指定** — 撤回自动派生 (讨论中曾提议 label
  从 argv 派生). 理由: pin_bash 加入时派生规则会崩 ("`bash "git log --oneline"`
  应派生什么 label"), 规则本身成为一门未申明的语言.
  - label = pin 声明第一字段, ground 内唯一, 承担 unpin 定位 + fenced block
    lang tag 双职. 字符集 `[a-zA-Z_][a-zA-Z0-9_-]{0,63}`.
  - description = 短评注 (marginalia). 合并原 "note" 概念. 长解说走 body.

- **K47. 拓扑分区 = 声明区 + 结果区** — frame 分节:
  - head: `ground: <label> @ <abspath>` + 可选 `$id`
  - body: GROUND.md body verbatim (场解说落这里)
  - 声明区: 每 pin 一行 `label:verb(args) # description`, 一屏可扫
  - 结果区: 每 pin 一个 fenced code block, label 作 lang tag

  声明区**不允许自由文本** (场解说走 body). 展示格式细节 dogfooding 中调整.
  这是 07-11 记录者 "桌子 + 贴纸" 直觉的 markdown 化.

- **K48. CLI 极小面 = spec / init / frame / observe 四命令** — 撤原
  pin / unpin / update / status / pins / instruction 六动词. 理由:
  - SPEC 就绪后直接编辑 GROUND.md 的 YAML 是最快路径 (K16 去发明纪律);
  - pin 动词的真正落点是 CTML channel 层 (K14), CLI 不必对应实现.

  CLI 只提供诊断与自解释:
  - `moss ground spec` — 打印 SPECIFICATION.md
  - `moss ground init [dir]` — 造空 GROUND.md 脚手架
  - `moss ground frame [dir]` — 渲染完整帧 (dogfood 主力)
  - `moss ground observe [dir]` — pin 观察诊断

  每次 CLI 调用一次性 `open→render→close`, 无跨调用 opened 状态 —
  session 状态归 CTML channel 层.

- **K49. 全库 desktop → ground 重命名** — K16 原判 "channel 名 desktop
  隐喻仍成立" 本轮覆盖. 混用 desktop (表面隐喻) + ground (目录实体) 两个词
  模型会崩, 全库统一:
  - `contracts/desktop.py` → `contracts/ground.py`
  - `core/desktop/` → `core/ground/` (SPECIFICATION.md 随目录搬)
  - `cli/desktop_cli.py` → `cli/ground_cli.py`
  - `moss desktop *` → `moss ground *`

  Grounds / Ground / GroundConvention 类名不变 (已是 ground 命名).
  feature 目录名 `ghost-filesystem-desktop` 保留 (git 历史语义, 不动发现链).
  重命名由人工 IDE 完成, 完成后按 SPEC 重写 concrete + CLI + 单测.

- **K30 修正 (轻方案 → pathspec)** — 原 K30 走 BUILTIN_TREE_IGNORE 常量 +
  tree_ignore_extra 加法口. 本轮 pin_glob / pin_ls 都需要 `.gitignore` 完整
  语义, 硬编码兜底集不够. 走 pathspec 依赖.

- **K33 更正 (记录失真)** — `channels/desktop_channel.py` 从未出现在任何
  提交 (`git log --all --` 空返回验证). 07-13 status_note "已写未测未装配"
  是模型对不存在文件的凭空断言. 真实状况: **CTML channel 层从未落地**.
  K14 装配等 SPEC 就绪 + K49 重命名后启动.

- **K34 完成 (旧 Stage 1 清理)** — 已由 commit d75a0112 (2026-07-14 03:32)
  删除 Stage 1 三份共 1548 行代码 (`core/desktop/desktop.py` 950 +
  `core/desktop/models.py` 65 + `tests/.../test_desktop.py` 529). 从
  "已知未决" 移出.

## 已知未决 (给下一个实例)

- **K23 (L2 模板库引导地址)** — moss 侧已定 .ai_partners/ (K38). 残余问题:
  ghost 携带 L2 与项目属地 L2 的关系 (K35 合成在 L2 层级的应用).
- **K24 (目光运行时侧影载体)** — .cache 级 gitignore 目录的具体约定. K36
  已定 seen_* 不入 GROUND.md; 侧影落盘位置未定 (SPEC 目前无强制要求, 只
  规定不能在 GROUND.md 中).
- **K25 (向下探索的场声明)** — 发现链形状已有 (K39: marker + glob), L1
  marker 文件名待收 (可能是 GROUND.md 加 `$id` 承担, 也可能独立 marker).
- **K40 (K35 合成语义与 K28 幂等的冲突; ghost 默认场)** — dogfooding 讨债.
- **K41 (pin 类型扩展 path#field, path## heading)** — argv 契约留位, 具体
  实现按 dogfooding 需求推进.
- **K43 (.grands/ 分支)** — 回退方案. 反悔判据: 多认知方法成为真实痛感.
- **多认知方法** — 未证明需求, 靠 dogfooding 讨债, 不预建机制.
- **Ghost 认知场初始化 (users/memory/skills/tasks/tmp 子场结构)** — 等
  GROUND.md 闭环 + L1 实例化动词就位后启动.

## 与关联基建的交叉

| 基建 | 关系 | 状态 |
|------|------|------|
| `subprocesses` / `job_supervisor` | 执行域, 从旧 desktop 拆出 (K11/K13 迁出) | 已迁出, contracts 已重绘 |
| `file-editor-contract` | 写路径 + 结构化 view; K6 撤守卫留下的空档承接 | draft, 2026-07-13 立项 |
| `momento-mori` (Memento) | 胶囊 (promote 后的 pin) 落永久记忆; drain 联合设计方 | FORMAT.md 契约层落盘 |
| `Matrix` | ground 不直接依赖; virtual channel 生命周期由 Channel Runtime 管 | 无直接关系 |
| `features` 体系 | K15 分形体系的实物证明; L2 dogfood 第一站 | 已运行, 自 2026-05 |
| `Ghost` / `Mode` | ground 进入哪些 mode 是 Ghost 层决策, OS 层不主动推 | 未开始 |
| 原生 drain 协议 (K19) | 独立 feature, 与 Memento 合并设计 | 未立项 |

## 下一步 (2026-07-19 视角)

**先手动**: 用户 IDE 重命名 desktop → ground 全库 (代码 + CLI + 文件路径).

**然后按 SPEC 重写**:

1. `contracts/ground.py` — Pin 字段调整 (label/pin/description; 撤 addr/note/pinned_at);
   GroundConvention 加 `$id: str | None` (pydantic alias `$id`), 撤 `template`;
   Ground ABC 保持完整动词 (K21 红利).
2. `core/ground/_l0.py` — 常量 `GROUND.md` + `## ground:pins`; pin 段 YAML
   shape 按 SPEC (`pin: [verb, args]`); Pin 反序列化按 argv.
3. `core/ground/_ground.py` — pin 内部走 argv dispatcher; load() 结束后对所有
   pin 并行 observe 一次 (populate 运行时侧影); pin/update 不写 seen_* 到盘.
4. `core/ground/_render.py` — 按 K47 拓扑重写 (head + body + 声明区 + 结果区);
   fenced block 用 label 作 lang tag; 声明区行格式 `label:verb(args) # description`.
5. `core/ground/_pin_types.py` (新) — 四个 pin verb 实现 (file/glob/frontmatter/ls);
   pathspec 依赖引入; 失败模式按 SPEC.
6. `cli/ground_cli.py` — 收敛到四命令 (spec/init/frame/observe); 撤原六命令.
7. 单测按新形状全部重写 (test_l0.py / test_ground.py / test_grounds.py /
   test_render.py + 新 test_pin_types.py). 87 单测预计缩到 ~40 (原 pin/update
   的观察态测试大量作废).

**验收**:
- 全量单测跑绿
- 本 feature 目录 dogfood: `moss ground init && 编辑 GROUND.md pin FEATURE.md
  + 两份 discuss + SPECIFICATION.md && moss ground frame` — 能重建到本轮
  决策为止的认知
- 提交按规范: 英文标题 + `coding by claude-opus-4-7` 后缀 + `via claude code`
  正文
