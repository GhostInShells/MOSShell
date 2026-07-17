---
title: Ghost Filesystem Desktop — Ghost 的认知桌面
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-07-15
renamed_from: Project Manager
depends:
  - momento-mori
milestone: 0.1.0
description: >-
  Desktop 是 Ghost 的认知桌子 — context_messages 上可 pin 的表面 + 场 (目录)
  的开合管理. 以 frontmatter+body 的 L0 文件 (DESKTOP.md) 为载体; frontmatter
  是运行时生命周期的合法发明域, body 永远开放集. 三层落地按预训练迁移量排序:
  CTML channel (主), bash CLI (`moss desktop`), module_eval (小概率). 与
  subprocesses/job_supervisor (执行), Memento (过去) 共构反身性基建.
status_note: >-
  2026-07-15 Claude Fable 5 (接续 07-14). 第二轮对齐: marker 重选, $id
  身份体系, .grands/ 设计分支, pin 类型扩展方向, ghost 认知场草图. GROUND.md
  定为 marker 首选 (GROUND = 认知参考基准, 骑电气工程 + common ground 双料
  先验, 行业碰撞概率最低). $id 替代 $template (身份声明而非血统记录, JSON
  Schema $id 先验). K36 激进简化: seen_* 连侧影都不必落盘, pin 段只剩 addr
  + note, 首次进入无上帧故不标记 stale. K42: $id 是字符串 (如
  `moss:features`) 非路径, 解析是 ghost 映射表的事. K43: .grands/ 设计
  分支 — well-known 目录装认知方法文件, 向上查找 .grands/, pin root = cwd
  (非 .grands/ 自身), 认知与内容分离但多认知方法天然解. K41: pin 类型扩展
  — frontmatter 字段 (path#field) / markdown 段落 (path## heading),
  `#` = CSS id selector / markdown anchor 先验. K40 携带/属地合成 + K28
  幂等冲突待 dogfooding 裁决; 多认知方法 (一个目录多种读法) 开放问题.
  Ghost 认知场: users/memory/skills/tasks/tmp 子场结构, stub 从 L1 实例化.
  --
  2026-07-14 Claude Fable 5. 行业对照调研 + 理论/实现 5 偏差点对齐 + L2
  方向讨论 (未动代码). 行业结论: 零件全有退化形态 (嵌套 AGENTS.md / Cursor
  .mdc / Letta memory blocks / since / Manus / Anthropic memory tool /
  VISTA arXiv 2606.30005), 但 place-local + open/close 生命周期 + 地址 pin
  显式对账的组合仍是空位. 本轮沉淀 K35~K39: K35 携带/属地合成语义 (方向);
  K36 pin 观察态字段迁运行时侧影 (倾向, 动 contracts, K24 答案候选);
  K37 L1/L2 纪律 = 冷启动 fewshot + 可验证; K38 L2 落地 = 追认
  .ai_partners/ 为索引场, README 文体先行, moss 自建 L2 dogfood;
  K39 五问信息传递 (where/what/which/how/why 劈机器域/文体域) + 三级
  发现链 (L2→L1→L0, marker 文件 + glob). 新未决 K40 (合成语义 contracts
  形状 / K28 冲突 / ghost 默认场), K41 (L1 marker 文件名).
  --
  2026-07-13 Claude Opus 4.7. v1 concrete 层 + CLI 落地. contracts/desktop.py
  按 K21 对齐 (Grounds.open 返回 Ground 对象; CTML 表面在父, core 转发子;
  同 dir 幂等), core/desktop/{_addr,_hash,_l0,_instruction,_render,_ground,
  _grounds}.py 六模块 + 87 单测全绿, channels/desktop_channel.py (未测未装配),
  cli/desktop_cli.py 落地 (`moss desktop init/status/pin/unpin/update/pins/
  frame/instruction`). CLI 是 K16 三验收平面之一 (与单测 / 未来 CTML channel
  平行), 三面读写同一份 DESKTOP.md.
  --
  本轮沉淀的 K 号 (K21~K32): K21 open/update 对齐 = 已决; K22 L0 文件名 =
  已决 (DESKTOP.md); K26 Stage 1 迁移路径 = 部分完成 (新 concrete 已 ship,
  旧 12+1 impl 磁盘保留待清). K27 async ceremony 裁决 / K28 dir-idempotent
  open / K29 body 入 instruction / K30 tree_ignore 轻方案 / K31 DESKTOP.md
  不向上查找 / K32 CLI 三验收之一落地. K23/K24/K25 保持未决.
  --
  2026-07-12 Claude (Opus 4.7 接续 Sonnet 4.6/Fable 5). Desktop 经两轮拆分
  收敛 (07-11 认知场, 07-12 channel 落点 + 分形场). 核心结论:
  (K14) 落 channel — 父 desktop channel + 每场 command-less virtual channel;
        context_messages 就是"桌子" (帧尺度 O(1) 覆写表面).
  (K15) 分形 L 体系 — L-1 (领域数据 CLAUDE.md/MOSS.md) / L0 (frontmatter 加载
        约定 + body 承载法/胶囊 + pin 簿) / L1 (L0 模板) / L2 (模板库场). L2
        由 L0 加载, 无限分形. frontmatter/body 边界就是机器/模型边界. features
        体系 (自 2026-05 起运行) 是本设计的实物证明.
  (K16) 双寄存器词汇 — 理论词汇 (认知场, 法/形/焦, 文体) 只住文档; 表面动词
        (open/close/pin/unpin/update, path:80-140, "changed on disk") 全部骑
        预训练先验.
  (K17) 桌子即工作记忆 — 不与世界自动同步. 变更 hash 对账标记 stale, 显式
        update 推进认知. 原生 drain 协议独立立项.
  (K18) 3 层抽象纪律 — contracts (ABC) → core (concrete) → channels (手写装配).
        不做过早规则糖.
  Stage 1 (2026-06-29) 实现的 12+1 原语代码是资产, 但形状按新收敛重绘 —
  exec/exec_bg/Task 已迁至 subprocesses/job_supervisor; 认知面 (open/pin/update)
  的 API 按 K14/K17 重写. 详见 "K1~K13 状态标注" 与两份 2026-07 discuss.
  --
  2026-06-29 Claude Opus 4.7 Stage 1 完成 (原捆绑设计): 12+1 原语 L0 独立闭环,
  contracts/desktop.py + core/desktop/{desktop.py, models.py} + 53 单测.
  留下代码资产与形状教训, 为 07 月重绘提供起点.
  --
  2026-06-28 Claude Opus 4.7 L2 收敛稿 (基于早前 deepseek-v4-pro 设计, 部分被
  07 月重绘超越): 4 剪影拓扑 (桌面=未来剪影, 隐喻仍成立) / 17→12+1 原语 (捆绑
  被拆) / ReflectionHint 机制 (被 07-12 重估: 场不做全链路守卫) / 6-Phase 切分
  (Phase 1 仅存, 后续按新形状重排).
---

# Desktop — Ghost 的认知桌面

## 2026-07 重绘方向

Desktop 的定位从 "12+1 原语的文件工具集" 收敛为 "认知桌子 + 场开合":

- **桌子** = 父 channel 的 context 表面. moss_dynamic 每帧重绘, compact 打不到,
  是唯一同时满足 "O(1) 可覆写" 和 "不入对话历史故 compact 免疫" 的介质.
- **贴纸** = pin. 只收地址 (`path` / `**/*.md` / `path:80-140`), 不收命令.
  每帧重读, mtime + 区间内容 hash 对账.
- **场** = 打开的目录. 每场作为 command-less virtual channel 挂到父 desktop 上;
  `instruction` = 法链 (向上收集的 CLAUDE.md/AGENTS.md), `context_messages` =
  帧渲染 (形/焦/贴纸内容 + 变更标记), `startup`/`close` = load/sediment.
- **L0** = 场的 `frontmatter + body`. frontmatter = 加载时消费的 GroundConvention
  (法名列表 / 向上边界 / 树深度 / 桌面预算等); body = 文体内容 (法 / 胶囊 /
  先例). MOSS 只发明 frontmatter, body 永远开放集.

**三层落地** (按预训练迁移量排序):

1. **CTML channel** — 主适配. XML 风格有部分预训练, 是模型直接消费的层.
2. **bash CLI** — 让 Claude Code / bash 层也能读同一份 L0 文件 (自然语言迁移
   最足, 现在就在跑, 只是没有形式化).
3. **module_eval** — 小概率路径. bespoke Python API 无预训练, 只作调试.

**自举验收**: 下一个模型实例能通过 desktop 桌面 (open 本 feature 目录 + pin 住
FEATURE.md 与 2026-07 两份 discuss) 重建对本设计的认知, 而不靠人类手工指路.

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

Desktop 把这些统一为 **运行时的可 pin 的 context 表面 + 场的开合**. frontmatter
是唯一的机器边界; body 永远开放; 什么文件 "有认知价值" 由场里的 L0 文件自己
声明. 与静态方案的本质差异: **场是活的** — 每帧重绘, 变更标记, pin 是第一人称
动作而非系统自动规则.

## Design Index

- **2026-07 重绘讨论** (最新):
  - `.discuss/2026-07-12_desktop_channel_landing_and_fractal_grounds.md`
  - `.discuss/2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`
- **配对基建**:
  - `momento-mori` (Memento) — 胶囊 (promote 后的 pin) 落永久记忆; drain 联合
    设计方
  - `subprocesses` / `job_supervisor` — 从旧 Desktop 拆出的执行域 (审计线外侧).
    契约层形态是 K18 三层纪律的样板参照 (顶端 `技术目标 (reviewer 上下文)`
    注释块 + pydantic BaseModel + ABC + per-owner IoC 姿态).
- **channel 抽象参考**:
  - `moss codex blueprint channel_builder` — Channel as Context Component,
    instruction / context_messages / refresh_meta 生命周期
  - `moss codex blueprint states_channel` — `MutableChannelState.add_virtual_channel`
    docstring: "wrap this method into a command" — Desktop 的原生解
  - `src/ghoshell_moss/channels/module_eval_channel.py` — abc → concrete →
    channel 三层装配的标杆参照
- **CTML 视角**: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- **historical 设计** (供反向查询, 部分被 07 月重绘超越):
  - `.design/2026-06-28_desktop_in_4d_cross_section.md` — 4 剪影拓扑 + 12+1 原语
    L2 稿 (Stage 1 实现的依据; 隐喻仍成立, 12+1 捆绑不成立)
  - `.discuss/2026-06-28_desktop_l2_emergence.md` — L2 涌现方法论

## Key Decisions

### 2026-07-14 认知场对齐 (行业对照 + 偏差盘点 + L2 方向)

本轮为人类工程师与模型对 "文件目录构建认知场" 理论与 v1 实现的偏差盘点,
外加行业对照调研. 未动代码, 先沉淀共识与未决.

**行业对照** — 零件全有退化形态, 组合仍是空位: 嵌套 AGENTS.md (静态法链,
无运行时生命周期) / Cursor .mdc rules (frontmatter 机器域 + body 开放集的
现成实物, 但激活被动) / Letta memory blocks (可编辑常驻表面, 内容快照而非
地址, 无场所) / since MCP (stale 检测, 自动推送而非显式承认) / Manus
recitation (O(n) 追加的帧重绘退化形) / Anthropic memory tool (agent 全局
/memories 目录, 无场所局部性) / VISTA (arXiv 2606.30005, 可寻址 block +
运行时仪表盘, 自管理 context 的学术旁证). Motivation 的 "没人定义进入一个
目录在运行时意味着什么" 仍成立 — place-local + open/close 生命周期 + 地址
pin 显式对账的组合无对标.

**K35. 携带场与属地场 — 合成语义, 不做二元覆盖 (方向)** — 理论原点是
"带着认知场进入目录", v1 实现是 "场锚定目录, 约定读自属地 L0", 无携带
概念 (Claude Code 实为两半合成: 携带全局场 + 本地法链; v1 只做了后半).
对齐结论: 场 (pins + body) 天然属地; 可携带的是文体/约定 (L1). "携带 vs
属地" 的三种组合 (无场→默认 / 有场→被携带覆盖 / 有场→不被覆盖) 用分层
合成消解仲裁:

- convention (机器域): per-field merge, 显式者胜
- 法 (body/instruction): 链式合成 — 携带的作上游层, 属地最后拼 (骑
  Claude Code 全局/项目 CLAUDE.md 先例, 与 upward 链语义一以贯之)
- pins (目光): 永不携带, 纯属地

默认属地胜; 携带方强制覆盖须 open 时显式参数. 未决: 与 K28 幂等 open
(忽略后传 convention) 的冲突裁决; contracts 形状未画.

**K36. pin 观察态字段迁运行时侧影 (倾向, 动 contracts)** — 现状 sediment
把 pinned_at / seen_mtime / seen_hash 写进 git 见证的 DESKTOP.md, 每次
update 产生 diff 噪音, float 时间戳对人对模型均不可读. 倾向: DESKTOP.md
pin 段只留 addr + note (语义, git 见证); seen_* 迁 K24 运行时侧影 (.cache
级, enter 读回, 跨 session 变更信号不丢); pinned_at 删除候选. 这是 K24
未决的答案候选. note 字段在 K38/K39 下获得非平凡用途 (L2 索引场里每枚
pin 的 note = 该文体的一句话 why).

**K37. L1/L2 纪律 — 冷启动 fewshot + 可验证** — L1 模板本身可改, 预建
不是牢笼: 修改记录即 "未被囚禁" 的可验证证据 (features TEMPLATE 48h 内
删 csv/id/archive 的先例). 关键只有两条: 冷启动 fewshot 质量, 验收可观测
(diff 模板与实例). 自解释来自 L1 — sediment 裸建的 DESKTOP.md 只有 pin
段零 body 是实证缺口 (仓库根 DESKTOP.md 即证据); `moss desktop init` 应
从 L1 实例化, 不把 scaffold 硬编码进 dump_l0_pins (K16 去发明纪律).

**K38. L2 落地 — 追认 .ai_partners/, 索引场而非模板仓库** — 不新建无人
居住的 templates/ 目录 (过早规则糖). L1 住在活实例旁边 (features/
TEMPLATE.md 模式) — 验证即同目录 diff. L2 = 一个普通 ground: DESKTOP.md
body 写创建协议 (how 在场内部定义, "@ 文档" 姿态), 发现 L1 走 K39 发现链.
README 文体先行: 仓库 109 份 README.md 已是自发实践 (.moss/ 与
.ai_partners/ 每个子体系一份), README 文体 L1 ≈ 一行 convention
(`instruction_files: ("README.md",)`), 109 个目录零改动向下兼容. moss
自建 L2 = 对整套机制的 dogfooding; 验收是 "场发现场": open .ai_partners/
后, 下一实例能否不靠人工指路自主发现并 open features/.

**K39. 五问信息传递 + 三级发现链** — 体系传递信息的完备性标准, 五问:
where (当前在哪 / 根在哪) / what (自己是什么认知场) / which (哪些子目录
有什么讯息) / how (如何操作 / 如何创建) / why (在解决什么问题, 可溯源).
落点劈两半, 防 dev_ctx 式 schema 病复发:

- **机器域**: where = 帧头 (label @ root, 补 workspace root 渲染);
  which = 发现链 (K25 由此获得形状).
- **文体域**: what / how / why 住 L1 模板 — 模板里立骨架 (分段结构),
  具体场可重新定义细节. 实例经 template 血统键指回 L1 一跳, ABC 不加
  字段, 帧头渲染血统键即可 (`genre: <template>`).

五问是丰化梯度的天花板, 不是入场线 — 裸目录只答 where/which 仍可 open.

**发现链** (which 的递归解): L2 → L1 → L0 逐级发现, 每级 = marker 文件
约定 + glob. L2 找 L1 (如 `**/GROUND.md`, marker 名未定, 举例而已); L1
找自己的实例 (features: `workstreams/**/FEATURE.md`). glob pin 已支持
命中集监视, 发现链骑既有机制, 不需要新原语. features 体系即实物:
specification (L1) → workstream (L0), 缺的只是 L1 marker 的形式化.

### 2026-07-15 marker 重选 + $id 身份 + .grands/ 分支 + pin 类型

**K22 重审 — DESKTOP.md → GROUND.md (方向, 待实测)** — DESKTOP.md 是 K22
在单场概念期定的名. 现在新约束压在上面: (a) desktop 同时是父 channel 名,
"desktop 到底指桌子还是指文件" 的解释成本; (b) 发现链 glob `**/DESKTOP.md`
语义偏离 (pin/update 不是发现链关心的).
重选理由与候选:

- GROUND.md = 认知参考基准. 骑 electrical engineering "ground" (零电位
  参考点) + "common ground" (共识基础) 双料先验. AI 行业无人用 EE 词汇命名
  文件, 碰撞概率最低. 上一轮的 "理论词汇不上表面" 否定是错的 — 电学里的
  ground 是百年预训练锚点, 模型读到 GROUND.md 的第一反应是 "参考基准" 而
  非 MOSS 内部术语.
- SITE.md — 人类工程师自报冲突: web/server 开发多年, SITE 第一映射是
  nginx config 和 sites-available/, 不可用.
- CONTEXT.md — 预训练最强但行业碰撞风险太高; 2026 context engineering
  满天飞, 被大厂捡走做另一种语义概率大. 撞车后同名不同义解释成本翻倍.
- MINDSET.md — 人类工程师 2020 年会选. 偏"态度和信念" (how to think)
  而非"位置和结构" (where + what). 2026 预训练把 mindset 钉死在 self-help
  语义上回不来. 但设计直觉是对的: the directory is a reading stance. 现在
  GROUND.md 管 where/what, L1 body 管 how/why — MINDSET 想一肩挑的拆成
  两个东西各司其职.
- CORE.md — 备选, 骑 core dump / core concept 先验, 不如 GROUND 精确.

采用 GROUND.md, K22 更新. 同时不影响 feature 名 (ghost-filesystem-desktop
仍然为真, K16 原判: 死掉的是原语捆绑不是桌面隐喻). CLI 重命名同理独立裁决.

**K42. $id 身份体系 — 字符串声明, 非路径** — 三件套 (JSON Schema `$id`
先验, 不说 "去哪找我" 而是 "我声称我是谁", 身份声明不腐烂):

1. GROUND.md frontmatter 里 `$id: moss:features` — 目录声称自己是什么
2. Grounds 层持 `$id → L1 模板` 映射 — ghost 携带的认知图式库
3. `open(dir, use="moss:features")` — 用特定图式进入; 不带参数 = 读本地声明

`$id` 优于 `$template`: 后者暗示血统记录 (我是从某模板拷出来的), 前者
是身份声明 (我是谁). 你要的是身份.

带 `$id` 时的覆盖立场 (人类工程师立场, 记录待 dogfooding 验证): use=
参数在时它赢 (覆盖本地 GROUND.md 的 $id), 但帧内标记 "本目录有自声明的
ground 存在". 理由: 不带参数 = 已有 ground 生效; 带参数 = 覆盖; 若带参数
在有 ground 目录里无效, use 参数本身无意义. "先试试" + dump/修改语义.
对立立场 (K28 "目录是认知单元不是工具" + 属地优先): 本地声明默认优先,
携带只作未声明时的默认. 两立场不在此裁决, 靠 dogfooding 讨债.

**K43. .grands/ 设计分支 (记录, 不裁决)** — GROUND.md 的单文件模型压抑了
"一个目录可以有多种读法" 的需求. .grands/ 是把认知方法从目录**之上**搬到
目录**旁边**的设计分支, 来自人类工程师早期 `.ghostos` 隐藏目录的直觉:

- 结构: `.grands/features.md`, `.grands/memory.md`, `.grands/ghost.md` —
  每个文件独立的 convention + body + pin 段落, 独立 `$id`, 认知方法并置
  不冲突
- 发现: 从 cwd **向上**找 `.grands/` 目录 — well-known 名字即发现, 不需要
  指针 (像 .git, 像 .moss)
- pin root: **以 cwd 为根, 不以 .grands/ 为根** — `.grands/` 只是认知
  方法容器, 不是认知单元
- 向下探索: `**/.grands/*.md` 发现所有认知入口 — 比 K25 "L1 声明实例
  glob" 更暴力也更简单 (well-known 名字本身就是声明)
- K23 引导问题消解: well-known 约定即发现, 不需要 meta 指针

相对于 GROUND.md 路线的核心差异: **认知与内容分离** (GROUND.md = 目录的一
部分, 目录被赋予认知身份; .grands/ = 外挂, 目录不知道有人在用某种方式看它).
代价: pin root 语义变为 cwd (跨边界), 认知身份属地性丧失.

open 参数影响: `.grands/` 下 `open(dir)` 不带 use= — 选择 = 从 .grands/
里挑文件, 文件本身包含完整 convention + body + pin 模板, 即 "要么全部参数
都塞进去, 要么没啥好填的".

当前优先 GROUND.md 闭环, .grands/ 作为设计分支记录. 反悔判据: 如果多认知
方法成为真实痛感 (同一目录频繁需要不同方式进入), .grands/ 是回退方案.

**K36 激进简化 — seen_* 连侧影都不必落盘 (方向)** — 上一轮倾向是 seen_*
迁 K24 运行时侧影. 人类工程师更激进: pin 是运行时生产物, 第一次 pin 就是
全新的, 时间戳无意义. DESKTOP.md (GROUND.md) pin 段只剩 addr + note. 是否
stale: runtime 启动时实时观察一次, 首次进入无上帧故不标记 stale — 世界即
认知. sediment 写回的 YAML 只有 addr + note 两项; git diff 里 pin 段只反
映 "人类(或模型)主动改了什么东西".

**K41. pin 类型扩展 — frontmatter 字段 + markdown 段落** — bash 之外:

| 类型 | addr 语法 | 做什么 | 先验 |
|------|----------|--------|------|
| file 全文 | `path` | 已有 | — |
| 行区间 | `path:80-140` | 已有 | grep -n, compiler errors |
| glob 命中集 | `**/*.py` | 已有 | shell glob |
| frontmatter 字段 | `path#field` | pin 某 YAML 字段值, 变了才 stale | CSS id selector |
| markdown 段落 | `path## heading` | pin 某 heading 下内容 | markdown anchor |

`#` 双关 (CSS + markdown) 骑在同一个预训练锚上. 行区间用 `:`, 段落用 `##`,
字段用 `#`——三个语法不冲突. 两条新类型是 K41 的候选, 待 dogfooding 验证.

**Ghost 认知场草图 (记录, 不动代码)** — ghost home 子场结构:
`GROUND.md` (根场, body: ghost 是谁 + what/why) / `users/` (用户记忆文体) /
`memory/` (持久记忆文体, K20 promote 出口) / `skills/` (能力声明文体) /
`tasks/` (运行时侧影, tmp 级) / `tmp/` (纯运行时). 初始化: stub 从 ghost
L1 模板库实例化. 与 project 级认知场是同一套机制在两个宿主上的实例化,
区别仅在 bootstrap 路径 (ghost home 是 well-known 位置, runtime 知道开门
在哪, 不需要发现链引导).

### 2026-07-14 认知场对齐 (行业对照 + 偏差盘点 + L2 方向)

**K14. 落 channel: 父 desktop + 每场 virtual channel** — Desktop 是父
PrimeChannel, 持有 `open` / `close` / `pin` / `unpin` / `update` 动词.
每个打开的场作为 command-less virtual channel 挂到父 desktop 上:
- `instruction` 闭包 → 法链 (向上收集的 CLAUDE.md/AGENTS.md, 到 boundary 为止)
- `context_messages` 闭包 → 帧渲染 (形/焦/贴纸内容 + `changed on disk` 标记)
- `startup` → load (读状态文件重贴)
- `close` → sediment (贴纸簿落盘)

拒绝的方案 (07-12 discuss "结构三选一"): (a) 单 channel 内自治多目录 (God-model
方向, 最近刚在解散); (b) 每场一个带命令集的真子 channel (pin 是纯 context 操作,
不该有 FIFO 轨道; 且命令集重复).

**不关门**: virtual channel 本质是 Channel, 将来某场需要文体专属动词时可挂
带命令的 virtual channel, 按需付费.

**K15. 分形 L 体系 — frontmatter 是唯一 schema 边界** — 层级:
- **L-1** = 领域数据文件 (CLAUDE.md, MOSS.md, CELL.md 等; 现阶段被 bash 直接
  读, 无运行时生命周期)
- **L0** = 场的 `frontmatter + body`. frontmatter = 加载时消费的
  GroundConvention; body = 文体内容 (法/胶囊/先例)
- **L1** = L0 的模板 (类型), 如 `.ai_partners/features/TEMPLATE.md`
- **L2** = 装 L1 模板的目录, 而 L2 目录本身也是一个 L0 加载后能读到的场

**加载路径封闭 / 创造路径开放**: 加载只需读 frontmatter+body, 机制唯一, 不需要
模板参与; 创造 (从 L1 拷模板给新场) 是开放动作. 故分形无递归基问题.

**frontmatter 不是 schema 病复发**: frontmatter 只放运行时生命周期 (MOSS 唯一
合法发明域), body 永远开放集. 物理分界线即机器/模型分界线. 24 年 dev_ctx 死因
是把 body 的事写进了 schema; 本设计里 schema 只覆盖生命周期, 一寸不多.

**实物证明**: `moss features` 体系自 2026-05 起自发运行至今, 就是 "围绕一个
L1 模板做的客制化 CLI". 本设计是给自发实践追认理论再泛化. features 体系应是
desktop 落地后第一个被 open 的场类型.

**丰化梯度** (构造性的向下兼容):
- 裸目录 (缺省约定即可 open) → +CLAUDE.md (法链自动收集) → +L0 文件 (场有了
  进入方式和沉淀) → 从 L1 模板实例化 (场有了文体)
- Claude Code 的静态约定 = 只有 L-1、文体硬编码为一种、无生命周期无 pin 的
  退化情形; skills = 有 L1 但全局无场所绑定的退化情形; 当年 ProjectManager
  四库 = 四个 L1 模板被误写成四套接口

**警戒线**: 模板替换语言保持 `$VAR` 级钝; 表达力来自 "给模型看的先例" 而非
渲染引擎 (拒绝 Jinja/条件块 — 那是 schema 病换装归来).

**K16. 双寄存器词汇 — 表面骑先验, 理论住文档** — 判据: 表面上的每个名词/动词
/语法必须命中一个预训练先验; 理论词汇一律不上表面.

**理论层** (只住 .design/.discuss/FEATURE.md, 模型永远不在 channel 表面看到):
认知场, 法/形/焦, 文体, 胶囊/目光, L-1~L2, 分形.

**表面层** (moss_static / 命令签名 / instruction):

| 表面词 | 骑的先验 |
|--------|----------|
| channel 名 `desktop` | OS 桌面 = 摆放工作集; feature 名 ghost-filesystem-desktop 不用改, 死掉的是原语捆绑不是桌面隐喻 |
| `open` / `close` | 标签页/文档先验, 支持多开; **不用 enter/exit** (后者暗示"身在其一", 与 N 场并开的事实相悖) |
| `pin` / `unpin` | 置顶消息, pinned tabs |
| addr `path` / `path:80-140` / `**/*.py` | 编译器报错 / grep -n / GitHub 行号链接 / glob (三重预训练) |
| `update(addr)` | "bring to current" |
| 变更标记 `"changed on disk"` | VSCode/vim 对话框原话 |
| `frontmatter` | Jekyll / Hugo / SKILL.md / 我们自己的 FEATURE.md |

**验收**: 未参与讨论的新实例仅凭 moss_static 零解释正确使用 (pin 一个行区间,
处理一次 `changed on disk`, update 后继续).

**K17. 桌子即工作记忆, 不与世界自动同步** — pin 钉的是地址不是快照, 但表面
**不自动同步** — 文件变更以 `changed on disk` 标记为 stale, 靠显式 `update(addr)`
推进认知. update 通过 command `<result>` 机制自然入对话历史 (无需新协议).

**语义修正**: 07-11 收敛时曾说 "桌上的永远比世界新" — 更准确的语义是
"桌上的永远是模型上次承认的世界" (工作记忆而非世界窗口). 桌面在任务中途不会
在模型脚下滑动, 变更只在被承认时进入认知.

**对账粒度**:
- **mtime 触发**: 变化时进入待检查
- **区间内容 hash 判定真伪变更**: 文件变了但 pin 住的区间没变, 不打扰模型
- **行区间 v1 必做**: `path:80-140` 语法一阶段实现 (人类工程师: "实际上我在
  很多地方都在手动告知行区间")
- **地址漂移 v1 不做自动重定位**: 标记过期让模型自己重 pin; 内容锚点重寻是
  后续精化

**两种 stale, 别混用词汇**:
- 世界→pin 的 stale = 对账语义 (hash + 显式 update)
- 模板→实例的 stale = 拷贝语义 (frontmatter 血统键 + 手动升级, 无痛)

**K18. 3 层抽象纪律 (内核纪律)** — 内核模块必须 `contracts/` (ABC 契约) →
`core/` (concrete 实现) → `channels/` (channel 装配, 手写闭包) 三层划分.
参照: `src/ghoshell_moss/channels/module_eval_channel.py`. 拒绝规则糖 (类
interface 反射为 channel 之类) — "只有'想要'的时候'才看见'是对的".

24 年过早规则糖的教训: 无人用时非但没收益, 还增加认知成本. features 迭代史
的第一定律也是同一件事: **去发明是主旋律, 且发生得极快** (features 自造的
csv 索引/id 字段/archive 命令都在诞生 48 小时内被删).

**K19. 原生 drain 协议独立立项** — `refresh_meta` 返回值作为 "被覆写的表面
信息" 的守恒律出口是真命题, 但辖域比 desktop 大:
- 涉及 channel 核心 ABC 变更 (所有 channel 的 refresh_meta 语义)
- 涉及 Memento "产生时入记忆" 的联合设计
- drain 载荷天然就是 MomentRecord — 两个东西是同一条管道的两端

**独立 feature 立项**. Desktop v1 用 command 返回值经 `<result>` 入史的既有
路径, 不等 drain.

**慎重点** (立项时优先讨论):
- 历史洪泛: drain 重开了写 O(n) 对话历史的路 (设计本来在保护这个最贵的资源).
  协议必须强制摘要尺寸 + 幂等 (每转移一次只 drain 一次, 靠 hash 对账簿)
- 落点: drain 消息插在帧的哪个位置 (new inputs 前? 系统观察?) — 影响模型把
  它当感知还是当日志
- 核心 ABC 动刀的兼容性: `refresh_meta` 现在返回 None, 改成返回 drain 载荷是
  所有 channel 都要过一遍的语义变更

**K20. 目光落运行时侧影, 不自动 sediment 到 L0 body** — 目光 (session pin)
不通过 close 钩子自动写入治理文件, 避免:
- git 噪音 (每次 close 都产生 diff)
- 违反 "沉淀是主动动作" 的哲学 (auto-sediment 是又一次注入病)

**落点分层**:
- 目光 → 运行时侧影 (.cache 级 gitignore 目录, enter 自动读回)
- 胶囊 (git 见证的持久沉淀) → L0 body, 靠模型显式 `promote` 动作

从 features 迭代史提炼: 头号运营威胁是状态说谎, 解法是显式生命周期纪律
而非自动同步. features README 三周打磨的教训: "CLI does not enforce this,
model incarnations follow it, human reviews for it" — 纪律是模型原生的.

### 2026-07-13 v1 落地打磨

K27~K32 是 concrete 层 + CLI 落地过程中的具体决策. 与 K14~K20 的区别: K14~
K20 是"重绘方向", K27~K32 是"落地路线上遇到的岔口"— 大都是"能推翻抽象不
能将就实现"精神下的即时收敛.

**K27. async ceremony 裁决 — 不容 empty async wrapper** — Python async 契约
的裁决点不是"内部有没有 await", 而是**语言缺陷追认后的 trade off**:
- 同步场景 (CLI) 里调 async 函数要拉起整套 loop 资源.
- Loop 里看到同步函数不敢用 (不知阻塞不阻塞), 100% 卸载线程池.

裁决: **要么 docstring 声明到不容质疑, 要么在两者中二选一并承认 trade off**.

具体落到 Ground / Grounds ABC:
- **`Ground.instruction() -> str` 同步** — 首次挂载缓存, 之后纯状态访问,
  docstring 声明"无 IO 无并发风险, impl 违反是 bug".
- **`Ground.refresh_instruction()` async** — 显式动作走 IO, 与 pin 的
  `update()` 姿态同源.
- **`Ground.context()` async** — IO 密集 (stat + read + hash 全 pin),
  docstring 声明"应用 asyncio.gather 并行, 不得串行, 违反是性能 bug".
- **`Ground.pin()` 同步 + 内部 observe_sync** — pin 是第一人称动作, 允许
  blocking IO 以省一次 loop 拉起 (K17 initial 承认).

**K28. dir-idempotent open — Grounds 幂等按目录 abspath** — 同 dir 再次
open 返回已 active 的 Ground 实例, 传入的 label / convention 被忽略 (以已
active 者为准). 幂等键 = `dir.resolve()`, 而非 label. 理由: **目录是认知
单元, 不是工具** — 一个 dir 只有一份 pin 集与法链, 同 session 内多次 open
无异议看到同一份. 跨 session 持久化由 L0 文件承担, Grounds 层不做记忆.

CLI 场景直接受益: `moss desktop pin a.md && moss desktop pin b.md` 两次进程
调用天然幂等, 每次都新构造 Grounds 但读同一份 L0.

**K29. DESKTOP.md body 入 instruction — K20 promote 的显式出口** — K20 定的
"胶囊 = promote 后的 pin", 但一直没定 promote 后内容住哪 / 从哪呈现. 本轮
定案:
- 载体: **DESKTOP.md body**. sediment 只重写 pin 段, body 保留 verbatim,
  模型或人可编辑, git 见证.
- 呈现: `Ground.instruction()` 返回 `upward CLAUDE.md 链 + DESKTOP.md body`,
  顺序钉死 (根最先 + 本地最后), 与 upward 同格式 (`<!-- from: ... -->` 前
  缀).

**语义分工**: CLAUDE.md 链 = 继承的法 (向上收集, 每层叠加); DESKTOP.md
body = 本 ground 的法 (per-scope, 不上不下); pins = 本 ground 的目光
(per-scope). body 和 pins 局部性对称 — 都随 ground 走.

**K30. tree ignore 走轻方案 — 不引 pathspec, K9 承接高级过滤** — 帧内 tree
段需要过滤 `.git .venv __pycache__ node_modules ...` 之类噪音, 两案选轻:
- 轻: BUILTIN_TREE_IGNORE 常量 + `GroundConvention.tree_ignore_extra: tuple`
  加法口. basename 精确匹配, 不解析 `.gitignore`. 零新依赖.
- 重: 引 pathspec 库, 完整 gitignore 语义 (`**/`, `!`, 路径锚定).

理由: **tree 是模型认知辅助, 不是 build 工具**. 完整 gitignore 语义
(反选 `!except.log` 等) 对模型呈现零价值; 且 K9 (未来 CTML pin bash) 会
承接更精细过滤 (`find | grep -v ...`) —— 现在做完 pathspec, K9 落地时又
一次废弃. 用户加法口留升级路径: 若真需要 pathspec, 换一处即可.

**K31. DESKTOP.md 不向上查找 — 与 K17 一以贯之** — 是否让 open 一个没有
DESKTOP.md 的目录时向上查找? 定案: **不做**.

理由:
- CLAUDE.md 承担继承的法 (向上收集, 每层叠加).
- DESKTOP.md body 承担本 ground 的法 (per-scope).
- pin 集也 per-scope.
- 三者局部性对称. 若 DESKTOP.md 支持向上查找, 会破坏 K17 的"第一人称"
  哲学: 打开哪个场变成 fs 遍历回答, 不是模型说的算. 且 pin 集写回哪里
  会产生歧义 (读上游写本地不对称; 写上游跨 ground 污染).
- 反悔判据: 只有出现真实痛感 (频繁抱怨"每个 workstream 都要手写 tree_depth"
  之类) 才反悔. 目前 0 痛感, 且共享法有 CLAUDE.md 链兜底.

**K32. CLI 三验收之一落地 — `moss desktop`** — K16 声明"三层落地按预训练
迁移量排序" (CTML channel > bash CLI > module_eval). CLI 版本先落地是有意的:
- **三条验收平面** (读写同一份 L0 文件): (a) contracts + core 单测,
  (b) `moss desktop` CLI dogfood, (c) 未来 CTML channel 集成.
- 平面 (b) 提前落地把 K16 的"bash 层"从"阶段二落地"提前为"阶段一验收路径".
  能立刻 dogfood, 且**层无关的直接证据** (同一份 L0 被单测和 CLI 消费).

**workspace_root 探测**: CLI 层用 `Project.discover().root` 作首选 (MOSS
capability 内使用 MOSS 能力, 与 K18 不冲突), 兜底最近的 `.git` / `.moss`
向上, 再兜底 fs root. Core 层不 import Project, 副作用属 discover 不该在
open Ground 时触发.

**越界错误 hint**: `PathOutsideRootError` 消息里附带 workspace 建议
(`moss desktop pin <addr> --in <workspace>`), K12 教育代价落在错误消息上,
不落在用户困惑上.

### 2026-06 原捆绑设计 (K1~K13 按 07 月重绘标注)

Stage 1 (2026-06-29) 的 53 单测代码存在于仓库中, K1~K13 的部分意图仍成立:

| K | 原决策 | 07 月状态 |
|---|--------|-----------|
| K1 | Project 级公共 API, 不进 Matrix | **仍成立** |
| K2 | 以任意目录为 root | **仍成立**, 由 K14 的 `open(dir)` 承接 |
| K3 | 17→12+1 原语三层 (发现/读写/执行) | **形状解散** — 执行迁 subprocesses/job_supervisor; 认知面由 K14/K17 重塑为 open/close/pin/unpin/update. Stage 1 认知面单测 (glob/read/pin/write) 可作形状校验参考, 但 API 变 |
| K4 | `_pin` 参数通用化 | **演化** — pin 从命令参数升格为独立动词, 只收地址 |
| K5 | 统一输出截断 (tmp) | **保留** — 读取超阈值写 tmp + 预览; tmp 路径不重复截断的不变量保留. 但 pin 触达 tmp 的语义在 K17 (对账 + update) 下需重考虑 |
| K6 | read-before-write 元规则 | **重估** — 07-12 结论: 场不做全链路守卫, 由写路径各自的卫生纪律负责. 第一版可不做, 让实践讨债 |
| K7 | frontmatter 原语, 不硬编码约定 | **强化并泛化** — K15 把 frontmatter 提升为整个体系的机器/模型分界 |
| K8 | DESKTOP.md 覆盖默认 instruction | **被 K15 具化** — L0 body 渲染即 instruction; L0 文件名单独 review (K22) |
| K9 | CTML pin (未来) | **仍成立**, 当前不做 |
| K10 | context_messages 组合 | **被 K14 具化** — virtual channel 的 context_messages 闭包 |
| K11 | ProcessManager 底层 | **已迁出** — subprocesses/job_supervisor 承接 |
| K12 | 空间边界零审批 | **仍成立** — 对认知面动词; 执行域由 subprocesses 侧权限管 |
| K13 | Channel 架构 (desktop → terminal/editor/tasks 子 channel) | **被 K14 替代** — desktop 只管认知面; terminal 归 subprocesses; tasks 归 job_supervisor |

## 已知未决 (给下一个实例)

已在 2026-07-13 v1 落地过程中收敛的原 K21/K22/K26: 见 §2026-07-13 v1 落地
打磨 与 K1~K13 状态表. K23/K24/K25 保持未决.

- **K23 (L2 模板库引导地址)** — `.moss/` 侧 (项目所有) 还是 `.ai_partners/`
  侧 (ghost 所有)? 涉及 "模板库是项目的还是 ghost 的" 归属问题.
  (2026-07-14 K38 补: moss 仓库侧已定 — 追认 .ai_partners/ 为 L2 索引场.
  残余问题变为: ghost 携带的 L2 与项目属地的 L2 的关系, 即 K35 合成语义
  在 L2 层级的应用.)
- **K24 (目光运行时侧影落盘位置)** — .cache 级 gitignore 目录的具体位置约定.
  场目录只读时的退化策略 (退到 workspace 侧影目录). K20 已定"目光不自动
  sediment 到 L0 body", 侧影载体本身仍未定. (2026-07-14 K36 补: seen_* 观察
  态字段是侧影的首个确定客户.)
- **K25 (向下探索的场声明)** — 一个场里如果有多个子目录都是 L0 文件, 父场
  frontmatter 里怎么声明 "我下面有场"? 影响 glob 语法 (向上 CLAUDE.md +
  `**/name.md` 向下探测的具体形状). 现在 `hint_children` 只做浅一层子目录
  CLAUDE.md 提示, 未涉及子场声明. (2026-07-14 K39 补: 发现链给 K25 提供了
  形状 — 向下场声明 = marker 文件约定 + glob, L1 声明实例 glob, L2 声明
  L1 marker glob. README 文体与 L2 dogfood 是第一批真实客户.)
- **K33 (channels/desktop_channel.py 装配未验证)** — 已写但未测未验收. CTML
  channel 层的 prompt 效果 (`virtual_children` 的挂载/卸载时序, 每帧刷新的
  `instruction` vs `context` 语义分区) 需要 moss-as-mcp 场景实测. 单元测试
  在这层无用 —— 那层没预训练锁死语义. 手感验收先于自动化.
- **K34 (旧 Stage 1 代码清理)** — `core/desktop/desktop.py` /
  `core/desktop/models.py` 已与新契约不兼容, 保留磁盘为反向索引参考. 旧
  `tests/ghoshell_moss/core/desktop/test_desktop.py` import 已废符号故
  collection error. K26 尾巴, 待独立 cleanup commit.
- **K40 (K35 合成语义与 K28 幂等的冲突 + ghost 默认场)** — 携带 L1
  约定与属地 L0 的分层合成如何进 open 签名与 GroundConvention; 携带方强制
  覆盖 vs 属地优先 的对立立场靠 dogfooding 裁决; ghost 默认场
  (Claude Code `~/.claude` 半边) 的声明位置.
- **K41 (L1 marker 文件名 + pin 类型扩展)** — L2 发现 L1 的 marker 是否
  与 GROUND.md 同名加字段区分, 或独立 marker. pin 类型扩展 (path#field,
  path## heading) 已记录语法候选, 待实现.
- **K43 (.grands/ 设计分支)** — 记录为回退方案. 反悔判据: 多认知方法成为
  真实痛感时启用. 与 GROUND.md 路线的取舍见 K43 条目.
- **多认知方法 (一个目录多种读法)** — GROUND.md 单文件模型下的开放问题.
  未证明需求, 靠 dogfooding 讨债, 不预建机制. K43 .grands/ 是已知回退.
- **Ghost 认知场初始化** — users/memory/skills/tasks/tmp 子场结构草图已定,
  实现等 GROUND.md 闭环 + L1 实例化动词就位后启动.

## 与关联基建的交叉

| 基建 | 关系 | 状态 |
|------|------|------|
| `subprocesses` / `job_supervisor` | 执行域, 从旧 desktop 拆出 (K11/K13 迁出) | 已迁出, contracts 已重绘 |
| `file-editor-contract` | 写路径 + 结构化 view (view/create/str_replace/insert/undo_edit), vendor openhands-aci. contract 独立, channel 层可合体. K6 撤守卫留下的空档由本 feature 承接 | draft, 2026-07-13 立项 |
| `momento-mori` (Memento) | 胶囊 (promote 后的 pin) 落永久记忆; drain 联合设计方 | FORMAT.md 契约层落盘 |
| `Matrix` | desktop 不直接依赖; virtual channel 生命周期由 Channel Runtime 管 | 无直接关系 |
| `features` 体系 | K15 分形体系的实物证明; desktop 落地后应首先支持的场类型 | 已运行, 自 2026-05 |
| `Ghost` / `Mode` | Desktop 进入哪些 mode 是 Ghost 层决策, OS 层不主动推 | 未开始 |
| 原生 drain 协议 (K19) | 独立 feature, 与 Memento 合并设计 | 未立项 |

---

## Stage 1 完成记录 (2026-06-29, Claude Opus 4.7)

> 保留为历史资产. 代码 (`contracts/desktop.py`, `core/desktop/`) 存在, 53 单测
> 绿. 认知面 API 形状按 2026-07 重绘调整 (K14/K17), 执行面已迁出 (K11).
> 以下内容 verbatim 保留, 供反向索引.

### 落地清单

| 文件 | 内容 |
|------|------|
| `src/ghoshell_moss/contracts/desktop.py` | Desktop ABC + ReadHistory Protocol + ReflectionHint dataclass + 全部公开数据模型 (FileContent / ExecResult / Match / Task / PinInfo / DirectoryTree) + 异常 (ReadBeforeWriteError / PathOutsideRootError / PinBudgetExceeded) |
| `src/ghoshell_moss/core/desktop/desktop.py` | DefaultDesktop 实现 — 12+1 原语 + 两条元规则 + LRU pin 预算 + 反思路径白名单 + ProcessManager 可选注入 + 裸 asyncio 兜底 |
| `src/ghoshell_moss/core/desktop/models.py` | PinRecord (实现内部) + InProcessReadHistory (缺省协议实现) |
| `src/ghoshell_moss/core/desktop/__init__.py` | 重导出契约 + 实现 |
| `tests/ghoshell_moss/core/desktop/test_desktop.py` | 53 个 acceptance 单测, 全绿 |

旧 `src/ghoshell_moss/contracts/project_manager.py` 已删除 — 与人类工程师对齐后整体废弃, Desktop 完全覆盖, 无外部 import 引用 (grep 验证).

### Acceptance 边界覆盖情况

- ✅ 12 原语 (+frontmatter 可选) 的契约用 ABC 表达 — 见 `contracts/desktop.py`
- ✅ ReadHistory protocol + 进程内缺省实现 — InProcessReadHistory, 单测注入第三方实现验证可替换
- ✅ read-before-write 守卫在 write/edit 上正确触发 — `test_write_existing_requires_read`, `test_edit_requires_read` 等
- ✅ 统一输出截断 + tmp_path 路径不重复截断 — `test_read_truncation_writes_tmp`, `test_tmp_path_read_does_not_truncate`
- ✅ 反思路径白名单触发 ReflectionHint — 覆盖顶层文件 / 目录前缀 / 自定义白名单 / 命中 vs 不命中
- ✅ Pin 注册 / 查询 / 移除 / LRU 淘汰 — `test_pin_lru_eviction`, `test_pin_lru_refresh_on_repin`, budget warning 标记
- ✅ ProcessManager 注入 vs 裸 subprocess 两条路径行为等价 — `test_exec_via_process_manager_cwd` 对照两路 cwd/exit_code
- ✅ 12 原语全部覆盖单测; read-before-write / 截断 / pin LRU / reflection 边界各有专门单测

### L2 偏差记录 (实现过程中相对 .design 的微调)

1. **`write` 返回类型从 `None` 改为 `ReflectionHint | None`** — .design §5 只描述了 hint 概念, 没有明确返回路径. 选择走返回值而非回调, 符合 §3.2 "Desktop 通过返回值发信号, 上层路由" 的纪律. `edit` 同理返回 `tuple[int, ReflectionHint | None]`.

2. **`Task` 把 `read()` / `cancel()` 做成方法** — .design §7 说 "tasks 返回结构持 `read()` / `cancel()` 方法". 实现侧用 dataclass + bound async callable (`_read`, `_cancel`) 让顶层不再需要 `read_task` / `cancel` 原语, 收口符合 12+1 数. 顶层 ABC 上只剩 `tasks()`.

3. **新建文件不触发 read-before-write** — .design 没明说. 选择: 路径不存在的 write 直接放行 (创建本身就是初始 epistemic 锚点), 路径存在的 write 强制 ReadHistory 命中. 这符合 Claude Code 的行为, 也避免 "为了写新文件先要 read 一个不存在的文件" 的死锁.

4. **`tmp_path` 读取不登记 ReadHistory** — tmp 文件是 Desktop 自己的截断产物, 不是 Ghost 主动观察的代码/配置. 登记 read history 没有反身性语义, 反而污染 Memento branch state.

5. **`reflection_paths` 改为 `dict[str, severity]` 构造参数** — .design §5 给了 5 个默认项 + 单 severity 概念, 实现把它统一成 `{pattern: severity}` 表, 默认值导出为 `DEFAULT_REFLECTION_PATHS`, 让上层可以覆盖. 支持目录前缀 (`.moss/`)、精确名 (`CLAUDE.md`)、glob (`*.toml`).

6. **`Task` 异常用 `KeyError`, 不是 `LookupError`** — `unpin` 不存在的 id 抛 KeyError 符合 dict 语义; `Task.read` / `Task.cancel` 等回调未绑定时抛 `RuntimeError`. 异常分层尽量贴近 Python 内建.

### 已知未决 / 待后续阶段 (2026-06-29 视角, 部分被 07 月 K21~K26 覆盖)

- **`frontmatter` 去留**: 当前保留. L1 (Stage 2 module_eval 试用) 后定. 倾向于删 — `read(limit=20)` + 模型自解析 YAML 可替代, 没必要做内置依赖 `python-frontmatter` 库. **07-12 补注**: K15 把 frontmatter 提升为体系分界, 此项作废 — frontmatter 是本设计的关键 primitive.
- **shutdown 幂等性的强保证**: 当前实现已是幂等 (set 清空), 但没有专门单测. Stage 2 试用时如果发现 shutdown 重入有问题再加 race condition 单测.
- **跨 worktree 的 Pin fork 行为**: Phase 6 处理.
- **DESKTOP.md 写守卫两步确认**: Phase 2 决策, 当前 reflection 只给 hint 不阻止写入. **07-12 补注**: K20 已裁定不自动 sediment; K6 已重估守卫由写路径各自守.

### 模型纪律自评

- ✅ interface 改一次, 实现和单测同步改一次 — 期间多次往返 (e.g. `Task` 从独立 `read_task`/`cancel` 收成方法, 三个文件一起改)
- ✅ 实现里不出现对 Matrix / Memento / Session 的任何 import — `grep` 验证
- ✅ ReflectionHint / ReadHistory 这类对外接口不预设具体下游 — `_RecordingHistory` 单测证明可外部实现 ReadHistory 而不动 Desktop 源码
- ✅ 没漂移加机制 — 反而把 17 原语缩到 12+1, 把数据模型全部上推到 contracts 让 core 只承担实现

### 下一步 (2026-06-29 视角, 已被 07 月重绘超越)

> 原文: 进入 **Stage 2 (eval channel 试用)** 之前等人类工程师评审 Stage 1 的
> 接口形状. 评审通过后包一个 `module_eval` 形态让模型在 MCP 里 exec desktop 
> API, 暴露 "用起来别扭" 的痛点.

**07-12 结论**: Stage 2 的 module_eval 试用不做 — 那一层无预训练 (K16). 直接
进入按 K14 的 channel 落点重绘 (Stage 2': ABC + concrete + channel 装配, 参照
`module_eval_channel.py` 的 abc→concrete→channel 三层结构). Stage 1 的 53
单测中认知面覆盖可作为新 API 的 acceptance 参考, 但要按新形状重写.
