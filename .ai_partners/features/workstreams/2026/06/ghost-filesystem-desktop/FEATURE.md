---
title: Ghost Filesystem Desktop — Ghost 的认知桌面
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-07-12
renamed_from: Project Manager
depends:
  - momento-mori
milestone: 0.1.0
description: >-
  Desktop 是 Ghost 的认知桌子 — context_messages 上可 pin 的表面 + 场 (目录)
  的开合管理. 以 frontmatter+body 的 L0 文件为载体; frontmatter 是运行时生命
  周期的合法发明域, body 永远开放集. 三层落地按预训练迁移量排序: CTML channel
  (主), bash CLI, module_eval (小概率). 与 subprocesses/job_supervisor (执行),
  Memento (过去) 共构反身性基建.
status_note: >-
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
  - `subprocesses` / `job_supervisor` — 从旧 Desktop 拆出的执行域 (审计线外侧)
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

### 2026-07 重绘

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

- **K21 (open/update 语义对齐)** — 07-12 记录时人类工程师标注 "我怀疑 open/
  update 其实我们没有真正对齐. 要推进到抽象重绘时, 对齐比现在容易一些". 推进
  到 ABC 重绘时优先处理.
- **K22 (L0 文件名)** — 全体系唯一发明的名词, 单独 review. 判据: 骑先验.
  候选: `DESKTOP.md` (沿用) / `.ground.md` (新造, 无先验, 不推荐) / 其它.
- **K23 (L2 模板库引导地址)** — `.moss/` 侧 (项目所有) 还是 `.ai_partners/`
  侧 (ghost 所有)? 涉及 "模板库是项目的还是 ghost 的" 归属问题.
- **K24 (目光运行时侧影落盘位置)** — .cache 级 gitignore 目录的具体位置约定.
  场目录只读时的退化策略 (退到 workspace 侧影目录).
- **K25 (向下探索的场声明)** — 一个场里如果有多个子目录都是 L0 文件, 父场
  frontmatter 里怎么声明 "我下面有场"? 影响 glob 语法 (向上 CLAUDE.md +
  `**/name.md` 向下探测的具体形状).
- **K26 (Stage 1 代码的迁移路径)** — 现有 `contracts/desktop.py` + `core/
  desktop/` 的 53 单测代码, 认知面动词 (glob/read/pin/write) 如何过渡到 K14
  的形状: 是重写还是渐进重构; 单测能保留多少作为 acceptance 参考.

## 与关联基建的交叉

| 基建 | 关系 | 状态 |
|------|------|------|
| `subprocesses` / `job_supervisor` | 执行域, 从旧 desktop 拆出 (K11/K13 迁出) | 已迁出, contracts 已重绘 |
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
