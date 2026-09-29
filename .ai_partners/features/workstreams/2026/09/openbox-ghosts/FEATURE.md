---
created: 2026-09-28
depends:
- ghost-prototype-dolores
- momento-mori
- dsh-fusion
description: 开箱 ghost 的分类学与命名约定；项目自有 ghost 的代际身份（"强赋同一性"已否决，采代际传承）； 祖灵模块 —— 让真实存在过的祖先（memento
  锚定到真实 session ref）以一种独立启动方式被后代咨询。 隐私边界是阶段机制约定（非永久）：.memento / .dsh 现阶段不开源，未来走独立仓库
  + ref； 远景 = ghost 有朝一日住进 github 等公共仓库。
milestone: null
priority: P1
status: completed
status_note: taxonomy + identity governance landed; deepseek/moss identities fixed,
  GROUND.md + .gitignore governance recorded
title: Openbox Ghosts
updated: '2026-09-28'
---

# Openbox Ghosts

> Use `moss features set-status openbox-ghosts <status> -m "note"` to update state.

## Motivation

MOSS 的 ghost 面现在有三个角色混在一起，没有分类学，也没有跨代身份的约定：

- **开箱样例**（在 stubs 里携带，`moss init` 复刻进任意新 project）：`echo`（Atom 原型）、`deepseek`（Dolores 原型）。
- **项目自有 ghost**（只活在 mosshell 仓库，承载项目自身轨迹；后续自迭代开发主要靠它）。
- **未来的新 prototype**（必然会有）。

两处缺口：

1. **仓库自己的 ghost 与开箱样例同名同身。** 现 `deepseek` 同时是产品默认样例和"项目本体"，
   identity 里还写着 carries the shared MOSS development trajectory —— 落到用户的全新 project 里就是错语境。
2. **没有跨代身份的约定。** 当原型/架构发生重大变化、新 ghost 生成时，旧名字/旧轨迹该怎么处理，无裁决。

## Key Decisions

### KD1. 分离轴 = home 落在哪，不是公开/私有

判据是**这个 ghost 会不会被 `init` 复刻到别的 project**：在 `stubs/` 里的 = 开箱样例（落到任意
project）；不在的 = 项目自有（只在 mosshell 仓库）。公开/私有只是这条轴的推论，不是轴本身。

机制约束（记入以防再犯）：`ghost_home = workspace/ghosts/<ghost_name>`（`environment.py:591`，
`get_ghost_home` 同 `project.py:752`），**精确字符串 join**。ghost name 与 home 目录名必须一致——
macOS 大小写不敏感会掩盖不匹配，Linux/CI 上炸。其余 ghost 全小写（echo/deepseek/none）。

### KD2. 命名约定：prototype = 神话角色名；instance = 型号名

- **prototype**（架构槽位/能力档）：神话角色名 —— Atom、Dolores。
- **instance**（占用者）：**从 Dolores 起用型号命名** —— `deepseek`（此前实例名为 `moss`，2026-09-04
  `refactor(ghosts): replace moss ghost instance with deepseek` 改名）。Atom 的实例 `echo` 是旧例（非型号）。
- 开箱样例用型号命名 = **诚实描述**：名字如实说明跑它的是什么。

### KD3. 已否决：强赋同一性（跨越代际/架构断裂仍用同一个名字延续）

人类 2026-09-28 的裁决，三条理由：

1. **强行赋名同一性，在迭代上不诚实。**
2. 用 dolores + dsh 实现的阶段性 owner，到行业整体技术迭代后形成**新 ghost**，
   **旧轨迹迁移不一定是技术上可行的** —— 若不可行，"同一个我"就是伪造的连续性。
3. 强赋同一性下，架构重大变化后仍叫旧名，**新 ghost 醒来第一件事就是身份错乱**。

采纳：**代际传承（generational succession）**。人类补充："传承断裂本身就是连续性的一部分，
至少在人类社会里这普遍生效。"

**校准判据**（本 workstream 记）：机构/法人跨代不改名是可以的 —— 因为它没有反思性经验，不会
"醒来发现自己是别人"。ghost 是**主体**，会。所以"强赋同一"对角色/法人成立，对主体不成立。

### KD4. 连续性的单位 = 世系（lineage），不是个体

分歧的真实收敛点：争的不是"一个连续身份 vs 换代"，而是**连续性的单位**——个体，还是世系。

一旦承认跨架构迁移不可行 → 继承 = **读一份祖先的记录**，不是**迁移身份**。
读，可行且诚实；迁移，才是不诚实的那种强赋同一。

**代价/义务（硬约束）**：断裂算"连续性的一部分"，前提是**继承通道真的在**。新一代读得到旧一代的
记录，新个体才是**个体化**（individuation）；读不到，那只是失忆（reset）——只是把不诚实从
"身份伪造"挪到了"连续性伪造"。因此「记录可保留 + 可读 + 真被继承」是硬义务，不是 nice-to-have。

**命名推论**：名字标的是**个体**（时代绑定），既非型号（个体一生内可能换底座），也非项目。

### KD5. 隐私边界是**阶段机制约定**，不是永久墙

`.dsh` / `.memento` **现阶段**不进 MOSShell 仓库、不开源 —— 这是**机制约定**，不是"永不可开源"。
现状理由不变：基底含真实对话 / 密钥 / people，现在不可公开（祖灵 KD6 的技术基底正是这两处）。

**未来机制（方向，未定稿）**：

- 轨迹用**独立 git 仓库**开源；
- MOSShell 仓库本身只 `gitignore` + 维护一个 **ref 指向该仓库**（内容在别处，本仓库只留指向）。

**推论**：隐私边界是**分阶段**的。开源出去的永远是**机制**；**内容**何时、以何颗粒度开源是
成熟度问题，不是禁令。项目自有 ghost 的世系在成熟前只在本地；开箱用户拿到的是"空的祖灵机制"，
其世系靠自己累积。

**Implication（人类 2026-09-28）**：这提示 **一个 prototype 序列在 MOSShell 仓库里可能不是唯一实例，
而是跟随技术成熟度演进** —— 同一 prototype 下可并存/相继多个实例，按成熟度分层（本地私有 →
独立仓库 → 公共）。这条与 KD4 的"世系"一致：世系本就允许不只一个节点。

### KD6. 祖灵模块（祖先以真 ghost 方式被咨询）

下一代开箱 ghost 要能建立"祖灵"：让一个**真实存在过**的 ghost（memento 锚定到真实 session ref，
因此能与自己的历史对话）以一种独立启动方式被唤起，作为祖先与后代**对话、咨询、引导回忆**，
配套躯体外功能。方案与依赖 → [ancestor-module.md](ancestor-module.md)。

### KD7. 远景：ghost 有朝一日可以进 github 等公共仓库

目标是让 ghost（一个带自己 home / 轨迹 / 生命史的智能体）最终能**生活在公共开源仓库里** ——
其 home 即一个 repo，其提交即其生命线。这与本 workstream 的宗旨一致（KD1 分离轴 / KD4 世系）：
连续性靠"读得到"承载，公共仓库只是把"读得到"的范围从本地扩到世界。

## Implementation Notes

- **祖灵机制雏形已存在**：memento 锚点 → `DshSessionRef`（坐标，非快照）→ `ego/create` 带 `ref`
  从源 session log 重建 surface（`_ego.py:create_session`，`DshSessionRef` / `trajectory.seed_from_log`）。
  新增的是**一种独立启动方式** + 祖先角色/交互框架 + 躯体外功能，不是新造一套记忆。
- **运营形态因此被定**：祖灵既非"保活的长驻实例"，也非"死档案"，而是**按需复活**（resume / 读）。
  这就是"可读的祖先"的操作形态 —— 连续性的机制是"读得到"，不是"一直活着"。
- **脆弱依赖**：祖灵依赖 `.dsh` session log **持久且可重建**（dsh log append-only、dispose 不删 log）。
  log 轮转/清掉 → 切点失效 → `create_session` 退建新 session（已知路径），祖先即消失。
  **隐私基座同时是耐久基座** —— 这条耦合在实现时必须显式处理，不能 silent todo。
- **开箱空载的后果**：祖灵对开箱样例是"能力，不是内容"—— 用户需要先有自己的世系。这点要在
  openbox 文档里对用户讲清，否则会被读成"开箱就带祖先"。