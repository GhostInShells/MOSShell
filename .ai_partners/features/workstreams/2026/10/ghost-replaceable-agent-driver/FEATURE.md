---
title: Ghost Replaceable Agent Driver
status: draft
status_note: >-
  讨论已留档，当前偏好路径一（dsh-first，补齐 moment bytes 迁移）；最终架构结论搁置，等 RSI 里程碑或第二实现出现再定。
priority: P1
created: 2026-10-28
updated: 2026-10-28
depends:
- dsh-version-tracking
milestone:
description: >-
  定义 ghost 的 agent / session 边界，让驱动它的运行时（今天是 dsh）成为环境里可替换的组件。
  本记录保留候选方案与判据，决策只作为当前偏好，不是最终结论。
---

# Ghost Replaceable Agent Driver

> `moss features set-status ghost-replaceable-agent-driver <status> -m "note"` 更新状态。

## Motivation

Dolores ghost 目前把 dsh 当作 agent 运行时。耦合在变贵，而且已经出现过一次真实事故：

- dsh 从 0.1.5 升到 0.2.0 后，v3 的 session 存档不再可读（v3→v4 迁移被 refuse）——memento 的 anchor 与摘要还在，但**恢复完整会话的 moment bytes 读不出来了**；
- dsh 越来越重，集成机制的灵活性不足；我们实际用到的只有 session logs、web 展示面、模型调用循环；沙箱与编辑能力常被禁用；
- `harness` 这个词把 engine / driver / application 三层糅在一起，导致无法讨论"哪一部分属于谁"。

ghost home 里最重要的数据资产目前是 **memento**（稳定坐标 + 摘要），不是 dsh session logs——session logs 是内容本体，相对 anchor 是次要的。真正要保的是 Dolores 的 ghost 产物与"运行时能稳定升级"这两件事。本 workstream 的目标是：把 dsh 降级为环境里一个可插拔的组件，ghost 不再被某个运行时定义。

## 术语与分层

"harness" 弃用：它命名的是"约束装置"，不是功能，无法承载组件语义。按职责拆开：

| 层 | 职责 | dsh 里的对应物 | MOSS 侧归属 |
|---|---|---|---|
| **model provider** | 模型调用与流式返回 | provider 接入 | `llms` / 外部 SDK |
| **agent driver** | session 语义、agent preset、上下文与工具调用循环 | `session` + `agent preset` | 本 workstream 要定义的契约 |
| **engine** | 进程/网关、workspace 治理、session 存储与鉴权、传输 | gateway、workspace、session 展示面 | Host / Matrix / Subprocesses |
| **application** | 面向人的产品面 | web profile | Shell / Node |

命名候选（暂定）：

- `AgentDriver`：可替换组件，实现 agent/session 语义；
- `Session`：一次连续交互线程，身份由 MOSS 分配、持久轨迹由 MOSS 持有；
- 具体实现：`DshAgentDriver`、未来的 in-house loop。

"driver" 是软件里语义明确的词（DB driver 一类）：固定接口背后的可替换实现。`core` / `kernel` 也被考虑过，但过于通用，不承担"可替换实现"的语义。feature 名备选过 `ghost-agent-driver-component`，最终取 `ghost-replaceable-agent-driver`（单数：交付的是一个契约）。

## 已核对的事实（2026-10-28）

- `GhostMeta` 在 `core/blueprint/ghost.py`，不在 stubs；stubs 只注册实例。
- `deepseek_harness` 只被 `ghosts/dolores` 引用；核心、host、matrix 不直接依赖它。
- dolores 对 moss 的依赖面很宽：core 53 处、memento 16、message 11、ground 9、contracts 6、channels 4。宽版本范围的边界是这张表，不是 `GhostMeta` 一个点。
- dsh 不是 pip 依赖，是外部可执行文件；`DSH_VERSION` 只是文档声明。
- ghost 发现目前是单一扫描根 `MOSS.ghosts`；`ghoshell_moss.stubs.workspace` 里的 `deepseek.py` 直接 import `ghoshell_moss.ghosts.dolores`。如果 dolores 独立发包，这里形成分发环——ghost 包必须自己注册，stub 也必须离开 moss 开箱。
- dsh 的 plugin / stubs 是与 dsh API 绑得最死的部分，比 Python client 更紧。
- **memento 已是权威索引，commit-ref 分离起步就在做**（`memento/abcd.py`）：commit 是纯 anchor（稳定坐标 `{branch_index}-{commit_seq}`，`commits.jsonl` append-only，fork 是引用不是拷贝），Note 是消息唯一权威来源；moment bytes 留在 agent session，memento 只在 `metadata` 里存 `session_id` + `tail` 键，不解读。
- **版本闸已实现**（`dolores/_ego_memento.py`）：`_UNREADABLE_MARKERS` / `SourceUnreadableError` / `_skip`——源 session 格式代际不兼容时写空 note、不标 broken、不静默迁移，两个方向都识别（v4 vs v3，或旧 0.1.5 ego session 在 0.2.0 的 protected head 规则下不合法）。
- 因此本记录前面提的"四条保险"里，import 边界、session 身份归 MOSS、版本闸**已经落地**；真正的缺口是 **moment bytes 的迁移/导出路径**，与 **ghost 冻结/后代程序**。

## 候选方案

拓扑划分（按讨论定）：**A 与 B 是同一道决策题（抽象时机，二选一）**；**C–F 是另一族命题（机制形态，可叠加、互不替代）**。

### 决策轴：A vs B（二选一）

#### A. dsh-first，RSI 优先，抽象延后（当前偏好）

先基于 dsh 快速实现 ghost 运行时自迭代（RSI），产品与功能逻辑成立后，再倒过来提炼 AgentDriver，或由 ghost 自己完成。

- 收益：不阻塞产品验证；抽象等第二实现出现后再定，避免把 dsh 的形状焊进接口。
- 代价（原文）：**方案 A 走偏的代价是一代 ghost 会死。**
  中和后的精确说法：真正丢失的不是 memento 索引，而是"session 格式漂移后，恢复完整历史的 moment bytes 读不出来，后代无法继承"。
- 保险（天级，不阻塞 RSI；**其中 1/3/4 已落地，见"已核对的事实"**）：
  1. **import 边界**：dsh 只允许出现在一个模块；加测试钉住（已满足，补测试）；
  2. **moment bytes 迁移/导出**：权威 anchor 之外，session 内容要有跨格式的导出/迁移路径（**缺口**）；
  3. **session 身份归 MOSS**：memento 坐标 MOSS-owned、append-only、fork 是引用（已落地）；
  4. **版本闸**：启动握手，不认识的格式拒绝并保留，不静默迁移（已落地：`_skip` / `SourceUnreadableError`）。

#### B. 抽象先行，阻塞内核

先定义 Agent / Session 两层，把 dsh 依赖全部屏蔽到抽象下，用 `agents.get("module@version")` 精确组合版本，并兼容所有 memento commit。

- 收益：实现可替换性最早到位。
- 代价：阻塞 RSI，约半个月建模；单实现下的抽象容易偏；接口一旦对外，改造成本从重构变为破坏兼容。
- 关键修正：**版本化寻址不解决数据继承**。旧 memento 能否被新 driver 读，取决于数据 schema 的版本与迁移，不取决于 driver 的定位方式。

### 机制族：C–F（可叠加，不替代 A/B）

#### C. 进程隔离：dsh driver 作为 node

driver 放进独立 venv 的 Matrix node，通过 channel 通信。

- 收益：Python 依赖解耦；可并存多个 dsh 版本；升级/崩溃只重启 node；对齐 MOSS 的 node 架构。
- 代价：dolores 现在深度使用 `deepseek_harness` 类型（session events / trajectory / tool call），要改成事件序列化；流式、背压、取消传播需要重新设计。这是集成面重构，不是打包。

#### D. 固定版本收编：vendor dsh 二进制 / gateway-only 重建

把某一版 dsh 的极简产物打进 Python 包，丢弃 web 面，从 gateway 重做。

- 收益：版本耦合消失，控制权最大。
- 代价：失去上游插件/web 生态；二进制进 wheel 的平台矩阵、体积、许可问题；插件机制不便。对应 `dsh-version-tracking` 的 C 档。

#### E. 每 dsh 版本一个兼容子版本

为每个 dsh 版本发布一个 driver 子版本，保住 dolores 产物机制，运行时可持续升级。

- 收益：保住上游生态；跟随成本可预算。
- 代价：版本矩阵与验证成本；需要能力探测与兼容矩阵。
- 与 A 不冲突：这是"默认追版"的实现方式。

#### F. ghost 包化与发现机制（与 A 并行）

- `ghoshell-dolores` 作为独立 ghost prototype 包；
- `moss ghosts prototypes` 通过 entry points（建议组 `moss.ghost_prototypes`，破坏性变更时升版组名）发现已安装原型；
- entry point 必须指向轻模块（只含元数据 + 惰性工厂），避免 `prototypes` 枚举拉起 dsh；
- stub 由 ghost 包自己提供，或 `moss init` 合并所有 provider；移除 `ghoshell_moss.stubs.workspace` 对 dolores 的硬引用；
- `ghoshell-moss[host]` 是否依赖 dolores 的范围，仍是待决问题（倾向 `[host]` 不依赖，`[openbox]` 才拉）。

## 当前偏好（preference，不是结论）

**路径 A**，并设两个检查点：

- **检查点 A（RSI 里程碑前）**：moment bytes 的迁移/导出路径必须到位；未到位不推 RSI。
- **检查点 B（第二实现出现时）**：自研最小循环、或接第二个模型 provider 时，立即把 AgentDriver 抽象提上日程。

改变偏好的条件：dsh 出现 moment bytes 迁移兜不住的数据破坏；或 RSI 本身要求频繁替换模型循环，使抽象成为前置依赖。

## 待决问题（deferred）

- AgentDriver 契约的具体面：生命周期 / 输入 / 输出事件 / session 能力 / version 与 capabilities 握手。
- 同进程 entry point 与跨进程 node 二选一，还是按能力区分。
- `agents.get("module@version")` 的寻址形式是否成立；它与数据 schema 版本的关系。
- ghost 冻结与后代（继承者 ghost）的程序：权威产物导出格式、后代如何读取。
- moment bytes 的迁移/导出路径：session 内容（v3/v4 格式）如何跨 dsh 格式代际迁移，使 anchor 之后的内容也能被继承。
- ghost prototype 包命名（`ghoshell-dolores`）与 out-of-box 归属。
- RSI 的具体验收：哪些能力属于自迭代，哪些必须人工把关。

## Rejected / 明确否定

- **`harness` 作为接口词**：语义空，糅合三层职责。
- **把 dsh session log 当权威记忆**：0.2.0 的损失正是 moment bytes 读不出来——memento anchor 幸存，但完整历史无法恢复。
- **用版本号分支做兼容**：改走能力探测 + 组名升版。
- **认为抽象层能解决数据继承**：可替换性 ≠ 可读性。
- **让 ghost 完全自己决定接口**：可以作为可能选项，但不能作为唯一计划。

## 关联

- `dsh-version-tracking`：C 档触发条件；本记录的数据事故是其判据的实证。
- `dsh-fusion`：分工裁决（dsh 推理中枢 / moss 记忆执行感知）。
- `ghost-ground-governance`：ghost 产物契约的一部分，无论 dsh 去留都需要。
- 本记录即当前讨论摘要；如需逐字辩论记录，另补 `discuss/`。
