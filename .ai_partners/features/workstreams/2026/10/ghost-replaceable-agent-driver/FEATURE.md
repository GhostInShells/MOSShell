---
title: Ghost Replaceable Agent Driver
status: in-progress
status_note: >-
  0.2.0 接缝搬家：接缝与 chat 已落地；note 仍走旁路、清账与删旧线未做（见「2026-11-11 执行计划」现状）。
  当前进入机制正规化：commit 生命周期 / 元机制资产 / 反身面 —— 目标与判据见文末「2026-11-11 夜」节；
  细节未对齐，一小步一小步来。
priority: P1
created: 2026-10-28
updated: 2026-11-11
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

---

# 2026-11-11 执行计划：0.2.0 接缝搬家（明天从这里继续）

> 这一节是行动面，不是新一轮讨论。上面的候选方案/判据仍然有效；本节只记录已核实的破坏面、他的裁决、和按顺序的施工计划。

## 目标

dolores ghost prototype 改用 **`ghoshell_dsh.v0_2_0` + `src/ghoshell_moss/deepseek_harness/v0_2_0`**；ghost home 的 `.dsh` 由 0.2.0 使用；之后删掉无版本的旧 seam 与旧 node。**本期不做 concurrent mind，也不做未来版本的 home 机制。** 验收 = ghost 能重启。

## 现状（2026-11-11 23:55）

- **1 插件与 patch**：完成（见 15:00 记录；形状以该记录为准——不装 `@local/ghoshell-dsh`，`plugin_patch` 保持单数）。
- **2 Dolores 换接缝**：完成（已真机跑）。
- **3 旁路改 chatOffRecord**：chat 完成；**note 未**（仍走 bypass 路由，每条落一个 dsh session）。
- **4 清账**：**未做**（`@ home` 336 个 session / 156MB；抽样 40 个里 28 个含 note prompt）。
- **5 删旧线**：**未做**（`deepseek_harness/*.py`、`nodes/deepseek-harness`、6 份旧测试仍在）。
- **6 收尾**：部分（ghost 已能重启；旧线删除的验证未做）。

## 已核实的事实（D11 00:05–00:20）

**引擎不是升级，是接缝搬家**：ghost 今天已在跑 `dsh 0.2.0-rc.2`（当前 ego session 是 v4）；旧 seam 已被补到 0.2.0（Remote、workspace follow、v3→v4 注释），所以破坏主要在 import，不在行为。

- 旧 seam 消费者：`ghosts/dolores/{_ego,_ego_memento,_run,_runtime,_tools}.py` + `test_dolores.py`（12 处 import）；旧 node `nodes/deepseek-harness/main.py`；旧测试 6 份（`test_session/runtime/channel/message_mapper/launcher/trajectory`）。
- v0_2_0 是旧面的**超集**：旧面用到的 session 动词全在；类型名 44 个同名（旧只多 `Self`）；`DshSessionRef` 字段一致（turn 主、seq 辅）；`render_transcript` 签名一致。
- 差异：`render_transcript` 现在走 `blueprint.transcript.TranscriptView`，渲染文本可能与旧实现有细微漂移（迁移后拿一段真实历史对比 memory 视图/note）。
- 插件：ghost 的 `.dsh` **没有** `@local/ghoshell-dsh`；ego 插件是 `profiles/web/` 下的裸 `.ts`；`DshLauncherConfig` 只有单数 `plugin_patch`，而 dsh 的 `--patch` 本身可重复。**ego 插件已经 0.2.0 形状**（`agentPresets.register`、v4 surface），不需要移植。
- 数据账（`.moss/ghosts/moss/.dsh`）：645 个 session 目录 / 535 registry / **198 个被 memento commit 引用** / 337 登记未引用 / 110 未分组；sessions 375MB，projcache 45MB，attachments 107MB。清理可回收约 211MB。
- 旁路：`_ego_memento` 走 `/moss-api/ghost/dolores/bypass/run`，每次 note/chat 用 ego preset **冷 seed 一条新 session**；`moss @ home` 326 条里约 319 条 note 旁路 + 7 条 chat 旁路。
- compact：现有机制是 run 边界阈值（`compact_tokens`）→ 原地 surface 前缀替换 + 新 epoch 的 memory recap；没有“fast compact”实现。模型切换已由 `thinking/enter` 在 turn 边界生效。

## 他的裁决（D11 00:22）

1. **compact 就用现有 run 边界阈值机制**。但提示词有压力泄漏：濒临边界时 ghost 会不敢回复。这时要**提示它主动 compact、切上下文**（不是换掉机制，是补一条主动提示/入口）。
2. **ego session 全部保留**——针对化历史的 ego session 是协作最大资产。**旁路 commit session 本质是 bug**，要清。
3. **未来版本 home**：`engines/dsh-022/.dsh` 这类统一治理；**本期不做未来机制**，0.2.0 直接用 ghost home 的 `.dsh`。
4. **清账只三块**：未分组 / `@ home`（旁路 commit 遗留）/ `works` 下的实验脚本。**不动 ego workspace**。

## 施工顺序

1. **插件与 patch（不改行为）**
   - 把 `@local/ghoshell-dsh` 装进 ghost `.dsh`（复用 `install.py` 的 fingerprint + `refresh_plugin`）。
   - ego 插件打包成 `@local/moss-dolores-ghost`：`package.json`（`dsh.bundle.patch`）+ `index.ts` + `cordis.patch.yml`，装进 `profiles/node_modules/@local/...`。
   - `DshLauncherConfig.plugin_patch` → `plugin_patches: list[Path]`；spawn 时在 `--port` 之前输出多个 `--patch`。profile patch 只留 UI 压制 + provider seed。
   - 重启 ghost，确认两个插件都加载。

2. **Dolores 换接缝**
   - import 切到 `ghoshell_dsh.v0_2_0.launcher/session/trajectory` + `ghoshell_dsh.blueprint.models.*`。
   - 跑 `test_dolores.py`；重启；对比 `render_transcript` 输出。

3. **旁路改 chatOffRecord**
   - note/chat 改为：源 ego session + anchor（end turn → seq）+ prompt，`readonly=True`、`effort=off`、`max_tokens`，走 `DshSession.chat_off_record`；**不建 session**。
   - 锚点 commit 记录 `end_seq`（或在 Python 侧从源 log 反查 turn→seq）。
   - 验证：产一条 note、问一次 chat，确认 session 数不增。

4. **清账（停机、先备份）**
   - 只删三类：未分组 110、`moss @ home` 的旁路（≈326）、`works` 实验 4 条。
   - 同步清 `workspace.json` 的 `sessionIds/archivedSessionIds/pinnedSessionIds`、`session_projcache` 对应条目。
   - 保留 198 个 memento 引用的 ego session + 当前活 session。

5. **删旧线**
   - 删 `deepseek_harness/*.py` 旧模块与 `types/`（保留 `v0_2_0/`，重写父 `__init__.py`）。
   - 删 `nodes/deepseek-harness/`。
   - 删 6 份旧测试；保留 `test_case_agent_hive.py` / `test_v0_2_0_hive_agent.py`。

6. **收尾**
   - ghost 重启验证；harness 101 + ghoshell-dsh 77 重跑。
   - 更新本 FEATURE 的“当前偏好”与 `dsh-version-tracking` 关联；未来版本的 home 机制记入 deferred。

## 验证清单

- ghost 能重启；ego 跑起来；ground/memento 正常。
- note 生产**零新 session**；chat 能从冷/归档 session 回答。
- 当前会话能续（memento 引用的 session 未被清）。
- 旧 seam 删除后 `deepseek_harness/__init__.py` 不再 re-export 旧 launcher。

## 风险与注意

- `--patch` 是 launcher flag：必须排在 `--port`/`--no-open` 之前；`args` 里塞 `--patch` 会变成 app 参数。
- `render_transcript` 的输出漂移会影响 memory 视图与 note；先对比再清账。
- registry 三处集合都要清（`sessionIds` / `archivedSessionIds` / `pinnedSessionIds`）。
- 清账前必须备份；dsh 必须停。
- 本期**不做**：版本 marker、未来 home 命名、concurrent mind、attachments/cache 的完整 GC（只清 session 相关）。

---

# 2026-11-11 15:00 施工记录（第一步：插件与 stubs 复制）— 已收束

> 逐字记录、探针输出与当轮单测数字见 git log（本文件此前的历史版本）。
> 这里只留仍然约束后续工作的硬事实与规则。

## 硬事实

1. **`node_modules` 不能放 `.ts`**（`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`）：`.ts` 源在安装时剥类型，安装副本是 JS；`mode: strip` 保字节偏移，stack trace 仍指源码。
2. **`--dump-config` 只组合、不 import**：它证明 patch 被读到，不证明 dsh 会加载；真判据只能是一次 boot。
3. **`install.py._provision` 必须设 `DSH_HOME`**，否则 provision 写进 `~/.dsh`。
4. **bundle 形状**：`@local/moss-dolores-ghost`（`package.json` + `cordis.patch.yml` + `index.ts`）；stamp 记**源**指纹；迁移只在 boot、spawn 之前，不在 dsh 运行时动活体 home，且只碰我们种过的东西。

## 他定的规则（claude ghost 初始化）

- provider 块必须在 `config:` 下 —— cordis 只把 `entry.config` 交给插件，顶格键没人读。
- 环境声明用 `$NAME`（名字，不是值）：seed 层剥 `$`，写成 `!!js 'process.env.NAME'`；凭据写成 `apiKeyEnv: NAME`。不带 `$` 直接报错。
- `baseURL` 永不落值，只落名字；home 存名字，环境存值。

## 验收三条（他定）

用户不需要理解参数（跑 ghost 就完成初始化）；自动完成版本升级；未来分叉单独做（不做版本 marker / 多代 home）。

---

# 2026-11-11 夜：机制正规化的目标（commit / 元机制 / 反身面）

> 记 **what / why**，不记施工单。机制细节尚未对齐，执行一小步一小步来；下面只写已裁决的约束。

## 为什么现在做

commit 机制这一轮不是 Dolores 的功能迭代，而是内核 driver 的一部分：落锚、旁路生产、元机制资产、反身控制面，都是将来 `GhostAgentDriver` / `GhostEngine` 要接过去的能力。先在 Dolores 上正规化，再抽契约，才不是把原型形状焊进接口。

现有机制的四个失真（问题清单，不是方案）：

1. 增长度量把 seed 当"未提交对话"：重启后第一轮就 warn / force。1-680 在 23:00 落，span=(0,3]，就是现场。
2. 触发检查绑在 turn/end，提醒只能下一帧到达。
3. commit 的语义（谁触发、为什么落）没有 typed 的家；case agent 的 `metadata["operation"]` 就是这个形状，已经是魔法值。
4. 旁路 note 每条造一个真实 dsh session：`@ home` 336 个 / 156MB，抽样 40 个里 28 个含 note prompt。

## 目标（what）

**A. commit 生命周期**
- token 增长是**会话运行时**状态（增量记账、commit 后归零），不挂 memento channel；
- 落锚只在 turn/end 生效：commit 是请求，不是一次写；轮中写的边界也切在轮末；
- ref 能拿到 `end_seq` 就必须记；切点 seq 优先；
- commit 语义（exit / 安全网 / 手写）进 typed 的 note 语义，不进 metadata；
- 优雅退出是 system record：title `exit:` 开头，body 用运行时缓存的 last final answer。

**B. 元机制资产**
- ghost home 有机制资产根（`.dolores/prompts`、`.dolores/configs`），与 ground、bootstrap 配置分开；
- prompt 是文件、seed-once、可改；内核提示词只留一小段，声明"并行心智"和它的话的容器；
- 配置是 typed model、按机制注册、独立于 `.dolores.yml`；watchdog 延后。

**C. 反身面与组织**
- ghost 根 channel 用 prime channel；meta / debug 约定挂 gated virtual children；
- 配置 store 只绑具体类，不占全局 `ConfigStore` 契约；
- Dolores 的思考机制收进 `mindset`（总名），按机制分包；迁移按接缝一刀一刀来，不做大爆炸。

## 判据（why）

- 分类：增长属于会话，记忆属于 memento。混在一起会把"还剩多少上下文"和"记到哪"搅成一条。
- 语义：能被读侧解释的控制含义必须有类型；`metadata` 只存不解释。
- 时间：锚点只能落 turn 边界——seed 切点必须是 turn/end，这是数据契约，不是实现口味。
- 资产：prompt / config 是 ghost 的认知基建，必须可改、可审计、seed-once，不能只活在代码里。
- 组织：机制按"塑造思考"聚合，不按文件类型平铺；`mindset` 是这层聚合的名字。

## 尚未对齐（执行前逐条落定）

- `Note.method` 的保留词表（note / anchor / exit）、case agent 的 op 是否共用、是否暴露给读侧；
- `mindhook` 的最终名与内核提示词措辞；
- 时间戳渲染（本地、分精度、跨年）；
- `.dolores.yml` → `.dolores/configs` 的迁移顺序；
- 旁路 note 是否本轮切 `chat_off_record`（0.2.0 计划第 3 步的存量）；
- `mindset` 的边界与迁移顺序；
- watchdog（明确延后）。

## 明确不做

- 机制语义写进 `metadata` 的魔法 key；
- budget 挂 memento channel；
- 大爆炸改名 / 搬目录；
- 细节未对齐时自动执行。
