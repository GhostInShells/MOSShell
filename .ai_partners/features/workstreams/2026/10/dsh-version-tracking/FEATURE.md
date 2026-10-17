---
created: 2026-10-18
depends:
- dsh-fusion
description: dsh 是外部运行时，其产品面演进会持续改变我们的适配成本。本 workstream 建立长期判定框架： 什么时候继续追、什么时候降级到
  gateway-only、什么时候丢弃或 fork；本期完成 0.2.0 合并。
milestone: 0.2.0
priority: P0
status: completed
status_note: 'stage2 验收达成: 0.2.0 已合并进 dev (771faa64). 判定框架保留, 后续追版周期复开.'
title: DSH 版本追踪 — 追 / 降级 / 丢弃的周期判定
updated: '2026-10-18'
---

# DSH 版本追踪 — 追 / 降级 / 丢弃的周期判定

> Use `moss features set-status dsh-version-tracking <status> -m "note"` to update state.
> 论证基础见 [discuss/2026-10-18-app-first-vs-host-first.md](discuss/2026-10-18-app-first-vs-host-first.md)；
> 可复跑的事实基线见 [research/2026-10-18-dsh-020-to-021-baseline.md](research/2026-10-18-dsh-020-to-021-baseline.md)。

## Motivation

`dsh-fusion` 已经裁决了分工：**dsh 做推理中枢，moss 做记忆 / 执行 / 感知**。但那是
"要不要融合"的裁决，不是"融合到什么程度、跟到哪一版"的裁决。dsh 是外部产品，它每一次
发版都在改变我们的适配成本结构，所以需要一个**长期判定框架**，而不是每次发版临时决定。

驱动这个框架的结构性差异（论证见 discuss/）：

- 上游拓扑的第一公民是 **app（profile）**：入口是 `dsh <profile>`，组合包由 profile 选择，
  "应用 = profile" 是其明示约定；同一内核在不同 app 下暴露不同的控制面。
- moss 拓扑的第一公民是 **host**：交互面由 node 展示，内核不持有面。
- 后果有三条：①驱动一个 live kernel 必须先选一个 app（多启动方式 = 多套启动适配）；
  ②控制面覆盖率随 app 变化；③集成成本与 app 的产品面规模正相关（`web-app` 90 条里
  59 条是 `client-*`）。
- 一处必要修正：`typert` / `api-gateway` / `remote-access` 都在 `base`，内核层存在
  **传输无关**的 Remote 面与准入策略，缺的是绑定。所以"去 web"是**成本问题**，不是无解。

## 阶段归属（stage2 / stage3）

| 阶段 | 内容 | 范围 |
|---|---|---|
| **stage2（本期，收敛）** | 合并 dsh 0.2.0 对齐（0.1.5 → 0.2.0 存在协议上的重大变化，必须合并） | 只做合并；**不做** launcher / Connection 抽象化 |
| **stage3（moss 第一个 rc）** | 不同 launcher 实现、同一 Connection 抽象；并按判据决定是否执行"降级去 web" | 抽象化与集成面收敛 |

判定框架跨阶段存在，执行按阶段切分。stage2 要交付三件事：

1. 把判定框架、指标与阈值落盘（本文件）；
2. **完成 0.2.0 合并**（stage2 唯一验收，见 §本期验收）；
3. 锁定集成面抽象方向（不同 launcher 实现、同一 Connection 抽象）——只锁方向，
   实现归 stage3，本期不动代码。

## 需求边界：取什么，不取什么

| 类别 | 内容 | 立场 |
|---|---|---|
| 依赖（上游做得好） | session 语义（log / surface / 格式迁移）、LLM provider 接入、稳定协议层（Remote / typert） | **取**，且是我们的主要收益 |
| 可选（需单独定价） | plugin 体系、web 体系、多智能体集群概念 | **不取为默认**；只在确实需要时作为适配成本计入 |
| 不使用 | 上游 web 作为 ghost 的交互面 | ghost 的交互面归 moss host/node |

原则：这两组**可分离**（`base` 与 `web-app` 是不同 bundle；非 web 驱动面独立存在），
因此融合可以"取一半"，不必在"全要"与"全不要"之间二选一。

## Key Decisions

**KD1 — stage2 追到 0.2.0，完成合并。** 把 `track/dsh-020` 合并进 `dev` 是 stage2 的
唯一验收项；stage2 **不**在此之上做 launcher / Connection 抽象化重构（那是 stage3 的
承诺）。执行细节见该分支上的 `dsh-020-track` FEATURE（合并后其状态置 completed，
并在 status_note 指向本 workstream，不另立平行判定文档）。

**KD2 — 不在代码里显式区分 0.2.0 / 0.2.1。** 若 0.2.1-rc 能直接启动，也只算兼容面扩大：
`DSH_VERSION` 仅作文档性声明，**不作运行时门控**；兼容性判断走能力探测；新增代码不得
引入 `if version >= …` 之类的分支，也不得为两版各写一条启动路径。

**KD3 — 三档判定，各自带前提 / 触发 / 成本 / 回退。**

| 档 | 含义 | 前提 | 触发信号（满足任一） | 成本 | 回退 |
|---|---|---|---|---|---|
| A 继续追 | 维持现有集成面 | session / llm / 协议面兼容 | 指标全部在阈值内（见 §判据） | 每版核对与少量适配 | — |
| B 降级去 web | 只保留 gateway-only 或进程内驱动 | 不依赖上游 web 作交互面 | UI 侧变更占比 > 50%；或启动变体 ≥ 3；或必须挂 `client-*` 才能拿到所需控制面；或 UI 故障影响 ghost 连续性 | 一个绑定层 / 驱动插件 + base 组合追踪 | 可回到 A（组合可逆） |
| C 丢弃 / fork | 停止跟随；首选 fork 一个 `deepseek-harness-for-moss`（fork 点待定），或自建 moss 专用内核 | B 已执行仍不划算，或协议 / 稳定性破坏 | session / 协议面出现无平滑迁移的破坏性变更；或**上游稳定性不足导致 dsh agent surface 无法保证**（每版都要在内部缝适配，M4 单调上升且 ≥ 5）；或核心控制面只能在 web app 内使用且无法剥离 | 最高（内核与测试重建） | 需要 stage 级决策，不在本 feature 内单方面决定；fork 命名先记下，fork 点（core / 协议层 / 整个 harness）留 stage3 讨论 |

**KD4 — 判据可观测化。** 所有结论必须能落到 §判据 的六个指标上，用观测值说话；
阈值首年用观察值校准，不做精确承诺；每次判定需至少 2 次连续观测，或 1 次硬触发。

**KD5 — 集成面抽象：不同 launcher，同一 Connection。** 方向锁定为"launcher 可替换、
Connection 抽象不变"，契约要素三件套：

- `endpoint`：gateway 地址（不假设由谁启动、启动了哪个 profile）；
- `auth`：**cookie 优先（可持久化），token 仅作 bootstrap**（token 每进程内存生成、
  由 `dsh-web-app` 打印；cookie 的签名密钥落盘于 `$DSH_HOME/.credentials.yaml`）；
- `version-probe`：wire 上没有版本宣告（`RemoteHostFacts` 只有 `home` / `isLoopback`），
  必须自带探测（spawn 读 `dsh --version` / attach 由配置声明 + 能力探测兜底）。

本期只锁方向，不改代码；抽象拆分不追求一次做干净，留给后续执行型 workstream。
**归属 stage3**：stage2 按现有 launcher 形态完成合并，不为抽象化改动任何接口。

**KD6 — 单一轨迹。** 判定只在本 workstream 内更新；不新建平行的"版本判定"文档。
执行型工作（合并、适配层）可以有叶子，但结论必须回流到这里。

**KD7 — 归档纪律。** 只留技术判据与证据，不记录对上游的立场性评价（discuss/ 已按此执行）。
理由是判定要能周期复核，立场会污染复核。

**KD8 — 复核节奏。** 每次上游 rc / 正式版发布后复核；无发布时每 6 周兜底一次。
复核只做三件事：更新观测值 → 重跑判据 → 更新结论与状态。

## 判据与阈值

| 指标 | 定义 | 采集方式 | 0.2.0→0.2.1 观测 |
|---|---|---|---|
| M1 追版本净改动 | 一次追版本在 `src/` 的净行数 + 触及文件数 | `git diff --stat` 于我方分支 | 待合并后测（基线：0.1.5→0.2.0 为 +280/−204，10 文件） |
| M2 上游检查面 | 需人工核对的协议 / 接口文件数 | 沿用 dsh-fusion 的"核心检查面"清单 | 本轮 6 个（2 个帧/传输文件 + 1 个事件 types + 3 个 README/契约） |
| M3 启动变体 | 需支持的 app profile 数 | 统计 launcher 适配器 | 当前 1（web）；gateway-only 仍为 1 |
| M4 缝适配点 | 在 dsh 私有符号 / 直写 / 劫持处的位置数 | `grep` 标记（如 `source.kind`、直写 `user/message`、preset register、persona section） | 待清点（dolores plugin 内可数） |
| M5 连续性风险 | 是否触达 session log / surface 语义或迁移；是否有静默行为变化 | 人工判定，0/1/2 级 | 0（本轮仅增 `plugin:` 记录，`SESSION_FORMAT_VERSION` 仍 4） |
| M6 UI 变更占比 | 上游 `packages/` 内 `apps/web` + `packages/client` 的改动行占比 | `git diff --shortstat` 分组 | ≈32%（50859 / 160608）；文件占比 1375 / 4292 |

阈值（首年校准）：

- **A 继续追**：M2 ≤ 10、M1 ≤ 300 行/版、M3 ≤ 2、M4 不增加、M6 < 50%。
- **B 转降级**：M6 ≥ 50%，或 M3 ≥ 3，或出现"必须挂 `client-*` 才能拿到控制面"的事实，
  或 UI 故障打到 ghost 连续性（人工判级）。
- **C 转丢弃/fork**：session / 协议面出现无平滑迁移的破坏性变更；或 M4 单调上升且 ≥ 5；
  或核心控制面只能在 web app 内使用且无法通过组合剥离；或上游停止提供非 web 驱动面。

## 本期验收（stage2：0.2.0 合并）

stage2 完成即视为 KD1 达成，其余为执行细则。**本期不做 launcher / Connection 抽象化**：
合并过程中遇到接口不顺，记录到本文件 Open Questions，不在 stage2 顺手重构。

- [ ] `DSH_VERSION` 从 `0.1.5-rc.2` 更正为 `0.2.0-rc.2`（文档性声明，不作门控）。
- [ ] `track/dsh-020` 上 `dsh-020-track` FEATURE 置 completed，status_note 指向本 workstream。
- [ ] 清理 track 上的死代码 / 死目录：`dsh_preset/` 与 `_sync_dsh_preset`；
      确认目标环境 `.moss/` 无旧 preset 产物残留（"目录发现已死但文件还在"会造成两套真相）。
- [ ] 合并 `track/dsh-020` → `dev`（当前 4 commits），解决 preset 程序化 register
      与 dev 侧旧路径的冲突。
- [ ] 验收：deepseek ghost 在 0.2.0 上能 `moss-ghost send` 对话，启动与鉴权走既有路径。
- [ ] 不引入 0.2.1 分支判断；不新增按版本切换的代码路径。
- [ ] 记录合并后的 M1–M6 观测值，作为下一轮判定的基线。

**不在本期验收内**：0.2.1 实跑验证（见 Not In Scope）。若顺手通过，只更新 research/，
不改变代码形态。

## 观测基线（0.2.0-rc.2 → 0.2.1-rc.2，2026-10-18）

完整事实与复跑命令见 research/。摘要：

- 协议帧层未变：mux 帧协议与 `session/follow` 传输**逐字节未变**；durable 事件仅增
  `plugin:` 实验记录；`SESSION_FORMAT_VERSION` 仍 4。
- 新增在身份 / 准入层：`remote-access`（方法级 RBAC，在 `base`）、Peer 显式化、
  `$events` 的 `interactions` 模式、`/connect` API Peer 握手。
- 产品面继续增长：18 天 1925 commits；UI 侧 +50859/−8879（1375 文件）；
  `experimental/` +70012/−3306；`web-app` 条目 85→90。
- 版本宣告缺失：wire 上无版本，外部驱动必须自带探测。

读数：**协议面稳定，产品面增长**——正好落在"可以继续追 A，但需要按 M6 / M3 观察"的区间。

## 生命周期与复开

```
in-progress ──(本期合并完成)──▶ parked ──(上游发版 / 每 6 周)──▶ in-progress ──▶ …
                                   │
                                   └──(判定为 C)──▶ dropped（并另立执行型 workstream）
```

- 每次复开：先读本文件 → 更新 §判据 观测值 → 重跑阈值 → 更新 Key Decisions（只追加，
  被推翻的决策标注日期与原因，保留原文）→ 更新 status_note。
- 判定为 C 时，本 workstream 转 dropped，另行讨论"fork 什么"（core 还是协议层），
  不在本文件内预设。

## Open Questions

- "降级去 web"的四件套组合（`base` + `client-connection` + `host-webserver` +
  `api-remotes` + 自有插件）能否启动并拿到全部所需控制面（未实跑）。
- 进程内 peer（`createPeer`）驱动与外部 launcher 的边界：审批是 peer 级的，若要保留
  人审批就必须有 `interactions: 'answer'` 的回答方，这个回答方归 moss 还是 dsh。
- cookie 跨 dsh 重启能否复用（签名密钥已落盘、audience 只绑 host:port；未实测）。
- 若走"进程内驱动 + 自有总线"，launcher 与 Connection 抽象是否还保留外部形态。
- 若触发 C 档：fork 点是 core、协议层还是整个 harness；`deepseek-harness-for-moss`
  的边界与上游同步策略（留 stage3 讨论，本文件不预设）。
- 极简 / 只读的 dolores 组合：工具集收缩会改变"行为版本"，需要把组合版本写进 session
  元数据以保持可追溯（与 KD6 的"文件即真相"一致）。

## Not In Scope

- launcher / Connection 的代码重构（**stage3**；本期只锁方向）。
- 追 0.2.1（只验证"能否直接启动"）。
- fork 或自建内核的具体方案（C 档触发后再立；命名先记为 `deepseek-harness-for-moss`）。
- 把上游 web 改造成 moss 的交互面。

## Design Index

- 论证基础：[discuss/2026-10-18-app-first-vs-host-first.md](discuss/2026-10-18-app-first-vs-host-first.md)
- 事实基线（可复跑）：[research/2026-10-18-dsh-020-to-021-baseline.md](research/2026-10-18-dsh-020-to-021-baseline.md)
- 上游：`dsh-fusion` FEATURE（融合裁决、核心检查面、版本声明）
- 执行叶子：`track/dsh-020` 分支上的 `dsh-020-track` FEATURE（仅该分支，未进 dev）
- vendor 源码：`.ai_partners/features/workstreams/2026/08/dsh-fusion/research/source/deepseek-harness`
  （gitignore；用前先 `git fetch --tags`）

## Methods

- 版本比对只做窄 diff：先对"核心检查面"清单，再看新增包 / 新增笔记，不做全量 diff 阅读。
- 判定证据分三级：**亲手探测**（可复跑命令）> **源码类型/契约** > **推理**；
  推理级别的结论必须标注，不得当作事实入库。
- 复核时优先看 `.agents/notes/implemented/` 的新增笔记标题与摘要——它是上游设计意图的
  低成本索引；再回源码确认。