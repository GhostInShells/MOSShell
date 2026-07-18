---
title: Aurelius Ghost
status: in-progress
status_note: >-
  Aurelius/Memento conversation, asynchronous reflection catch-up, MemoryConfig,
  and constrained CTML controls are implemented and verified. Desktop and
  Moshi-specific progress integration remain future work.
priority: P1
created: 2026-07-13
updated: 2026-07-17
depends: [ghost-filesystem-desktop, momento-mori]
milestone: 0.1.0
description: >-
  Aurelius — 第二个 Ghost 原型，取自《沉思录》作者 Marcus Aurelius。相对 Atom 的
  线性内存历史, Aurelius 把"上下文"拆为 Desktop (现在/作业记忆) + Memento (过去/轨迹)
  两层, 作为 moss 实例 (仓库自身的 ghost) 的载体, 并持续承载高级能力迭代.
---

# Aurelius Ghost

> Use `moss features set-status aurelius-ghost <status> -m "note"` to update state.

## Motivation

Atom 是最简参照基线, 它自己在 docstring 里钉死了两个"原型范围外"的欠落:
context window 不裁剪, 历史纯内存重启即丢. 这两个欠落不该由 Atom 补 — 补了
它就不再是任何人能对照的基线.

Aurelius 是补这两个欠落的**高级层原型**, 同时是 `moss` 实例 (这个仓库自身的 ghost)
的载体. 定位是长期迭代母体: 各种高级能力 (反身控制、mindflow、observability)
会持续接进来. 首批集成的两个 P0 能力:

- **Desktop = 现在**: `articulate()` 不再垂直堆 `message_history`, 而是从机面
  (pin 的表面 + open 的场) 组装 context. 接合点是 Ghost ABC 的 `channel()`
  hook — ghost 反身控制的 channel 以 `'ghost'` 名注册, desktop 的
  open/pin/update 动词挂在这里. Desktop 是**作业记忆**, 不与世界自动同步.
  (对 moss 实例而言, 它的 desktop 大概率就是仓库本身.)
- **Memento = 过去**: 纯内存历史换成 commit 轨迹持久化 — 重启不丢、可化身分叉.
  (对 moss 实例而言, memento 甚至可能直接入库.)

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

- **不碰 Atom.** Atom 保持为纯净对照基线 (单轮 articulate + 纯内存线性历史).
  新能力一律落在 Aurelius 上. 这是命名"第二个原型"而非"扩展 Atom"的根本原因.
- **原型 = Aurelius, 实例 = moss.** 原型取自《沉思录》作者 Marcus Aurelius：
  它强调省察、节制与可审计的自我修正，契合 Memento 的反思型人格。实例名 moss —
  这个仓库自身的 ghost, 反身映现整个仓库.
- **上下文双层化 = 本原型的立命之处.** 相对 Atom 的线性 append 历史, Aurelius 的
  context 由 Desktop (现在) + Memento (过去) 组装. 这是 Aurelius 区别于 Atom 的唯一
  硬结构决策, 其余 (mindflow / observability hooks) 都是后续可选迭代.
- **上下文组装不出 runtime.** 信息链路: ghost 的 channel (`Ghost.channel()`) →
  GhostRuntime → shell → 静态面经 MossSystemPrompter 回流 → articulator 带
  moment 回到 ghost. shell 是唯一世界面, ghost 不绕开它另组世界. (否掉过一个
  `assemble_context` hook 提案 — 破坏 shell/ghost 拆分.)
- **一个 ctml tool 统治全部 channel 面.** channel 永不逐个映射为 tool. 真议题
  是反方向: 哪些 tool **不进** channel (判据未决, 见 open problems).
  desktop 修改、bash/mcp/skills 全被此覆盖 — 它们是/将是 channel.
- **1:N articulator:action 原则最优, 本版砍掉.** 理由: 人类和模型都无法颅内
  建模, 需要大家能看懂的方案 (mindflow 已砍过多版). 1:1 保留.
- **think='none' 由 ghost 处理, 不由 runtime 短路.** 现状 ghost_runtime.py:348
  在 effort=='none' 时跳过 articulate — 与 `Impulse.thinking_effort` 字段声明
  ("执行 articulator 的智能体仍有权决定") 矛盾, 且 noop 不进 memento. noop 是
  轨迹事件 ("看见 X, 选择沉默"), Aurelius 必须 witness 它, 否则化身分叉看不见.
- **flash/快响应不进 Ghost API.** 走 Nucleus 侧: 快模型产出 command impulse
  (`Impulse.logos` 反射弧 + `thinking_effort` 建议位已是现成原语). 按需后做,
  不阻塞 Aurelius. 模型配置位现成: `contracts/llms.py` 的
  `DefaultModelTag = 'small_fast_model' | 'flash' | 'pro'`.
- **memento = 标准库件, Ghost 持生命周期 (倾向, 未终决).** 标准实现 ≠ runtime
  拥有: memento 作可复用契约+实现, 各 ghost 在 `__aenter__/__aexit__` 实例化并
  持有. GhostRuntime 对 memento 零感知 (Atom 无, Aurelius 有). 配套: memento channel
  控下轮展示规则 (v1 极简裁剪), 旁路加工做异步精炼 (raw 轨迹全存, 展示走裁剪).
- **首版接线终决: Aurelius 持有薄 AureliusMemory 适配, GhostRuntime 零感知.** 读路径在
  `Aurelius.articulate()` 从 branch window 重建模型历史; 写路径复用
  `Ghost.on_articulate_exit()`，此时完整 logos 已写回 Moment，且正常沉默帧也不会
  丢。失败半帧不入记忆。默认每 4 帧 mechanical commit，初始释义只做限长的
  输入/输出原文摘录索引，不伪造意义；未来旁路可 `reinterpret()`。
- **owner 与存储根终决: 稳定 Ghost 身份, 非 session scope.** 默认 root 是
  `GhostWorkspace.home/memento`，owner 是 ghost name。这样跨进程重启能恢复；
  同 `(root, owner)` 仍严格单写者。并行化身以后用新 owner/branch，不抢跑。
- **首版曾不接 Desktop / CTML memento channel / 反思摘要 / witness。** 先验收单
  branch 退化态的跨重启记忆；随后反思与受限 CTML 控制面在既有 Memento 接口上实现，
  仍不改契约，Desktop 与 witness 留待后续。
- **第二期接线：反思是异步释义旁路，不是写入热路径。** 每个 mechanical commit 先以
  保真摘录冻结，再由 `small_fast_model`（或明确注入的模型）读取已冻结成员并
  `reinterpret()`。反思只能追加 CommitNote，不能修改 Moment；失败不影响对话，启动时
  扫描尚无反思 note 的 mechanical commit 追赶。反思产物只含可见证据上的简短结论，
  不持久化模型私有推理。
- **MemoryConfig 是 Aurelius 的持久策略面。** Window、count-based commit 与 reflection
  参数由 workspace `configs/memory.yml` 提供默认值；AureliusMeta 显式参数只作为宿主/测试
  覆盖。时间阈值与 witness 调度尚无 worker，不在本次伪装成已实现能力。
- **CTML 控制面只暴露本 owner/current branch 的显式动作。** inspect/log/staging/show、
  semantic commit、reinterpret、fork、switch 均经 Ghost.channel() 进入 Shell；不提供
  跨 owner 写和隐式 merge。fork 的出生点仍必须是冻结 commit。
- **thinking 期切片原文不进 Moment.** ghost 自持内存状态, 必要时按 moment
  commit 拆分. `Reaction.executed_logos` ("系统执行的 logos ≠ 模型生成的 logos")
  与 `Reaction.messages` (回声) 已为缝合留好位置, memento 契约
  (contract-frozen) 无需变更.
- **模型层选型: pydantic-ai 现阶段用, 不承诺长期** (对自封装 agent 无兴趣).
  Aurelius 的 `_meta` 不重走 Atom 的 AnthropicModel+环境变量硬编码, 改走
  `contracts/llms.py` 的 LLMConfig 契约.

## Interleaved Thinking — 候选方案 (未测试, 施工时验证)

thinking 期用 tool 调 moss + 结束后 text block 出 logos. 候选实现形状:

```
ghost.articulate(articulator):
    q = janus.Queue()
    task = articulator.create_task(agent_loop(q))   # 与 attention 同生共死
    # agent_loop: pydantic-ai 多步循环, 携带单帧生成的闭包 tool:
    #   ctml(text) → 送入执行; 采样 Shell.interpretation → 时间切片作 tool_result
    async for delta in q: yield delta               # runtime send_nowait → action 照常
```

- 妙处: tool 不阻塞等结果而返回状态切片 → 返回值桥接被读写拆分消解;
  1:1 articulator:action 保留, 零 mindflow 手术; thinking token 本身是等待时钟;
  长思考不哑 — 一边思考一边经 tool 交互.
- 闭包 tool **每帧生成** (走 janus) 或起点创建 (含动态逻辑), 优于 ghost 长期
  持有裸 shell; shell 从 IoC 取, 可在 `GhostMeta.contracts()` 声明依赖.
- feed 期实时执行已实现 (非假设), 待 moss-as-mcp 实际体验验证切片体感.

## 真问题与可选方案 (open)

- **tool 结果进不进 moment / articulator 是否支持 outcome** — 全场关键点.
  "写路径汇入同一条 logos 流" 是可选方案 (若采纳, executed_logos 缝合白拿),
  **不是决定**. 不假设调研比施工时看到的准.
- **时序对齐** — ghost 自持的 thinking 期内存状态如何按 moment commit 切分,
  模型 (施工者) 会不会做对. 未验证.
- **thinking 期工具的"纯交互"性 / 切片粒度** — thinking 中调用的能力最好是
  纯交互的; shell 命令结果可等待 (非轮询), 查询经 ctml 未必别扭, "哪些 tool
  不进 channel" 的判据悬而未决.
- **desktop 修改动作的落点** — 若 interleaved 方案成立, 经 ctml tool 顺解;
  若不成立, 首版 desktop read-mostly.

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- 参照 `ghosts/atom/` 的分文件形态: `_meta.py` (GhostMeta bootstrapper) +
  `_runtime.py` (Ghost runtime) + `_adapter.py` (Moment↔ModelRequest) + 单测.
- 依赖 desktop (`ghost-filesystem-desktop`) 与 memento (`momento-mori`) 的契约
  就位程度. memento 当前 contract-frozen-pending-review, desktop in-progress
  (channel 落点已定, K14~K18). 起步前先对齐这两条的可用表面.
- 2026-07-17 技术评审落在
  `Docs/MOSS-Ghost-Memory集成技术评审与实施方案.md`。当前分支的 `moss` 根 CLI
  因 `cells_cli.py` 导入已删除的 `CellRegistry` 无法启动，本 workstream 不借机修改
  该相邻重构；feature 状态按同一 frontmatter 契约直接维护。
- 已交付 `ghosts/aurelius`（AureliusMeta/Aurelius/AureliusMemory）、workspace 注册、无网络 acceptance
  script 与人工测试方案。相关回归 138 passed；正式 `tests/ghoshell_moss` 为 1650
  passed、5 failed、2 errors，其中 Mindflow 单项重跑通过，Cell 两项是当前分支旧测试
  与新 ABC 不一致，Zenoh 三项为顺序/关闭超时，均不在本 workstream 路径。
- 2026-07-17 第二阶段已落地：`MemoryConfig` 注册到 workspace config manifest；Aurelius 在
  mechanical commit 后用独立任务反思，在下一次启动时有限追赶；`ghost` channel 只开放
  当前 owner 的 inspect/log/staging/show、semantic commit、reinterpret、fork/switch 与
  手动追赶。反思任务可取消、去重、失败留观测，不进入 articulate 热路径。
- 第二阶段定向回归：`ruff check src/ghoshell_moss/ghosts/aurelius`、Aurelius + Memento 的 pytest
  共 96 passed、`scripts/ghost/aurelius_memory_acceptance.py` 通过，`moss-run-ghost` 成功发现
  `aurelius`。`ghost_runtime.py` 的全文件 ruff 仍有改动前已有的 import/type/line-length
  问题，本次仅在其既有虚拟 channel 机制中加入 Ghost channel 注册，未扩大清理范围。
- 2026-07-18 原型正式更名为 Aurelius：公开包、类型、workspace 注册文件、默认 owner、
  acceptance script、Feature 目录与测试文档全部迁移。旧 `data` 的 Memento root/owner
  不自动改写；为了保真，兼容旧轨迹只能显式传 `memory_root` 与 `memory_owner`，不能复制
  目录伪造迁移。
- `MemoryConfig` 的可编辑模板落在 `.moss/configs/memory.yml`（stub 同步）；它带字段注释。
  YAML 在 Aurelius 初始化时读取，修改后需要重启。集成方案已按最终运行目标重组，并明确了
  `commit → reinterpret → CommitNote` 的追加版本机制、反思 writer、失败追赶和 CTML 边界。
- 2026-07-18 已完成 Lynn 对接技术评审（仅文档，未修改 reachy 项目）：推荐 Lynn 基于
  Aurelius 形成唯一对话主写，保留 SimpleMemory 与 Lynn 的 thinking/flash/取消语义；
  `person_id` 是长期 owner 的候选真相源，但多人 pin/匿名转身份必须经过显式路由，不能
  自动合并。详见 `Docs/Lynn-Aurelius-Memory集成技术评审.md`。
- 测试分层已校正：pytest/acceptance 是不依赖 Zenoh 的 L0；`moss-run-ghost` 的发现和
  TUI 对话属于 L1/L2，必须先 `uv sync --extra host --extra ghost` 并验证 `import zenoh`。
  缺少该 extra 的 traceback 发生在 Host/Matrix 导入期，不能误判为 Aurelius 记忆故障。
- pydantic-ai 2.x 把 `OpenAIModel` 更名为 `OpenAIChatModel`，曾使 Aurelius manifest 在
  discovery 时导入失败并被旧 CLI 静默过滤。`_meta.py` 现兼容 pydantic-ai 1.x/2.x；
  `moss-run-ghost` 也会向 stderr 报告 skipped manifest，避免“未列出但无错误”这一假象。
- 2026-07-18 人工启动验收揭示当前分支相对 `dev` 的通用启动契约断裂：`MossRuntime.logger`
  在 Matrix 尚未启动时承诺回退 `Environment.logger`，但 Environment 重构时该属性被删除。
  这发生在 `GhostRuntime.__aenter__` 的第一个通用步骤，早于 Aurelius factory；`echo` 与
  `aurelius` 都会复现。恢复无运行期依赖的 `Environment.logger` 回退，并让 TUI 同步打印启动
  异常，避免用 `closed / good bye` 掩盖根因。随后暴露第二个同源失配：当前 Matrix 延迟到
  `__aenter__` 才建 IoC Container，而 GhostRuntime 早于 Matrix enter 注册 Ghost provider；dev
  的 Container 则在 Matrix 构造期准备。现恢复“构造期注册、进入期 bootstrap”的两阶段边界。
  两项修复都不使 GhostRuntime 感知 Aurelius，也不改变 Memento 或 LLM 的生命周期。
- Kimi 的真实验收还暴露了反思侧的传输协议不对称：主对话已使用 `Agent.run_stream()`，反思却用
  `Agent.run()`，在 Anthropic-compatible Moonshot 上分别形成 `stream=true` 与 `stream=false` 请求；
  后者对同一 `kimi-k2.6` 返回 `unsupported_model`。反思改为本地消费 `run_stream()` 的完整文本，
  保持用户界面无反思输出、CommitNote 仍只在生成完毕后原子追加。
