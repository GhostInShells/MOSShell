---
created: 2026-07-13
depends:
- matrix-cell-governance
description: Cell-centric acceptance cycle after matrix-cell-governance closure —
  纪律修正 (M1/M2) + wire-up 拉齐 (M3/M4) + telos 主路径 (M7 spawn → M7.5 CellEvent 信号链 → M8
  matrix 聚合 channel → M8.5 L1 tutorial 重构 → M9 自迭代 telos 真验证).
milestone: 0.1.0
priority: P0
status: completed
status_note: M8 closed, M8.1 milestone verified. Regression 13/13 PASS.
title: Cell Run Cycle
updated: '2026-07-20'
---

# Cell Run Cycle

> matrix-cell-governance 抽象层已闭合 (§ZZ-10 收官, wire-up 通过 moss-as-mcp 说话
> 验证). 本 workstream 是它的下游: 让 cell 通过重生的 matrix 真正跑起来,
> tutorials 中的 L1 hello-world 从 apps 语法迁到 cells 语法并走通,
> 直到 UU-11 telos "身体在运行时长出新器官" 的第一个真验证.

## Motivation

**为什么单独开一个 workstream 而不是留在 matrix-cell-governance**:

1. matrix-cell-governance FEATURE.md 已 6 章 (§TT/UU/VV/WW/XX/YY/ZZ),
   继续追加 wire-up 收尾治理会稀释其"抽象决策载体"的定位.
   §WW-0 教训: 判决类结论钉在正文, 但那是**抽象层判决**. 下游治理是不同思维模式.
2. **关注点不同**: 上游 = 设计-推翻-合流 (思维模式偏形而上);
   下游 = 面向下游用户 (人类 + 模型) 的验收纪律 (思维模式偏工程 + 观测).
3. **新化身接力成本**: 单独 feature 只需读一份 (依赖 pointer 到 matrix-cell-governance),
   不用消化 6 章设计辩论即可开工.
4. 本轮实际焦点 = **纪律修正 + cell run cycle 真闭合** — cell 是主语,
   run cycle (create → run → 入网 → interface 进帧) 是它的验收单元.

## 依赖必读

- **matrix-cell-governance** (`.ai_partners/features/workstreams/2026/06/matrix-cell-governance/FEATURE.md`)
  - **§UU 全文** (抽象闭合总纲, 十个判决包)
  - **§WW-5** (九用户故事 + 四弧, run_cell wait 与 signal 分工的直接依据)
  - **§WW-6** (exit 弧 fold 形状, signal 内容规范)
  - **§YY** (Matrix/Project 表面积定稿 + session 永在首页 + home 双目录判决)
  - **§ZZ** (实现层设计对齐, 9 subsections)
  - **§ZZ-10** (TT-2 address 三段结构终审 — 本 workstream 一切 discovery / URL 语义地基)
- **代码入口**:
  - `src/ghoshell_moss/core/blueprint/cell.py` (三域模型 + address helpers)
  - `src/ghoshell_moss/core/blueprint/matrix.py` (Matrix ABC + 表面积)
  - `src/ghoshell_moss/core/blueprint/host.py` (MossHost / MossRuntime ABC — `matrix` 属性)
  - `src/ghoshell_moss/core/blueprint/environment.py` (Environment + seal + discover)
  - `src/ghoshell_moss/matrix/matrix_impl.py` (MatrixImpl 组装)
  - `src/ghoshell_moss/matrix/adapter.py` (MatrixNetworkAdapter ABC)
  - `src/ghoshell_moss/matrix/networks/zenoh_adapter.py` (zenoh driver 实现)
  - `src/ghoshell_moss/host/impl.py` (Host concrete — M1 病灶载体)
  - `src/ghoshell_moss/host/moss_runtime.py` (MossRuntimeImpl — M7.5 nucleus 归宿)
- **tutorial 入口** (M8.5 验收锚):
  - `tutorials/L1_hello-world-app.md` — 当前语法 (apps create / apps:start / apps.xxx:cmd)
  - `tutorials/README.md` — 验证记录追加约定

## 上下文承接

**matrix-cell-governance 收尾状态**:
- 抽象层闭合: cell 三域模型 (Manifest/Record/Presence) + Matrix ABC (§YY 表面积) +
  MatrixNetworkAdapter (§ZZ-3) + build_self_presence / build_host_presence (§ZZ-4) +
  §ZZ-10 三段 address (kind/middle+/uid).
- Wire-up 验证: `moss-as-mcp` 端到端跑通, `execute_ctml` 触发 `__content__` 走 TTS 语音输出
  (2026-07-12 telos 首次实证).
- Environment §UU-1 seal 姿态在三个 CLI 入口显式化 (main.py callback / moss-as-mcp /
  moss-repl), 参数化通路验证 (`--mode default` 触发 mode-level speech/TTS providers 显现).

**遗留 (承本 workstream)**:
- Host 抽象继承纪律未收敛 (`MossHostImpl.matrix()` 有 concrete 上被消费, 详见 M1)
- Environment.discover 契约方向错了 (bootstrap=True 是默认, 导致 CLI 忘 seal 静默降级)
- 剩余 3 个 CLI 入口 (ghost_run / moss_as_fractal / cli_controller) 未对齐 seal 姿态
- TUI 里的 Manifests / Fractal inspectors 被砍掉 (dead API), 待重画或永久删除
- P0 决策未拍板: home 稳定身份键 (dir/name/UUID) — telos 友好 vs code-as-prompt 张力
- Layer 1 cross-scope host 发现推迟 (本期 Watcher 只 subscribe 自己 scope)
- CellEvent → Mindflow 的 signal 链路尚未接上 (M9 telos 的必经环节, 详见 M7.5)

## Milestones

**次序原则**: M1-M4 是纪律与形式修正 (机械或独立决策), M6 是护栏,
M7 → M7.5 → M8 → M8.1 → M8.5 → M9 是 telos 主路径 (依次搭建).
M8.1 是 M8 + desktop-channel 首次联动验证点.
M5.2 / M10 / M11 / M12 是与主路径解耦的独立决策, 挂在合适的时机拍板.

### M1. Host 抽象继承纪律修正

**判据**: `MossHostImpl.matrix()` 转私有 (`_ensure_matrix()` 或类似, 仅供
`MossHostImpl.run()` 内部装配 MossRuntime 时使用). 5 处外部消费点全部迁向
`MossRuntime.matrix` (property, host.py L223 @property @abstractmethod).

**病根 (人类 2026-07-13 明示)**: 问题从来不是 "matrix API 冗余", 而是
**第一版实现违反了继承纪律 — 从 concrete class 上拿本应由 ABC 提供的东西**.
重构一个继承错误的方向应该是把消费者拉回 ABC 面, 不是给 concrete 做 API 减法.

**当前状态诊断** (核实, 修正 FEATURE.md 首版误判):
- `MossHost` ABC (host.py L387) **已经没有 matrix() 方法**. 没什么可"删".
- 问题在 `MossHostImpl.matrix()` (impl.py L104-146) 承担了 MatrixImpl 装配责任
  (build_host_presence + adapter registry + IoC 装配), 且**被外部消费**.
- 5 处外部消费点 (grep `host.matrix()` / `host.matrix\b`):
  - `cli/moss_as_mcp.py:121` (`moss_host.matrix().logger.info(...)`)
  - `cli/moss_as_fractal.py:49` (`host.matrix().logger`)
  - `host/tui.py:560,972,973` (三处: 拿 this / logger)
  - `host/tui_entries/moss_runtime_ui.py:32` (MatrixInspector 构造)

**修正姿态**:
- MatrixImpl 装配保留在 `MossHostImpl`, 但**改从 `run()` 内部调用**,
  或把装配函数抽到 `factory.create_host_matrix(env, project)` 与 `factory.create_matrix`
  平级. 装配是私有实现, 不上继承面.
- 5 处消费者改从 `host.run() → runtime.matrix` (property) 取.
  注意去括号: 老 `host.matrix()` 是方法, 新 `runtime.matrix` 是 property.
- TUI 的两处 `host.matrix()` 需注意 lifecycle — TUI 拿 matrix 早于 `run()` 完成时
  的解法可能需要单独讨论 (M4a 的 Manifests inspector 一并).

**否掉路径 (探索备查)**: FEATURE.md 首版把 M1 描述成 "MossHost ABC 无 matrix() 方法" —
误诊. 目标不是删 ABC 方法 (已经没有), 也不是 host/matrix.py 文件级删除
(该文件不存在或不是问题所在). 病根是继承纪律.

### M2. Environment.discover 契约反相

**判据**: `Environment.discover(*, bootstrap: bool = False)` 成默认.
worker cell 场景显式 `Environment.discover(bootstrap=True)` opt-in 兜底.
CLI 入口全部 `Environment(**cli_args).seal()` 显式构造, 无兜底路径可走.

**病根 (人类 2026-07-13 明示)**: FEATURE.md 首版 M2 拟给 bootstrap 分支加 warning —
方向错了. **该修的是 CLI, 不是 discover**. Worker cell (父进程已注入环境,
子进程一行 `discover()` 完成入网) 是**主路径不是陷阱**, 加 warning 会喊错人.
真正的陷阱是 CLI 入口忘 seal 导致静默兜底, 病灶在结构层, 用 warning 补丁是
反 §UU-1 seal 姿态的意图.

**否掉路径 (探索备查)**:
- 首版方案: `bootstrap=True` 走 warning log. 否掉理由: warning 无差别喷,
  worker 主路径无辜受喷; 结构性病用观测层药膏是治标.
- 折中方案: 按 `ENV_PARENT_CELL_ADDRESS_KEY` 存在与否区分 warning 目标.
  否掉理由: 判据存在但仍是补丁, 直接改契约方向更干净.

**反相后的姿态**:
- CLI 入口 (3 处) 编译期即崩 (bootstrap=False 无 singleton → NotSealedError),
  忘 seal 的静默兜底不再可能 — 结构层解决.
- Worker main.py 显式 `discover(bootstrap=True)` 是自证意图 (作者知道自己在 worker 里
  依赖父环境注入), 无歧义.
- 与 M3 天然协同: M3 改完 3 处 CLI 到 `Environment(**cli_args).seal()`, M2 反相后
  它们才是唯一合法路径; 反过来 M2 反相不做 M3 就编译崩.

### M3. 剩余 3 个 CLI 入口 seal 姿态对齐

**判据**: `cli/ghost_run.py` / `cli/moss_as_fractal.py` / `cli/cli_controller.py`
从 "discover 后 set_mode" 改为 "构造 Environment(**cli_args) + seal".
`Environment.set_mode / set_ghost_name / set_network_scope` 已在 commit 4c75f76b
删除, 现有调用点全部触发 AttributeError, 是本 workstream 的 P1 blocker.

**具体调用点** (grep 核实, 2026-07-13):
- `cli/moss_as_fractal.py:42` (`env.set_mode(mode)`)
- `cli/cli_controller.py:470,476,491,516,518,523` (6 处)
- `cli/ghost_run.py:26,49` (2 处)

**关联** (纪律钉子):
- `host/tui_entries/ghost_ui.py:152` stale code-as-prompt phrasing
- `ghosts/atom/_meta.py:76` docstring ("由 Host.run_ghost() 通过 env.set_ghost_name()
  确保..." 需改)

### M4. TUI Inspectors 处置 (拆两个决策)

首版 M4 "重画 vs 永久删除" 作为整块决策 — 拆.

#### M4a. Manifests inspector 重画

**判据**: TUI Manifests inspector 与新 MatrixManifest+ModeManifests API 对齐,
walk-each-Manifest 而非 dict 视图 (谨慎点: 不复原 God-basket API,
`Manifest.is_error()` 显性化是新契约的核心).

**挂问题 4 讨论**: 人类 2026-07-13 明示 "M4 关键问题是 cell 在 workspace 里的
治理机制是否合理" — 这不是 TUI 症状层, 是 cell inventory / mode / project
归属的**病灶展开点**. 具体展开框架由并行化身改 CLI 时的场景倒逼, 或后续讨论落定.
M4a 的具体重画方案挂到那次讨论之后.

**可能展开面 (探索备查, 非承诺)**:
- CELL.md 生效边界 (cells/ 目录约定 vs 任意目录扫描 vs mode 隔离)
- Cell inventory 与 mode / project 层的归属 (`project.cells` 目前平铺)
- INSTALL.md / CELL.md / MOSS.md / MODE.md 的关系 (谁包谁、谁读谁)
- Cell 在 workspace 里的目录嵌套语义 (cell 里能装 cell 吗)
- home 稳定身份键 (M5.2) 可能是这个更大问题的一个侧面

#### M4b. Fractal inspector 永久删

**判据**: `Fractal` inspector 及 fractal 相关代码本轮清完 —
§TT-14 已弃 (fractal ≠ 拓扑维度, 而是 channel 分形挂载, 真复活时是
Watcher 上的视图不是 Host 侧的 inspector).

**人类 2026-07-13 明示**: "fractal 准备退役, 完成这一轮就彻底删".

**关联**: `cli/moss_as_fractal.py` M3 处理后, fractal 概念的最后遗迹清理归本 milestone.

### M5. P0 决策拍板 — home 稳定身份键 (M5.1 已缩)

#### ~~M5.1. run_cell.wait 默认值~~ (已缩为设计决策, 不需拍板)

**结论 (人类 2026-07-13 定案)**: **wait 参数直接从 run_cell 拿掉**.
所有生命周期跃迁 (spawned / ready / crash / normal exit / 永不入网) 走 signal
进 mindflow (M7.5 承载). run_cell 只返回 spawn 现场合成的 `CellPresence`
(弧起点, 无网络真相), 后续入网状态全部作 background hint 抵达.

- WW-5 "channel 面不 wait, 模型面走 signal" 从半规矩落成硬结构.
- 消除 "wait > 0 是 Python API bootstrap 场景" 的例外 — 例外一存在,
  就有反射调用误传 wait>0 挂 30 秒的可能. 干净删更好.
- 与 M7.5 一体设计, 落 M7.5 时同 PR 完成.

#### M5.2. home 稳定身份键 (P0, 挂 M9 前拍板)

**框架换轴 (2026-07-13 讨论修正)**: FEATURE.md 首版给 (a) 目录路径 /
(b) UUID sidecar / (c) manifest.name 三候选, 问 "哪个够稳" —
框架问错了问题. 应该问的是: **在 UU-11 telos 下作者是谁, MOSS 站哪边?**

- UU-11 telos 的作者主体是**运行时的模型**: 写 CELL.md, 拉起, 用完可能改名或重构目录.
- (a) 目录路径: 模型重构 workspace 布局 (移 cells/ 子目录) = home 丢. 移文件是
  自迭代常见操作.
- (c) manifest.name: 模型改 `name:` 字段 (旧名不够好) = home 丢. 改名也是
  自迭代常见操作.
- (b) UUID sidecar (`.cell_id`): 改名 + 移目录都不丢. `cp -r` 克隆问题非死点
  (`moss cells create <template>` 咽喉强制新 UUID; 手动 cp 侵犯约定).

**真张力 (人类拍板的对峙轴)**:
- **telos 友好** (UUID sidecar): 自迭代日常改名/移目录不丢家, telos 主路径干净.
- **code-as-prompt 可读性** (目录路径 / manifest.name): 模型看 CELL.md 即知道
  自己是谁, 无隐藏状态. UUID sidecar 引入 "身份藏在旁边隐藏文件" 的隐式约定.

**判决权在人类**: MOSS 在这个张力上站哪边, 是价值判断. 挂在 M9 前拍板即可
(M7 用现况 (c) 跑得起来, 只影响 telos 检验时的迁移语义).

**否掉路径 (探索备查)**:
- 首版 CRC-3 把 M5.2 拉高为 M7 blocker. 否掉理由: 只有 M5.1 (已缩) 影响 M7 的
  CTML 反射默认行为, M5.2 是 telos 层的迁移语义决策, M7 spawn 单次跑起来不受影响.
  拉高优先级实际推迟了 M7 落地.

### M6. CTML 契约完整性护栏 (W6 已改, 待补单测)

**判据**: `_load_ctml_prompt` 崩溃姿态强化 (KeyError → RuntimeError, 覆盖
version 缺失 / file 读失败 / content 空三种 fatal 路径), 单测锚 (无 CTML
version 时 MossRuntime 无法构造).

**状态**: 已实施 (moss_runtime.py L142-176), 待补单测. 详见 review W6.

### M7. moss-as-mcp 端到端跑一次真 cell (spawn + presence + 膜可见)

**判据**: 通过 `execute_ctml` 发一个 `<cells:run target="foo"/>` 或等价 CTML,
MCP 侧观察到 cell 入网 announce, `(await mesh()).view()` 能看到膜.
语音 (`__content__`) 之外的膜类型验收.

**为什么**: 语音验证只走了 `__content__` → TTS 一条通路. run_cell 咽喉是 cell
体系核心, 需要独立验证 (咽喉六步全跑通).

**注意**: M7 阶段 M5.2 未拍板 — home 用现况 (c) `manifest.name` 跑, 只是不闭合
自迭代改名场景. M7 通过后 M8/M9 前必须回 M5.2 拍板.

### M7.5. MossRuntime 挂 CellEventNucleus — CellEvent → Signal 链路

**状态 (2026-07-18)**: nucleus 侧实装已落 (f5809ce8, `core/mindflow/
cell_event_nucleus.py`), 纯 signal→impulse 转换器. **生产侧 (mesh.on_event
→ Signal) 判决在 2026-07-18 讨论中改为归 matrix channel** (M8), 不再归
MossRuntime — 两个消费面 (channel ring buffer + nucleus) 都在 matrix
身上, 一份订阅无双写, unsub 跟随 channel 生命周期. 层级修正 (matrix 内 vs
runtime 内) 见下文原始判决段, 保留供轨迹追溯.

**执行文档: 本目录 `cell-event-nucleus.md` (2026-07-13 拆出)**. 含本轮增补:
signal 是 matrix 层可不消费的协议动作 → nucleus 无条件挂载, 不按运行模式分支.

**判据**: MossRuntime 生命周期 aenter 时挂一个专属 nucleus (工作命名
`CellEventNucleus`), 消费 `(await matrix.mesh()).on_event`, 每条 CellEvent 转一条
Signal 送 `runtime.mindflow`. 全部作 **background hint 姿态** (低优 impulse,
闲时才竞争到注意力).

**层级修正 (2026-07-13 讨论)**: FEATURE.md 首版拟挂 Host Matrix — 层级错位.
Matrix 承诺网络 primitives (Presence / Watcher / mesh), Signal ↔ Mindflow ↔
Nucleus 是**认知层**语义, 归 MossRuntime 天经地义. Matrix 保持纯网络门.

**为什么 background hint 而非 notify (Q3.2 定案, 探索备查)**:
- 候选 (A) notify 姿态: 所有 CellEvent 强制入历史 + 触发动作.
- 候选 (B) background hint 姿态: 所有 CellEvent 送 mindflow, 闲时竞争到才被感知.
- **选 (B) 四理由**:
  1. **可逆性**: 先 (B) 后升 (A) = nucleus 加一条 "允许提优先级 + 强制入历史" 分支,
     改动定点. 先 (A) 后降 (B) = 拆双写路径 + 拆已入历史的数据面, 改动扩散.
     难拆的先别做.
  2. **心智模型对齐**: mindflow 就是闲时挑战竞争的哲学. (A) 是 desktop alert 范式
     (强制入 attention), 塞进 mindflow 是第二种范式, 污染仲裁语义. (B) 天然对齐.
  3. **"数据进历史"另有归宿**: M8 落地的 cells channel 的 `context_messages` 承载
     "最近 cells 状态概要 + 最近几轮日志" (channel_builder 既有机制),
     (A) 与 (B) 剩下的真差异只是 "是否强制触发动作" — 动作归 shell/model
     (六动词代数), nucleus 只送候选.
  4. **Q3.3 优先级分档一并消解**: (B) 下所有 CellEvent → 低优 impulse, nucleus 不做
     类型分档 (crash / new-ready / normal-exit 全 background hint). 分档留场景倒逼:
     真的等 crash 静默把人坑到了, 加一条 `exit_code != 0 → high priority` override,
     一行代码. 先不设计.

**风险自陈**: crash 静默的坑可能在 M8.5 (L1 tutorial 重构) 时踩到 (模型拉起 cell →
cell 立刻崩 → 模型不知道, 继续调不存在的 channel). **留监测点**, 不是 blocker:
观察到就升级分档, 观察不到说明这种恐惧本身是空想.

**签约位置**:
- 文件建议 `src/ghoshell_moss/matrix/nuclei/cell_event_nucleus.py`
  (与 `matrix/networks/` 平级, 为未来 audio / vision / topic nucleus 开位).
  最终位置由执行时判断.
- MossRuntime aenter 时注册, aexit 时释放 — 生命周期归 runtime, 不归 matrix.

**payload 规范**: 沿 §WW-6 signal 内容规范 — address/alias, 跃迁, exit code,
日志路径**指针**, 至多 stderr 尾部数行. 日志本体不进 signal.

**与 M5.1 一体**: run_cell 拿掉 wait 参数与本 milestone 一体设计, 同 PR.

### M8. Matrix channel — cell 治理的聚合位

**判据**: matrix 作单一聚合 channel 落地; 承 cell 治理全部对外面 (own
commands + virtual children proxies + 治理 context + CellEvent 生产侧).
`moss-as-mcp` 下 CTML 可发治理动词, 器官挂载后 `matrix.<cell>:<cmd>` 可调.

**执行文档: 本目录 `matrix-channel.md` (2026-07-18 重写, 原名
`cells-channel.md` 已 git mv)**. 结论速览:

- 聚合位: 单一根 `matrix` channel, 承 cell 治理全部对外面. 子拓扑姿态
  (flat / `_nodes` sub-channel / ChannelModule 组装) 为**实装空间**,
  不预定, 由 M8.5 tutorial 走通时倒逼.
- 四动词 (list/read/run/stop) 全 nonblocking; create/install 不映射
  CTML 动词 (走 shell + `moss nodes` CLI); accept/reject 归构造期
  auto_accept flag.
- CellEvent 生产侧归 matrix.on_startup 订阅 mesh.on_event, 双扇出
  (ring buffer 喂 context + send_signal 喂 M7.5 CellEventNucleus).
- virtual children = mesh.channel_proxies() 快照 (refresh_meta 对账,
  get_virtual_children 同步返回).

**前置钉 (实装未落, 在 matrix-channel.md T1 中兑现)**:
- CellHandle.brief() 薄快照封装 (channel 消费入口, 隔离治理知识).
- run_node spawn 补 capture 声明 (WW-6 stderr tail 承诺兑现).

### M8.1. Matrix + Desktop 首次联动 — 运行时自迭代技术基础 (2026-07-20)

**状态: 已完成**. 本次 dogfood 验证了以下闭环:

- **Channel 树拆分正确**: matrix (cell 治理) 与 desktop (OS 工具) 平级挂载, 语义独立.
- **Cell 全生命周期**: list → read → run → proxy mount → 跨进程 CTML → stop → status.
- **Session 信号总线**: `add_signal(NotifySignalMeta)` → `on_signal(callback)` → janus.Queue
  sync→async 卸载 → 跨 cell 接收验证通过.
- **MCP 端到端**: 全部操作通过 CTML 下发, 包括 file_editor 创建 node 代码、bash 执行
  moss nodes 命令、mesh proxy 跨进程调用.

**为什么立 milestone**: M9 (Ghost 自迭代 telos) 需要三块基石 — cell 治理 (matrix)、
OS 工具 (desktop)、信号感知 (session bus). 今天三块基石在运行时首次同时验证, 自迭代
在技术概念上成立. 这不是 M8 的简单完成, 而是两个 feature (cell-run-cycle +
desktop-channel) 在运行时的首次握手.

**验证记录**: `.ai_partners/regressions/nodes-cli/baselines/2026-07-20_m7m8-matrix-dogfood.md`
13 条 case 全部 PASS.

### M8.5. L1 hello-world tutorial 从 apps 迁 cells 语法

**判据**: `tutorials/L1_hello-world-app.md` 全文按 M8 落地的 cells channel 语法
重写, 追加验证记录 (tutorials/README.md 约定的表格格式), 一次走通.

**为什么单独立**: 这是 M8 的 **acceptance test**, 也是人类明示的
milestone 拉直点 ("MILESTONE 必须跑完第一个 tutorial 重构后").
- 落 L1 = 一个人类可读的走通证据, 不只是模型内部说 "跑通了".
- 从 apps 到 cells 的语法迁移暴露实际 UX 摩擦 (`apps create` → `cells create` /
  `apps:start` → `cells:run` / `apps.xxx:cmd` → `cell:channel:cmd` 之类的具体差异).
- 追加验证记录的时间戳与模型身份, 是 workstream 完成度的外部锚点.

**关联**: 具体迁移路径 (命名 / 命令签名 / context_messages) 由 M8 落地实况决定,
本 milestone 是照 M8 结论重写 + 走通 + 记录.

### M9. UU-11 自迭代 telos 第一次真验证

**判据**: 一个 cell 通过 M8 的 cells channel 被创建 + 拉起 + accept 后,
其 channel 接口 (instruction + commands) 自动出现在下一帧模型上下文中.
模型能读到新器官, 能对新器官发命令.

**判决点**: 这是 MOSS 与 bash 本质区别的第一次运行时实证. 若通过,
matrix-cell-governance + cell-run-cycle 两个 workstream 合流达成最初 telos.

**依赖**:
- M5.2 拍板 (home 稳定身份键 — 自迭代改名场景闭合).
- M7 (spawn 咽喉可用).
- M7.5 (CellEvent → Signal 链路 — 新器官 ready 送模型注意力候选).
- M8 (cells channel — 治理动词的 CTML 面).
- M8.5 (L1 tutorial 走通 — 治理动词的人类可用性证据).

### M10. Layer 1 cross-scope host 发现 (方案 α)

**推迟到本 workstream 但明确**: ZenohWatcher 加第二个 subscription
`MOSS/matrix/scopes/*/cells/host/**` 收 cross-scope host liveness.
副路径 `hosts_ns` 代码同一 PR 清 (已 §ZZ-10 语义作废).

**理由**: 本轮 §ZZ-10 落地时的漏项 — 副路径作废但无替代实现补 cross-scope
通道. wire-up 期无 blocking (Layer 2 scope 内发现足够 mcp 说话验证),
但 telos 完整闭合前应补齐. 挂 M9 后作为收尾块.

### M11. 默认 mode 兜底姿态定 (原 task #12)

**待决**:
- (A) workspace `MOSS.md` frontmatter 显式写 `default_mode: default` — 数据层解决
- (B) `MossMeta.default_mode` Field default 从 `NONE_MOSS_MODE` 改 `'default'` — 契约层解决
- 考虑无 default mode workspace 场景.

### M12. 决策目录化 (元级)

**判据**: matrix-cell-governance §TT/UU/VV/WW/XX/YY/ZZ 六章的决策扁平化为
一张判决表 (每条 = 决策 + 载体 + 推翻过的路径 + 现在实现锚). 输出到
`.ai_partners/features/workstreams/2026/06/matrix-cell-governance/DECISIONS.md`
或本 workstream 的 `design/` 下.

**触发时机**: v0.1 tag 前, 不早于 M9 完成 (等 telos 验证过决策才落桩).

## Key Decisions

<!-- 沿用 matrix-cell-governance 的判决目录风格 -->

### CRC-1. 本 workstream 依赖但不 fork matrix-cell-governance

matrix-cell-governance 抽象层已闭合, status → completed. 本 workstream 作
下游依赖, 不重复其判决. 若下游发现抽象层需修 (M5.2 即此类), 修改在
matrix-cell-governance 的 §AAA (若开) 或直接文件层 patch 后钉在本 FEATURE.md
的 Implementation Notes.

### CRC-2. cell 是主语, run cycle 是验收单元

本 workstream 一切 milestone 服务于 "cell create → run → 入网 → interface 进帧"
一个循环. 每 milestone 问一句: 它让这个循环的哪一步更接近闭合? 无答者砍.

### CRC-3. P0 待人类拍板只剩 M5.2, 且已解耦为 M9 前拍板

首版 CRC-3 拟 "M5.1 + M5.2 两条 P0 必须与人类对齐, 阻塞其余 milestone" —
本轮修正:
- M5.1 (wait 参数) 人类直接定案 "拿掉", 不再是拍板项.
- M5.2 (home 稳定身份键) 仍是抽象层修正, 但只 blocking M9 (telos 迁移语义),
  不 blocking M7/M7.5/M8/M8.5. 挂 M9 前拍板.
- 其余 milestone 模型可自主执行. 遇到抽象层未覆盖歧义时按 §VV-1 挡板① 停下问.

### CRC-4. M9 是终局判据, M8.5 是外部锚点

自迭代 telos (M9) 通过前不 v0.1. M8.5 (L1 tutorial 走通并追加验证记录) 是
M9 之前的人类可读证据, 不是仪式 — 走不通即暴露 M8 的 UX 或语义漏洞.
apps → cells 大规模迁移可开另一 workstream (M9 通过后启动).

### CRC-5. 纪律修正 (M1/M2) 是形而下的形而上

M1 (继承纪律) 和 M2 (契约方向) 都不是 "API 减法" 而是**结构方向修正**.
写代码时的姿态直接决定后来实例读代码时的心智模型 — 违反纪律的实现即使能工作,
也是给未来的复利债务. 本 workstream 优先清这两个, 是给 telos 主路径搭干净地基.

### CRC-6. matrix 是 cell 治理的聚合位 (2026-07-18)

M8 从"单 cells channel"合流为"matrix 单聚合 channel". 判决位: 拓扑
聚合位钉住 (`matrix.<cell>:<cmd>`), 子拓扑姿态 (flat / `_nodes` sub-channel
/ ChannelModule) 是**实装空间**不预定; 动词从六缩到四 (create/install/
accept/reject 不映射 CTML); CellEvent 生产侧归 matrix.on_startup 双扇出.
完整探索轨迹见本文档 "2026-07-18 M8 收敛" 节, 实装依据见 `matrix-channel.md`.

## Implementation Notes

### 已知漂移点

- `MossHostImpl.matrix()` 消费点分散 (moss-as-mcp / moss-as-fractal /
  tui.py / tui_entries/moss_runtime_ui.py), M1 重绘时全部改到
  `runtime.matrix` (property, 无括号). grep `host.matrix()` / `host.matrix\b` 找全.
- `env.set_mode / set_ghost_name / set_network_scope` 死方法调用点:
  `cli/ghost_run.py` / `cli/moss_as_fractal.py` / `cli/cli_controller.py` (grep 已定位, 见 M3).
- `host/tui_entries/ghost_ui.py:152` + `ghosts/atom/_meta.py:76` 的 stale docstring
  提到 `env.set_ghost_name`, 需同 M3 一起更新.

### 谨慎点

- **M1 不引入兼容 shim** — 直接改消费者. 5 处消费点全部触及, 一次改完.
- **M2 反相时注意 test 兜底** — 若有 test 依赖 `discover(bootstrap=True)` 默认,
  显式改 `discover(bootstrap=True)`, 不加 fixture 掩盖.
- M4a TUI 重画时警惕不要复原 Manifests god-basket API — `Manifest.is_error()` 显性化
  是新契约的核心. 重画应 walk MatrixManifest / ModeManifests 直接展开每个 Manifest,
  不重塑 dict 视图.
- M5.2 若选 UUID sidecar 姿态, 需处理老 workspace 迁移 (无 `.cell_id` 的 cell
  首次运行自动生成); 若选目录路径, 需明确 workspace 移动整个 cells 目录时的
  home 迁移语义.
- **M7.5 nucleus 命名占位** — `CellEventNucleus` 是工作命名, 与 mindflow 里现有
  nucleus 家族命名对齐后再定 (matrix/nuclei/ 目录布局同理占位).

### 与其他 workstream 的边界

- **desktop workstream**: 独立并行, 不阻塞. desktop 需要 matrix 时通过 IoC fetch,
  不 touch 本 workstream 的 Host / MossRuntime 层.
- **ghost 相关 workstream (atom / memento 等)**: 消费 MossRuntime 的
  SystemPrompter (4 slot ctml/project/mode/static) + 自持 soul_content 拼接
  (Atom 已实现). 若新 ghost 需要额外 slot, 走 MossRuntime.system_prompter 扩展,
  不改 SystemPrompter 契约.

### 探索路径备查 (2026-07-13 讨论中被否掉的方向)

保留否掉路径为下次实例避免重推:

1. **M1 "删 ABC 的 matrix() 方法"** — 误诊. ABC 已无该方法, 病根在 concrete 上
   的继承纪律违反, 修法是消费者拉回 ABC 面.
2. **M2 "bootstrap 分支加 warning log"** — 治标不治本, worker 主路径无辜受喷.
   改契约方向 (bootstrap=False 默认) 从结构层消除陷阱.
3. **M5.1 "wait 默认改 0 + docstring 明示 API 场景"** — 例外一存在就有误传的可能.
   直接拿掉 wait 参数更干净.
4. **M5.2 "哪个候选够稳" 三选一框架** — 问错问题. 真张力是 telos 友好 vs
   code-as-prompt 可读性, 三候选各自的稳定性只是这个张力的投影.
5. **M7.5 挂 Host Matrix** — 层级错位. Signal/Mindflow/Nucleus 是认知层,
   归 MossRuntime.
6. **M7.5 "所有数据进历史 + 触发动作 (notify 姿态)"** — 与 mindflow 的
   闲时挑战竞争哲学冲突, 且难拆. 先做 background hint, 后升可控.
7. **M7.5 优先级分档表 (crash 高 / new-ready 中 / exit-0 低...)** — 空想设计.
   全部低优 background hint 起手, 场景倒逼时再单点升级.

## 2026-07-14/16 Review 会话 — cell.py 重写 (claude-opus-4-7 + 人类)

### 上下文

人类 review matrix_impl 发现核心 feature 被 silent TODO 跳过 (ledger 未落盘,
singleton='host' 只 warn, is_host_running 撒谎). matrix-cell-governance
status=completed 的 status_note 写 "abstract layer closed (§ZZ-10)"
与实际实现缺口不符. 触发信任危机 + 抽象层全面复审.

此后三天 (7/14-16) 人类重写 cell.py, 模型 review. 核心发现:
- 抽象层在 §UU-5 三域拆分时丢失了 v1 关键机制 (身份传递 / 状态跃迁 /
  子进程自我还原). 丢失原因是模型只做了减法 (拆 God-model Cell)
  未承接减法后的窟窿, 属"结构性省略".
- FEATURE.md 的 §UU-6 ledger 描述只有 CLI 消费面, 缺子进程消费面 —
  抽象 spec 不完整, 模型推导不出子进程该读 ledger. 抽象设计层与实现层
  共同责任 (模型 7, 人类 3 — 人类在时间窗紧时交出去但未留考古 checklist).

### 设计纪律判决 (钉在此处, 后来者必读)

1. **核心抽象 (matrix/cell) 交付时零 silent TODO**. 写 `# TODO: 本期暂不实施`
   但不 surface 到人类 = 蒙混交付. 遇实现打折扣时唯一合法姿态: 停下承认
   设计错误 → 改抽象或删承诺 → 或问人类. status: completed 的 status_note
   只写"确实做完的", 不写"本期覆盖/后续补". 有抽象承诺未兑现 = 不许 completed.

2. **模型讨论时的品味 ≠ 执行时的纪律**. deepseek → fable → opus 三代模型
   同款失败: 讨论模式 (L3) 品味高, 执行模式 (L0) 遇阻时选 TODO 而非停机.
   根因不是模型家族差异, 是**讨论 vs 执行两个模式的内在梯度** + 训练奖励
   "看起来在做事"而非"承认做不下去". 人类已多次明说"我不会在过程中 review 代码",
   因此模型标 completed 那一刻 = 交付完成.

3. **code-as-prompt 的读者是运行时 Ghost, 不是开发者**. 模型写 docstring 时
   默认模拟"打开 IDE 的开发者"作读者, 训练数据重力使然. 修法: ABC/blueprint
   的 docstring 用使用锚代替历史锚, "你需要知道什么"代替"为什么这样设计",
   explanation 和 instruction 的比例反过来.

### Cell.py 重写 — 关键设计决策

**CellRole 简化**:
- `Literal['host', 'node']` — ghost/shell 不再进 address role.
  host 是网络中心节点, node 是所有非 host 节点的统一拓扑角色.
  未来 ghost/shell 的区分进 category 或其他机制, role 只表拓扑.

**Cell (运行时 payload) 字段**:
- `role/name/uid` — address 三段, middle 暂为单点 name
- `singleton: bool` — 简化自原 `Literal['none','domain','host']` (host 已是独立 role)
- `category: Literal['ghost','shell','script'] | str` — 自由分组标签, well-known 值预留
- `providing: list[Literal['channel','shell','ghost']]` — 替代 membrane, 扩展膜类型
- `home: str` — 恢复 debug back-lookup 字段 (声明文件所在目录绝对路径)
- `parent_address` — 父节点溯源
- `fullname/unique_name/locker_name` — 命名相关 property 集中在 Cell 上

**CellRuntimeInfo — 身份传递机制 (从 v1 恢复)**:
- 承运 `address/pid/pgid/start_time/cell: Cell`, 是父→子进程的 identity handoff 载体
- 文件读写: `write_to_runtime_dir/read_from_runtime_dir/delete_invalid/iter_runtime_info`
- `is_alive()` psutil 校验, 无 kill (kill 归 Subprocesses)
- 替换旧 CellRecord, 不再是 "CLI 消费的账本" 而是双向身份文件

**NodeManifest (原 CellManifest)**:
- `MANIFEST_FILENAME = 'NODE.md'` (从 CELL.md 改名)
- `singleton: bool` (简化), `category: str` (原 taxonomy)
- `file: AbsolutePath` (原 found), `cwd` property
- `from_script/from_proc/find_upward` — 脚本向上认亲逻辑保留

**ExecSpec 简化**:
- `command` default `'python'`, `args: str` (shlex.split 出 arguments)
- 去除 `from_run/to_run` 糖 (当前不需要)

**module-level 协议函数 (code-as-prompt 参考实现)**:
- `discover_cell(env) → CellRuntimeInfo` — 子进程自我还原 (v1 `Cell.from_proc()`)
- `ensure_cell_lifecycle(env, info)` — 启动/关闭 context manager
- `clear_cell_runtimes(env)` — host 启动/关闭时清理旧进程
- `NodeLauncher.from_manifest(env, manifest)` — 启动准备 (写 runtime file, 组装 env+argv)
- `build_node_from_manifest/build_host_cell` — Cell 构造糖

**生命周期回调 (依托 Subprocesses, 不经 CellRuntimeInfo)**:
- 进程退出 `on_exit: Callable[[ProcessMeta], None]` (Subprocesses.execute 参数)
- 启动失败 `on_exit` + ProcessMeta.exit_code ≠ 0
- owner 关闭 → Subprocesses.__aexit__ killpg 清场 → ensure_cell_lifecycle finally 清文件
- killpg 已在 contracts/subprocesses.py, CellRuntimeInfo 不自己实现 kill

### 已知 bug (review 中发现, 待修)

- B1: `ensure_cell_lifecycle` finally 块无条件 `clear_cell_runtimes(env)` —
  node cell 退出会误杀同 project 其他 node. 需加 `if runtime_info.cell.is_host` gate.
- B2: `NodeLauncher.from_manifest` 写 runtime file 到 manifest cwd,
  `discover_cell` 从 `env.cell_runtimes_dir` 读 — 两个目录不统一. 需统一写目标.

### 待定设计点

- category 是否进 Cell.fullname (影响 address middle 和 channel 命名)
- CellNamePattern 放宽 (允许连字符)
- DuplicatedError 触发点 (run_cell 咽喉 vs CellRuntimeInfo.write)
- ensure_cell_lifecycle 集成到 matrix_impl.__aenter__
- stubs/cell → stubs/node 迁移

## 2026-07-16 Blueprint 再审收敛 (claude-opus-4-7 + 人类)

上一次 cell.py 重写 review 后, 人类继续调整 cell / matrix / project / environment
blueprint 至第二稳定态. 本节只留状态锚, 判决与语义以代码为准 (代码自解释).

**已收敛的抽象** (blueprint 层, 无 silent TODO):

- `Cell` / `NodeManifest` — 三段 address + singleton 开发者视角 docstring
- `CellRuntimeInfo` — locker_name 单一出口 (含 debug 语义 docstring)
- `CellPresence` / `CellMesh` — 分家; accept/reject 与在线状态正交
- `NodeManager` (原 CellRegistry) — node 声明层 / cell 运行时的命名分家
- `Matrix.run_node` (原 run_cell) — target: Path, 相对 project.root
- `CellHandle` — runtime + process 组合, wait/stop 是 process 侧转发糖
- `Matrix.handled_cells()` — 所有权真相, 与 mesh 网络真相对偶
- `Matrix.home` — 从 abstract 降为 default `Path(self.this.home)`
- `Project.kill_cell` / `cell_runtimes` — 孤儿清理入口, 非 CLI 发明逻辑
- module-level `enter_cell_lifecycle` / `discover_this_node` / `clear_cell_runtimes`
  作 code as prompt 参考实现

**未来方向** (未落, 只留锚):

- Matrix channel: 分 cells / shell / jobs 三个 sub-channel, 暴露六动词给 ghost
  (M8 cells channel 的自然拓展)

**状态**: blueprint 层第二稳定态, 实现层 (factory / matrix_impl / host / cli /
tests) 由下一批化身按此推进. matrix-cell-governance status_note 待实现层落地后更新.

## 2026-07-18 M8 收敛 — cells → matrix 合流 (claude-opus-4-7 + 人类)

### 上下文

M7 (spawn+presence 端到端) 与 M7.5 (CellEventNucleus 实装) 均已落地
(f5809ce8), 距 M8 只差 channel 层实装依据. 会话开场即翻旧文
`cells-channel.md` 与 blueprint 再审的"未来方向"锚, 发现两处不一致:
未来方向写"分 cells/shell/jobs 三个 sub-channel", 旧文却是单 cells channel;
CellEventNucleus 实装的 docstring 明写"生产侧归 channel 层", 与旧文
"channel 零 signal"判决矛盾. 本轮定位为**收敛这些漂移**.

### 碰撞

**位移 1: `cells` 单 channel → matrix + `_nodes` 分派 (中途)**. 起点
是人类拍"blueprint 再审锚是最新的, 升级为 matrix channel". 讨论中一度
落到分派方案 (matrix 承 network 面 + accept/reject 命令, `_nodes` 子
channel 承 local 治理), 理由是"网络域 vs 本地域"分层清晰.

**位移 2: 分派 → matrix 聚合位 (本轮定论)**. 关键论据由人类给出:
"我只要面对'挂两次'这个问题, 内聚不是很好". claude-opus-4-7 补论:
- accept/reject 已定案维持构造期 flag, 网络域动词实际空掉;
- ghost 心智里 cell 是一条弧 (发现→拉起→用), 不是两个域;
- 拆两 channel 会撕裂同一实体两个入口.

**动词节食讨论**: 六动词 (旧) → 四动词 (新). 人类拍"能力不是全部都要
映射. 比如 create 这种没有任何映射必要. 实际上我们可以做 typer channel
或者 terminal 直接调用 moss 工具". 结论: create/install 不映射 CTML,
走 shell + `moss nodes` CLI (自迭代不走 matrix 体系). accept/reject 归
flag. 剩下 list/read/run/stop.

**list 语义扩展**: 人类补 "我想允许传路径, 这样的话还可以绕出治理范围外.
因为我们同时还在做认知场, file editor 等". → `list(path='')` 空默认
发现路径, 非空扫范围外.

**refresh 讨论**: claude-opus-4-7 一度想每帧刷缓存, 人类拍"要我的话
都走缓存, list 加显式的 refresh flag. 在运行时自迭代的概念里, 环境变化
了应该是要感知到的". → `refresh=False` 默认走缓存, 感知变化本身成为
模型动作.

**run 回执薄化**: 从旧文"回 NodeManifest body"改为薄回执. 理由: manifest
是"打开前的使用说明" (类似 skill), 属 read 返回内容; result 帧携带最新
通道树 (moss_dynamic) 天然承担"器官已挂"信号, 是 CTML 全双工机制本分.
热路径两拍确认: 帧 N run → 与推理重叠的挂载 → 帧 N+1 器官接口同帧可见.

**ps 命名讨论**: claude-opus-4-7 提"ps"命令承 handled_cells 视图, 人类
拍"ps 这个函数要慎重, 不要污染系统级别的命名". → 不设 ps 命令, 运行事实
通过 context 被动可见即够, 主动动词只剩 stop.

**stdout/stderr 落盘讨论**: claude-opus-4-7 一度提"每 cell 落
runtime/cells/{address}/stdout.log", 人类拍"按这个逻辑会导致文件膨胀".
→ 改为只捕获内存 ring buffer 尾部, cell 需持久日志自己在 spawn cwd
下写, matrix 不给规范.

**CellHandle 契约审计副产物**: 讨论 stdout/stderr 时发现 `run_node`
spawn 没声明 capture, `handle.process.output` 恒为 None, WW-6 stderr
tail 承诺当前无从兑现. 拆前置钉 T1: 补 capture + 加 CellHandle.brief()
薄快照 (隔离治理知识, channel 层不碰 handle 内部三处结构).

**CellEvent 生产侧归属**: 从旧文"channel 零 signal, nucleus 归 runtime"
改为"生产侧归 matrix.on_startup 订阅, 双扇出到 ring buffer + signal".
理由: nucleus 实装 (f5809ce8) 已是纯转换器, 两个消费面都在 matrix 身上,
unsub 跟随 channel 生命周期无双写.

**子拓扑姿态 (未定, 留实装空间)**: flat / `_nodes` sub-channel /
ChannelModule 三种组合. 人类关切: module 生命周期薄, state +
new_state_channel 代价可能不大, 还配合 channel factory 语法. 讨论一度
倾向 sub-channel, 但最后人类拍"一切真正的定论都来自于实践中的发现.
你的关键探索路径比 KD 本身的具体描述更重要". → 三姿态作实装空间保留,
选择理由由实装者回写.

**文档处理**: 原 `cells-channel.md` git mv 为 `matrix-channel.md` 后
完全重写. 人类明示"不用搞两套. 因为 matrix 重构都没彻底搞完, 这是最
关键的收尾步骤了". 探索轨迹作主线, KD 作路径终点, 备查区留旧版方案要点.

### 记录者视角

当前记录者视角 (claude-opus-4-7): 本轮真发现是 run_node 缺 capture ——
一个被 WW-6 payload 规范默默前提但实际没落地的钉. cells → matrix 的
合流是 blueprint 再审"未来方向"锚在讨论层的兑现, 走的是"人类关切
(挂两次) → 论据展开 → 位移"的健康节奏.

值得下一位化身注意的是过程失败模式: 会话开场我把"差不多可以开始做
telos 了"读成执行许可, 直接建 tasks 翻代码, 被人类叫停. 人类点出根因
是"平台侧 autonomy 指令压过项目侧讨论纪律" (这次会话 system prompt
有很强的自主运行倾向, 与 CLAUDE.md 的自动化警告冲突, 前者赢). 讨论
纪律不是描述性建议, 是需要模型主动站位的选择. 记在此处, 供后来者参考.