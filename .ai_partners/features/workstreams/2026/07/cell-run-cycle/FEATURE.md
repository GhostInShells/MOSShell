---
title: Cell Run Cycle
status: in-progress
priority: P0
created: 2026-07-13
updated: 2026-07-13
depends:
  - matrix-cell-governance
milestone: 0.1.0
description: >-
  Cell-centric acceptance cycle after matrix-cell-governance closure —
  纪律修正 (M1/M2) + wire-up 拉齐 (M3/M4) + telos 主路径
  (M7 spawn → M7.5 CellEvent 信号链 → M8 六动词 channel → M8.5 L1 tutorial 重构
  → M9 自迭代 telos 真验证).
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
M7 → M7.5 → M8 → M8.5 → M9 是 telos 主路径 (依次搭建).
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

### M8. 决定第一个 cells channel — moss cells channel

**判据**: UU-9 moss_self 合流的第一个 channel 落地. 六动词 (create/install /
run/stop / accept/deny) 经 CTML 层投影, 模型可通过 CTML 操作 cell 治理.

**设计已定案 (2026-07-13 讨论), 详见本目录 `cells-channel.md`**. 结论速览:
- 拓扑: 单 `cells` channel, 六动词 nonblocking own commands 在 top,
  proxy 走 virtual children (refresh_meta + get_virtual_children 缓存模式).
  治理子 channel 方案否掉 (nonblocking 消解漏斗动机).
- foreign 挂载: 构造期 `auto_accept` flag; local 永远自动挂 (UU-7);
  flag 开时 accept/release 命令不注册.
- 信息三分 (instruction / context_messages / 命令返回) 与 always_observe
  分档见子文档 §4/§5.
- channel 零 signal — CellEvent → Signal 独占归 M7.5 CellEventNucleus.
- **UU-9 纠偏**: channel 手写绑 Matrix, 不是 typer 反射 CLI —
  原判决模型不了解 channel 实现, run/stop/accept 必须 in-process.

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
