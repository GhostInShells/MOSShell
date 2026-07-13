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
  app→cell migration, wire-up refinement, MCP end-to-end, first cells channel,
  UU-11 self-iteration telos validation.
---

# Cell Run Cycle

> matrix-cell-governance 抽象层已闭合 (§ZZ-10 收官, wire-up 通过 moss-as-mcp 说话
> 验证). 本 workstream 是它的下游: 让 cell 通过重生的 matrix 真正跑起来,
> app 逐个迁移为 cell, 直到 UU-11 telos "身体在运行时长出新器官" 的第一个真验证.

## Motivation

**为什么单独开一个 workstream 而不是留在 matrix-cell-governance**:

1. matrix-cell-governance FEATURE.md 已 6 章 (§TT/UU/VV/WW/XX/YY/ZZ),
   继续追加 wire-up 收尾治理会稀释其"抽象决策载体"的定位.
   §WW-0 教训: 判决类结论钉在正文, 但那是**抽象层判决**. 下游治理是不同思维模式.
2. **关注点不同**: 上游 = 设计-推翻-合流 (思维模式偏形而上);
   下游 = 面向下游用户 (人类 + 模型) 的验收纪律 (思维模式偏工程 + 观测).
3. **新化身接力成本**: 单独 feature 只需读一份 (依赖 pointer 到 matrix-cell-governance),
   不用消化 6 章设计辩论即可开工.
4. 本轮实际焦点 = **app 重构 + matrix 治理, 重点围绕 cell** — cell 是主语,
   run cycle (create → run → 入网 → interface 进帧) 是它的验收单元.

## 依赖必读

- **matrix-cell-governance** (`.ai_partners/features/workstreams/2026/06/matrix-cell-governance/FEATURE.md`)
  - **§UU 全文** (抽象闭合总纲, 十个判决包)
  - **§YY** (Matrix/Project 表面积定稿 + session 永在首页 + home 双目录判决)
  - **§ZZ** (实现层设计对齐, 9 subsections)
  - **§ZZ-10** (TT-2 address 三段结构终审 — 本 workstream 一切 discovery / URL 语义地基)
- **代码入口**:
  - `src/ghoshell_moss/core/blueprint/cell.py` (三域模型 + address helpers)
  - `src/ghoshell_moss/core/blueprint/matrix.py` (Matrix ABC + 表面积)
  - `src/ghoshell_moss/matrix/matrix_impl.py` (MatrixImpl 组装)
  - `src/ghoshell_moss/matrix/adapter.py` (MatrixNetworkAdapter ABC)
  - `src/ghoshell_moss/matrix/networks/zenoh_adapter.py` (zenoh driver 实现)
  - `src/ghoshell_moss/host/impl.py` (Host concrete, wire-up 后的形态)
  - `src/ghoshell_moss/host/moss_runtime.py` (MossRuntimeImpl, mode-aware runtime)

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
- Host 抽象重绘未完 (Host.matrix() 方法冗余, 参见 M1)
- 剩余 3 个 CLI 入口 (ghost_run / moss_as_fractal / cli_controller) 未对齐 seal 姿态
- TUI 里的 Manifests / Fractal inspectors 被砍掉 (dead API), 待重画或永久删除
- P0 决策未拍板: run_cell.wait 默认值; home 稳定身份键 (dir/name/UUID)
- Layer 1 cross-scope host 发现推迟 (本期 Watcher 只 subscribe 自己 scope)

## Milestones

按依赖倒序 (后依赖前), 每 milestone 尽量独立 commit:

### M1. Host 抽象重绘 — Host.matrix() 方法删除

**判据**: `MossHost` ABC 无 matrix() 方法, moss-as-mcp / moss-cli 等消费者从
`host.run() -> moss_runtime.matrix` 访问 matrix. `host/matrix.py` 老文件 `git rm`.

**理由**: `Host.matrix()` 是 wire-up 阶段的兼容 shim, MossRuntime 已 direct-through
matrix (blueprint/host.py L200-226). Host 收敛为只做 "环境发现 + 编排 run/run_ghost".

**关联**:
- moss-as-mcp L114: `moss_host.matrix().logger.info(...)` → `state.toolset.matrix.logger.info(...)`
- host/tui.py 若有 host.matrix() 用点, 一并迁 moss_runtime.matrix (但注意 TUI 拿 matrix 早于 run)
- host/tui_entries/moss_runtime_ui.py L30: 用 MossRuntime 而非 Host 拿 matrix

### M2. Environment.discover(bootstrap=True) 加 loud warning

**判据**: 走 bootstrap 分支 (无 singleton 时) log warning: "No sealed Environment
singleton found; auto-bootstrapping. If you meant to inject CLI params,
entry-point should call Environment(**cli_args).seal() first."

**理由**: §UU-1 seal 姿态落地后新引入的静默陷阱 — 入口忘记 seal 导致 CLI
参数无声丢失. Worker cell 场景合法 (父 export os.environ 兜底), 故不阻断;
但需可观测. 详见 review W4.

### M3. 剩余 3 个 CLI 入口 seal 姿态对齐

**判据**: `cli/ghost_run.py` / `cli/moss_as_fractal.py` / `cli/cli_controller.py`
从 "discover 后 set_mode" 改为 "构造 Environment(**cli_args) + seal".
`Environment.set_mode / set_ghost_name / set_network_scope` 已在 commit 4c75f76b
删除, 现有调用点全部触发 AttributeError, 是本 workstream 的 P1 blocker.

**关联**:
- host/tui_entries/ghost_ui.py L152 的 stale code-as-prompt phrasing
- ghosts/atom/_meta.py L76 的 docstring 更新 ("由 Host.run_ghost() 通过 env.set_ghost_name()
  确保..." 需改)

### M4. TUI Inspectors 重画或永久删除

**判据**: `Manifests` inspector 与新 MatrixManifest+ModeManifests API 对齐,
`Fractal` inspector 或迁到新拓扑或永久 remove (fractal §TT-14 已弃).
决策拍板: 重画 vs 永久删除.

**上下文**: 本 workstream 承接时二者已从 tui_entries/moss_runtime_ui.py 砍掉 (dead API
无法 import), 保 TUI 起来. 重画归本 milestone.

### M5. P0 决策拍板 — run_cell.wait 默认 + home 稳定身份键

**M5.1 run_cell.wait 默认值**:
- 现: `wait: float = 30.0`
- 冲突: WW-5 故事 4/5 "channel 面不 wait, 模型面走 signal"
  — 30 秒默认在 CTML channel 反射调用时会挂模型 30 秒
- 建议: 默认改 `wait=0`, docstring 加 "wait > 0 仅 Python API bootstrap 场景,
  CTML channel 反射调用必须显式 wait=0 走 mindflow signal"

**M5.2 home 稳定身份键**:
- 现: `{workspace}/cells/{normalize(manifest.name)}`
- 塌陷风险: manifest.name 是 CELL.md frontmatter 字段, 作者可改, 改完重启
  home 找不回旧数据. systemd unit name = 文件名 (改就是重装), 不同构.
- 三个候选:
  - (a) CELL.md 目录相对路径 (改 name 不改路径; 移文件即视为新 cell)
  - (b) `.cell_id` UUID sidecar (systemd machine-id 同构; 稳但要开新文件约定)
  - (c) manifest.name (现况; 塌陷风险已知)
- **人类拍板必须**: TT-2 身份终审同盘.

### M6. CTML 契约完整性护栏 (W6 已改)

**判据**: `_load_ctml_prompt` 崩溃姿态强化 (KeyError → RuntimeError, 覆盖
version 缺失 / file 读失败 / content 空三种 fatal 路径), 单测锚 (无 CTML
version 时 MossRuntime 无法构造).

**状态**: 已实施 (moss_runtime.py L142-176), 待补单测. 详见 review W6.

### M7. moss-as-mcp 端到端跑一次真 cell (非语音验收)

**判据**: 通过 `execute_ctml` 发一个 `<cells:run target="foo"/>` 或等价 CTML,
MCP 侧观察到 cell 入网 announce, watcher.view() 能看到. 语音 (`__content__`)
之外的膜类型验收.

**为什么**: 语音验证只走了 `__content__` → TTS 一条通路. run_cell 咽喉是 cell
体系核心, 需要独立验证 (咽喉六步全跑通).

### M8. 决定第一个 cells channel — moss channel

**判据**: UU-9 moss_self 合流的第一个 channel 落地. 六动词 (create/install /
run/stop / accept/deny) 经 CTML 层投影, 模型可通过 CTML 操作 cell 治理.

**待决**:
- channel 命名 (moss / moss.cells / etc.)
- 六动词的具体命令签名 (对应 UU-4 治理代数)
- state 分组 (M6 若 CTML 视图更多, 分层拓扑)

### M9. UU-11 自迭代 telos 第一次真验证

**判据**: 一个 cell 通过 M8 的 cells channel 被创建 + 拉起 + accept 后,
其 channel 接口 (instruction + commands) 自动出现在下一帧模型上下文中.
模型能读到新器官, 能对新器官发命令.

**判决点**: 这是 MOSS 与 bash 本质区别的第一次运行时实证. 若通过,
matrix-cell-governance + cell-run-cycle 两个 workstream 合流达成最初 telos.

### M10. Layer 1 cross-scope host 发现 (方案 α)

**推迟到本 workstream 但明确**: ZenohWatcher 加第二个 subscription
`MOSS/matrix/scopes/*/cells/host/**` 收 cross-scope host liveness.
副路径 `hosts_ns` 代码同一 PR 清 (已 §ZZ-10 语义作废).

**理由**: 本轮 §ZZ-10 落地时的漏项 — 副路径作废但无替代实现补 cross-scope
通道. wire-up 期无 blocking (Layer 2 scope 内发现足够 mcp 说话验证),
但 telos 完整闭合前应补齐.

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
下游依赖, 不重复其判决. 若下游发现抽象层需修 (M5 两条即此类), 修改在
matrix-cell-governance 的 §AAA (若开) 或直接文件层 patch 后钉在本 FEATURE.md
的 Implementation Notes.

### CRC-2. cell 是主语, run cycle 是验收单元

本 workstream 一切 milestone 服务于 "cell create → run → 入网 → interface 进帧"
一个循环. 每 milestone 问一句: 它让这个循环的哪一步更接近闭合? 无答者砍。

### CRC-3. M5 两条 (P0) 必须与人类对齐

run_cell.wait 默认值 + home 稳定身份键属抽象层修正, 不可自行推进. 其余
milestone 模型可自主执行, 遇到抽象层未覆盖歧义时按 §VV-1 挡板① 停下问.

### CRC-4. M9 是终局判据

自迭代 telos 通过前不 v0.1. M9 通过后, apps → cells 大规模迁移可开另一
workstream.

## Implementation Notes

### 已知漂移点

- `Host.matrix()` 消费点分散 (moss-as-mcp / tui.py / tui_entries/moss_runtime_ui.py),
  M1 重绘时全部改到 `moss_runtime.matrix`. grep `.matrix()` 找全。
- `env.set_mode / set_ghost_name / set_network_scope` 死方法调用点: `cli/ghost_run.py` /
  `cli/moss_as_fractal.py` / `cli/cli_controller.py` (grep 已定位).
- `host/tui_entries/ghost_ui.py` L152 + `ghosts/atom/_meta.py` L76 的 stale docstring
  提到 `env.set_ghost_name`, 需同 M3 一起更新.

### 谨慎点

- Host 重绘 (M1) 不要引入 "MossHost.matrix_lazy_hint" 之类的兼容 shim, 直接删
  matrix() 方法, 消费者迁移 (只有 <5 个调用点).
- M4 TUI 重画时警惕不要复原 Manifests god-basket API — Manifest.is_error() 显性化
  是新契约的核心. 重画应 walk MatrixManifest / ModeManifests 直接展开每个 Manifest,
  不重塑 dict 视图.
- M5.2 若选 UUID sidecar 姿态, 需处理老 workspace 迁移 (无 `.cell_id` 的 cell
  首次运行自动生成); 若选目录路径, 需明确 workspace 移动整个 cells 目录时的
  home 迁移语义.

### 与其他 workstream 的边界

- **desktop workstream**: 独立并行, 不阻塞. desktop 需要 matrix 时通过 IoC fetch,
  不 touch 本 workstream 的 Host / MossRuntime 层.
- **ghost 相关 workstream (atom / memento 等)**: 消费 MossRuntime 的
  SystemPrompter (4 slot ctml/project/mode/static) + 自持 soul_content 拼接
  (Atom 已实现). 若新 ghost 需要额外 slot, 走 MossRuntime.system_prompter 扩展,
  不改 SystemPrompter 契约.
