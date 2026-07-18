# Matrix Channel — M8 探索轨迹与当前定论

2026-07-13 首轮定案 (cells-channel.md, claude-fable-5 + 人类)。
2026-07-18 推翻重写 (本文件, claude-opus-4-7 + 人类)。

本文档以 **探索路径为主线**，KD 作路径终点标记。文档的价值不在具体 KD，
而在留给下一位化身"我们如何走到这里、否掉过什么、为什么"的完整轨迹。
一切真正的定论来自实装中的发现——本文档的所有 KD 均待 M8 实装验证。

上游判决引用 matrix-cell-governance FEATURE.md (§UU-4/UU-7/UU-8/WW-5/WW-6/WW-7)。

## 0. 探索路径主线 (最重要的上下文)

**位移 1: UU-9 纠偏 (2026-07-13)**

UU-9 原文"moss_self 反射 CLI 成 channel，两个面免费获得"是不了解 channel
实现的判决。cells channel 必须**手写、in-process、绑 Matrix**。理由:
- run/stop 的 owner 必须是 host 进程 (WW-6 dead 信号源 = Subprocesses done
  callback，只对 owner 生效)；shell out 到 CLI 会让 owner 变成瞬态进程。
- accept/release 是 Watcher 进程内状态 (UU-8)。
- 只有 create/install (纯文件动词) 可与 CLI 共享底层函数。

**位移 2: cells 单 channel + 六动词 (2026-07-13, 已推翻)**

初版方案: 一个 `cells` channel，六动词 (run/stop/create/install/list/status
+ accept/release 按 flag) 全 nonblocking，virtual children = 已挂载 proxies。
详见 §A 备查区。

**位移 3: matrix 与 cells 分派 (2026-07-18 中途, 未落地)**

讨论中一度倾向: matrix 父 channel 承 mesh (network 面 + accept/reject 命令
+ mesh view context)，`_nodes` 子 channel 承 local 治理 (list/read/run/stop
+ category 计数 context)。理由是"网络域 vs 本地域"分层清晰。

**位移 4 (当前定论): matrix 作 cell 治理的聚合位 (2026-07-18)**

推翻分派方案的关键论据:
- accept/reject 已定案维持构造期 flag (不作 command)，"网络域动词"实际空掉。
- ghost 心智里 cell 是一个概念，不是两个——"发现→拉起→用"是一条弧，跨
  channel 会撕裂 (`<cells:run camera/>` 之后 `<matrix.camera:capture/>`
  是同一实体两个入口)。
- 人类关切"挂两次"内聚问题——单聚合位消解之。

**当前姿态**: matrix 作聚合位这个方向钉住；具体挂载拓扑 (flat own commands
vs `_nodes` sub-channel vs ChannelModule) 是**实装空间**，不预定，由 M8.5
tutorial 走通时倒逼。

## 1. 当前定论 (路径终点标记)

### 1.1 matrix channel 聚合形态

- 单一根: `matrix` (main.import_channels 一次挂载)。
- 承 cell 治理全部对外面: own commands + virtual children (proxies) + 治理
  context + CellEvent 生产侧。
- 器官路径: `matrix.<cell.fullname>:command` — 前缀成本 (~2 tokens
  "matrix.") 是常数税，换 code-as-prompt 语义清晰 (器官是网络资源)。
- 子拓扑姿态待实装决: **flat** (所有治理动词直接挂 matrix own_commands) 或
  **`_nodes` sub-channel** (list/read/run/stop 聚 `_nodes`, matrix 只留网络
  面) 或 **ChannelModule 组装** (三者内聚度不同，实装体感为准)。

### 1.2 own commands (四动词，全 nonblocking)

| 命令 | 签名 | 返回 | always_observe |
|---|---|---|---|
| `list` | `(path='', category='', installed=None, refresh=False)` | 表: relative_path/name/category/installed/description | True |
| `read` | `(target)` | NodeManifest 全量 (含 metadata) | True |
| `run` | `(target)` | 薄回执: address+pid+"器官将在后续帧自现" | True |
| `stop` | `(address)` | 薄 ack | False |

- **list 允许传 path**: 空 = 默认发现路径 (project.nodes_discover_paths)；
  给路径 (含绝对) = 扫治理范围外——为认知场 / file editor 留接口。
- **refresh 默认 False**: startup 扫一次建缓存，之后模型显式 `refresh=True`
  感知环境变化——自迭代语境下"感知变化"本身成为模型动作。
- **run 回薄回执**: NodeManifest 是打开前的使用说明 (read 拿), body 不进
  run 回执。热路径两拍: `run` (帧 N nonblocking) → announce+mount (与推理
  重叠) → 帧 N+1 result 抵达同帧器官已挂树、interface 可见 → 直接用。
- **create/install/accept/reject 不映射 CTML 动词**: 前两者走 shell channel
  调 `moss nodes` CLI (自迭代不走 matrix 体系)；后两者归构造期 flag。
- **run 护栏**: per-target in-flight dedup (§7 防蠢，从旧版继承)。

### 1.3 context_messages (每帧, 全走缓存)

数据源纪律 (WW-7): Subprocesses 内存句柄 (handled_cells) + mesh 视图 join，
**永不读 ledger**。每帧内容:

- inventory 概要: `本地 N nodes — tools:6, sensors:4, scripts:3, 未分类:1,
  未安装:2` (category 计数是模型的检索词汇表)
- 运行中 cell: 每个 handled_cell 的 `handle.brief()` (见 §1.5)
- 最近退出: dead_cells 尾部 (address / exit_code / stderr tail 路径提示)
- mesh 概要: `网络上 X cells (accepted Y)` 一行
- 最近事件: CellEvent ring buffer 尾部 3-5 条

### 1.4 virtual children = mesh.channel_proxies() 快照

- mesh 已是挂载真相源: accept 表 + `_should_build_proxy` + `channel_proxies()`
  在 zenoh_mesh 实现层完成。
- channel 侧薄镜像: `refresh_meta` (async) 拷 `mesh.channel_proxies()` 快照
  进缓存；`get_virtual_children` (sync) 返回缓存。
- 实例稳定由 mesh 的"同 address 同 proxy"保证，channel 不自持挂载状态。

### 1.5 CellHandle.brief() 薄快照 (前置钉)

**动机**: channel 消费 CellHandle 时需要 address/alive/pid/uptime/日志线索
/stderr 尾部的组合视图。当前散在 `handle.runtime` + `handle.process.meta`
+ `handle.process.output` 三处，channel 自拼会把治理知识泄漏进 channel。

**契约**: CellHandle 上加 `brief() -> CellBrief` (dataclass)，字段:
address / role_name / alive / pid / exit_code / uptime_seconds /
stderr_tail (从 ProcessOutput 内存 ring buffer 取，缺省 N 行) /
instance_dir (spawn cwd，well-known 根供 ghost 自己 shell/file-editor 探索)。

`wait` / `add_done_callback` 等阻塞或回调型接口 channel 一概不碰——
wait 违反 nonblocking，callback 是 matrix 内务。

### 1.6 CellEvent → context / signal 生命周期绑定

- matrix.on_startup 一次 `mesh.on_event` 订阅，两个扇出:
  (a) 写 channel 自持 ring buffer (喂 context_messages)；
  (b) `CommandUtil.send_signal(CellEventSignal)` → CellEventNucleus 转
      Impulse(background_notice)。
- matrix.on_close unsub。
- 推翻旧判决"channel 零 signal": CellEventNucleus 实装 (2026-07-18) 已明写
  "生产侧归 channel 层"，本轮拍板 = matrix channel。一份订阅无双写，
  unsub 跟 channel 生命周期干净。

### 1.7 stdout/stderr 捕获约定

**前置钉 (实装未落)**: `run_node` spawn 目前**未声明 capture** (matrix_impl.py
run_node)，导致 `handle.process.output = None`，WW-6 stderr 尾部承诺无从
兑现。

**方案**: 
- 只捕获**内存 ring buffer 尾部** (bounded, ~32KB/stream)，channel 不给
  matrix 侧强制落盘。
- cell 需要持久日志，自行在 `instance_cwd/` (= spawn cwd = handle.runtime
  已知路径) 下写文件——matrix 承诺该目录为 well-known 根，ghost 可通过
  shell/file editor 浏览，但不做规范。
- context 里出现的是 stderr **tail 内容**本身或短提示，不是"我们给你落盘了"。

### 1.8 instruction 分层 (机器人 ghost 友好)

- **interface** (code as prompt): 命令签名 + docstring 承机制细节。
- **instruction** (静态): 三句话级别，只讲概念。参考措辞: "这里是你领地内
  可生长的器官清单。run 之后器官会出现在你的通道树里，不需要等待。
  异常会在上下文中显现。" **不出现路径、文件名、CLI 命令**。
- **未安装 / crash 等具体线索走 error message 与 context**，不预载 instruction。

## 2. 否掉的路径 (备查，防复推)

按位移顺序列出，含推翻理由。这一节是探索轨迹的骨架，比 §1 更重要。

### 2.1 UU-9 "moss_self 反射 CLI 成 channel" (2026-07-13 否)
理由: run/stop 需 in-process owner；accept/release 需 Watcher 进程内状态。
CLI 反射方案只在 create/install 文件动词上成立。

### 2.2 cells 单 channel + 六动词 (2026-07-18 否)
理由: create/install 不必映射为 CTML 动词——机器人 ghost 场景不需要，
coding agent 场景走 shell+CLI 更自然 (`moss nodes` 已完备)。accept/release
归构造期 flag。剩下的四动词 (list/read/run/stop) 无需 cells 独立命名空间，
可直接聚 matrix。

### 2.3 治理子 channel (`cells.manage` / `cells._discover`) (2026-07-13 否)
理由: 立子 channel 的唯一结构动机是漏斗 (怕 blocking 治理动词阻塞器官)；
nonblocking 后动机消失。

### 2.4 `_` 前缀命名 (2026-07-13 弱否)
理由: 训练分布中读作"内部勿碰"，主动词每次调用有认知摩擦。若未来 sub
channel 复活，`_` 前缀作实装期取舍空间保留 (不需要动 CellNamePattern，
cell 名与 sub-channel 名在不同命名面)。

### 2.5 proxy 挂 main channel 顶层 (2026-07-13 否)
理由: 误解。proxy 从 matrix 的 get_virtual_children 返回，CTML 树状寻址；
main 顶层挂 proxy 引入与静态 channel (speech/shell) 的命名保留字规则。

### 2.6 matrix (network) + `_nodes` (local) 分派 (2026-07-18 否)
理由: 见 §0 位移 4。accept/reject 归 flag 后网络域动词空掉；ghost 心智
里 cell 是一条弧不是两个域；"挂两次"内聚问题。

### 2.7 ChannelModule 三分 (mesh/processes/jobs) (2026-07-18 转为实装空间)
起因: 想用 ChannelModule 承 mesh 支持，processes/jobs 同构复用。人类
关切: module 生命周期薄，state + new_state_channel 代价可能更小，且带
factory 语法免费。**结论**: 不预定，作为实装期的组装姿态之一保留。

### 2.8 run 回 NodeManifest body (2026-07-13 → 2026-07-18 否)
理由: NodeManifest 是"打开前的使用说明" (类似 skill 描述), 属于 read
的返回内容。run 回执保持薄，让 result 帧携带最新通道树 (moss_dynamic)
承担"器官已挂"的信号——CTML 全双工机制的本分。

### 2.9 channel 零 signal (2026-07-13 → 2026-07-18 否)
理由: CellEventNucleus 实装 (2026-07-18) 明写"生产侧归 channel 层"。
matrix.on_startup 订阅 mesh.on_event 双扇出比 MossRuntime 层生产更内聚:
两个消费面 (ring buffer + signal) 都在 channel 身上，unsub 跟随生命周期。

### 2.10 ps 命令 (2026-07-18 否)
理由: "ps" 是系统级命名，channel 命令表面不该占用它 (会污染模型对 shell
体系的直觉)。运行事实通过 context 被动可见即够，主动动词只剩 stop。

### 2.11 stdout/stderr 强制落盘 (2026-07-18 否)
理由: 每 cell 强制文件会导致膨胀。改为只捕获内存 ring buffer 尾部；
cell 自需持久日志自己在 instance_dir 下写 (§1.7)。

## 3. Tasks

**执行前提**: 本文档所有 KD 均待实装验证。Tasks 判据留活口——按当前定论
起步，实装中发现问题即回讨论修正 KD (走 features 纪律，不 silent todo)。

### T1. CellHandle.brief() 契约与 run_node capture 补齐

**判据**: CellHandle 上加 `brief() -> CellBrief` (§1.5 字段)；`run_node`
spawn 时声明 capture (内存 ring buffer, bounded)，`handle.process.output`
非空。matrix channel 消费 brief() 无需碰 handle.runtime / process.meta /
output 三处内部结构。

### T2. matrix channel 实装

**判据**: 按 §1 实装 (建议 `src/ghoshell_moss/channels/matrix_channel.py`
或近似路径)。四动词 own commands 全 nonblocking；virtual children 镜像
`mesh.channel_proxies()`；context_messages 按 §1.3 缓存；on_startup/on_close
绑 mesh.on_event 双扇出；per-target run dedup 护栏。子拓扑姿态 (flat /
`_nodes` / module) 实装者按体感选，把选择理由回写 §0。

### T3. 单测

**判据**: 脱 ghost/mindflow 可测。覆盖:
- 四动词 nonblocking 并发下发不阻塞；
- run(target) burst 只落一次 spawn；
- virtual children 跨 refresh 稳定 (同 address 同实例)；
- context_messages 不触 ledger；
- CellEvent → ring buffer + signal 双扇出各一次；
- CellHandle.brief() 字段覆盖 alive/dead 两态。

### T4. 接线与验收

**判据**: mode-level channels 装配处注册 matrix channel；`moss-as-mcp` 下
`<matrix:run target=.../>` (或 `<matrix._nodes:run/>` 视 T2 拓扑决定) 可
发；spawn 后下下帧器官 `matrix.<name>:<command>` 可调 (M7 端到端验收载体)。

## A. 备查区 — 旧版方案要点 (2026-07-13 cells 单 channel)

保留供追溯，勿据此实装。

- 拓扑: 单 cells top + own commands 六动词 (run/stop/create/install/list/status
  + auto_accept 关时的 accept/release) + virtual children (已挂 proxies)。
- 数据源纪律 (仍适用): context_messages = Subprocesses 内存句柄 + Watcher
  视图 join，永不读 ledger。
- always_observe 分档 (仍适用): run/list/status/accept True；
  stop/release/install/create 默认。
- 并发护栏 §7 (仍适用): per-target in-flight dedup，dup 返回不带 observe
  权重，不用 raise_observe。
- 完整轨迹见 git log: `git log --follow -- .ai_partners/features/workstreams/2026/07/cell-run-cycle/matrix-channel.md`

