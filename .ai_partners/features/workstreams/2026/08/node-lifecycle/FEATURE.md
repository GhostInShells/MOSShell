---
created: 2026-08-04
depends: []
description: Node 生命周期治理，从 node-migration 独立。四层方案已收敛为 node 最佳实践探索： 四地址发现、一次性 node、事件分级已落地；重新论证后决定
  drop node id、做 probe 预启动闸门。收尾中。
milestone: 0.1.0
priority: P1
status: completed
status_note: '启动面治理落地: parallel bringup + probe timeout + bringup failure event; smoke
  验证通过'
title: Node Lifecycle — 身份、入口、验证与记忆
updated: '2026-09-13'
---

# Node Lifecycle

> 人类架构师 + claude-opus-4-7 + deepseek 家族。node 生命周期治理 workstream。

## Motivation

node 的就绪状态没有进入管理：`.installed` marker 只回答"装没装过"，不回答"环境现在
能不能用"；启动失败只在 stderr 和 bounded FIFO 里，不进模型上下文。需要一个覆盖生命
周期的治理链。早期方案是"身份 → 入口 → 验证 → 记忆"四层，随后演进为按需生长的
最佳实践探索（见 Compaction Note）。

## Compaction Note (2026-08-14)

历史决策 1–10.x 与三轮调研增补（启动成本实测、事件分级、一次性 node）已折叠，详细
轨迹在 git log：

```
git log -- .ai_partners/features/workstreams/2026/08/node-lifecycle/FEATURE.md
```

| commit | 主题 |
|---|---|
| `9f4ecbd9` | 初始四层治理：identity / entry / probe / ghost memory |
| `9c636408` | 启动成本调研 + 决策 5–8（砍 zenoh 否、anthropic import、事件分级）|
| `98c2cf5a` | 事件分级 + MatrixOperator 方向（决策 9）|
| `19a42de8` | 一次性 node 角色 + event_level gating（决策 10.x）|
| `0947b0cd` | $GHOST/$MODE 四地址发现前缀；记录当前共识 |

## Landed（代码即真相，2026-08-29 核实）

以下均已实现且经核查与当前代码一致：

| 能力 | 落点 |
|------|------|
| **四地址发现前缀** | `resolve_node_dir()`（environment.py:173）；`MossMeta.node_paths`/`HostModeMeta.node_paths` 默认四组合；`Environment.ghost_home`/`mode_home` |
| **Matrix.new 默认 persist=False** | matrix.py:128；一次性脚本节点不声明 singleton（matrix.py:141-142）|
| **event_level 五档 + persist** | `CellEventLevel`（DEBUG..CRITICAL，对齐 logging 层级）；一次性 node→DEBUG 静默，常驻→INFO 感知（cell.py:867-870）|
| **一次性 node 角色** | `NodeManifest.persist`（default True）；persist=False → run-to-completion 阻塞拿 stdout/stderr/exitcode |

配合层 **matrix-manifest-layers（三层 manifest 声明隔离）** 已"implementation 完成"
（HostMode ABC + LocalHostMode + MatrixImpl._prepare_container MATRIX wiring + mode stubs +
manifests CLI explain 三层展示）。音频 provider 搬迁是后续独立 feature，不在本 workstream。

## 重新论证（2026-08-29）— drop node id，做 probe

原四层方案的 layer 4（Ghost 记忆）已因"决策 4 纠正"取消——node 级记忆落点是 skills
声明式约定 + ground 认知，`NodesMemoryContract` 移除。据此重新论证两个候选：

### node id（`.node_id` UUID）— 已 drop

**弃掉的理由比原判断更强**：`.node_id` 的唯一消费者是 `NodesMemoryContract`（按
node_uuid 键控 ghost 记忆）。决策 4 移除该契约后，node_id 失去了键控对象。剩下的
"跨目录重命名/搬机器不丢身份"价值，已被现有 `Cell.uid`（per-spawn unique_id）+
`CellAddress` + `project_id`（治理域）覆盖。`.node_id` 会成为平行第二身份源，是重复
而非补充。**结论：drop，干净减法。**

### probe（启动前闸口）— 已落地

这是原四层里唯一仍直击 Motivation 核心的一项：把"静默的启动失败"（只在 stderr 和
bounded FIFO 里）变成"拉起前闸门 + 明确 broken reason"，让"环境现在能不能用"进入
模型上下文。设计要点：

| 设计点 | 内容 |
|--------|------|
| **形态** | NODE.md 可选声明 `check: {command, args}`（复用 ExecSpec），或约定 `check.py`（今回先做显式声明，约定式预留）|
| **独立性** | 独立进程、语言无关、目标脚本**零配合**（不逼对方走到 `Matrix.__aenter__`）|
| **闸门** | exit 0 → 通过；nonzero + stderr → 返回 broken reason，**不拉起主脚本** |
| **语义纪律** | 只用 exit code，**不发明新 ready 状态机**（吸取 matrix_impl.py:174 上一版"猜 provider ready 信号"翻车教训）|
| **不加新字段** | 主脚本拉起后靠既有 `process alive + ledger providing` 兜底，不加 CellRuntimeInfo 字段 |

挂载点：probe 收敛在 `NodeManager.spawn_node` 内（唯一 spawn 咽喉），CLI `run`/matrix
`run_node`/CTML `nodes:run` 都经由它触发，不再各挂一次。

stdout 本次不落地（闸门只用 exit code + stderr 作 broken reason）；"probe stdout 作为动态
self-description 进模型认知窗口"是后续评估项，本 workstream 先做闸门主体。

## 启动面治理（2026-09-12 重开）

feature 此前判 completed；人类回忆起启动 node 的失败面与启动阻塞当时就在意图内、但未落
进本文件。核实代码后确认三项缺口 + 一条闸口纪律。

### 缺口

| 缺口 | 现状 | 位置 |
|---|---|---|
| **bringup 阻塞启动** | `_bringup_nodes` 在 `__aenter__` 内串行 `await`，probe 挂死即永久阻塞，shell 起不来 | `moss_runtime.py:249-260`、调用点 :442 |
| **probe 无超时** | `_run_probe` 走 `_await_exit` 等子进程退出，无上界；挂死探针泄漏且钉住 bringup | `node_manager.py:277,299` |
| **失败无感知** | spawn 前失败（resolve/未安装/probe/singleton）只进 host 日志；spawn 后硬挂（SIGKILL/segfault）父侧 `_on_cell_exit` 也不 publish，**两类网络上都无痕** | `node_manager.py:181-204`、`matrix_impl.py:371-390` |

失败无感知一项与既有 Open Question（publish_event 级别锁死）是同一个洞的两面。

### 闸口结论（已核实，2026-09-12）

spawn 面**单喉唯一** = `NodeManager.spawn_node`（`node_manager.py:159`）。全仓两个 caller：

| caller | 场景 | 有无 matrix |
|---|---|---|
| `matrix.run_node`（`matrix_impl.py:346`） | bringup + CTML `matrix.run` 命令（`matrix_channel.py:266`） | 有 |
| CLI `moss nodes run`（`nodes_cli.py:442`） | 前台自持 | 无 |

二者取的是**同一 NodeManager 实例**（matrix 的 container 即 project 的 container，
`matrix_impl.py:96→594`；`local_project.py:72` 在此 container 上 set NodeManager；
`project.nodes` 亦为同一 force_fetch）。**所以预检只在一处，不存在多位置重复。**

关键区分：**publish 不能兜在 spawn_node**。`spawn_node` 是 project 级组件，只有
`env/subprocesses/logger`，没有 matrix/presence，物理上 publish 不出去
（`MatrixImpl.publish_event` 要求 `_presence` 已 enter，`zenoh_presence.py:165-169`）。
硬塞进去等于发不出去 + 逼 matrix 层再发一次 = 真正的"两处做同一件事"。

**收敛为"一层一件事"**：

| 层 | 职责 | 位置 |
|---|---|---|
| spawn 闸口（预检 + exec） | 唯一，CLI 也走 | `NodeManager.spawn_node` |
| 父侧治理 + 通知（handle 登记 / done callback / publish） | 唯一，仅在有 matrix 时 | `matrix.run_node` + `_on_cell_exit` |
| 无 matrix 的 owner | 前台阻塞，通知 = 终端 returncode | CLI `nodes run` |

**纪律**：有 live matrix 时 spawn 必须经 `matrix.run_node`，不得直接
`project.nodes.spawn_node`（会绕过 `_handled_cells` 登记与 done callback，父侧拿不到任何
通知）。今日只有 CLI 直连且 CLI 无 matrix，故干净；需在 `spawn_node` docstring 上写明。

### 本轮形态（决策）

- **bringup task 化（per-node 并行，无顺序语义，已落地 2026-09-13）**：每个 mode 声明的 node 各起一个后台
  task，`__aenter__` 不再 await `_bringup_nodes`。**不串行、不做 DAG 启动图**——node 依赖
  组织未来交给特殊 node，bringup 只逐条 fire，一个 node 的 probe 挂死/失败不拖累其余。
  语义变化："host 起来 = nodes 已在网"消失（nodes 供能力到网络给 ghost，host shell 的
  channel 树来自 `mode.manifests().channel()`，与 nodes 无依赖，解耦安全）。spawn 之后的
  存活/退出治理本就归 matrix（handle 登记 + `_on_cell_exit`），故不 provision 进 matrix 的
  task 托管（那层只做容错/关闭退出）。取消经 exit stack 回调，排在 matrix teardown 之前；
  `CancelledError` 是 `BaseException`，不被 task 内 `except Exception` 吞掉，取消语义天然正确。
- **dead end：bringup 串行（2026-09-12 偏航，当场纠正）**。首次实现写成了"单 task +
  循环串行 await"，沿用了旧 `_bringup_nodes` 的"按声明顺序"字样，凭空发明了列表序的启动
  顺序语义。两处错：① 顺序本就是假保证（旧串行也只保"发起顺序"、不保"就绪顺序"）；
  ② node 已明确声明不做 DAG 启动图，bringup 里塞顺序即越界。教训：旧代码里的措辞
  （"按顺序"）不等于设计意图，动到语义时要先与人类对齐，不要顺着字面续写。
- **probe 超时入 manifest（已落地 2026-09-13）**：`ExecSpec.timeout: float | None`（默认 None，
  显式声明才限时），`_run_probe` 用 `asyncio.wait_for` 包 `_await_exit`，超时 → `_terminate_probe`
  （`Subprocesses.killpg` 杀进程组）→ 判 broken reason "probe timed out after Ns"。
  沿用 exit-code-only 语义，不发明 ready 状态机。
- **失败通知（已落地 2026-09-13，scope 收窄到 mode bringup）**：区分两路——channel 侧
  `nodes:run` 已用 `raise_observe` 兜底（`matrix_channel.py` 把 DuplicatedError / NodeProbeError /
  FileNotFoundError / RuntimeError 全转成命令返回），无需事件；只有 mode bringup 是启动面、
  无直接调用方拿返回，才需要事件通知启动中的 ghost。故 publish 落在 `_bringup_one`
  （moss_runtime），**不在 `matrix.run_node`**。实现：`publish_event` 四层（Cell / Matrix /
  MatrixImpl / Presence）加 `event_level` 覆盖参数，bringup 失败 → `publish_event(content,
  event_level=ERROR)`。**不涉及 CellEvent schema 变更**：bringup 失败时 cell 不存在、无跃迁
  可标；`_on_cell_exit` 的退出/崩溃事件（`CellTransition.CRASHED` 补债）是独立问题，本轮不碰。
- **已知残留**：事件是瞬时的，mesh channel ring buffer per-instance；host boot 时 ghost 可能
  尚未订阅，恰是最该知道 bringup 失败的时刻最容易漏。事件单独用保不住 boot 场景，
  是否补"可 refetch 状态"承载（账本 vs main channel 命令）待定，本轮回合先做事件主体。

### smoke 验证（2026-09-13）

新增两个 system_test node：`probe_hang`（check 睡 3600s 不退出）、`probe_timeout`（check
睡 3600s 但 `timeout: 2`）。临时把 `system_test/HOST.md` 的 bringup_nodes 设为
[probe_hang, probe_fail, hello_world]，用 `moss-shell --mode system_test log` 实测：

- **启动不阻塞**：matrix + shell 起来后 `__aenter__` 立即返回；三 node 同一毫秒并发 spawn。
- **隔离**：probe_hang 卡探针、probe_fail 探针 exit 1 → 只记 `ERROR bringup node failed`
  （带 traceback），hello_world 照常拉起。三者互不拖累。
- **优雅退出回收挂死探针**：SIGINT → 完整 teardown，`Subprocesses stopping — 1 executing`
  杀掉挂死 probe，无孤儿。
- **probe 超时兜底**：`moss nodes run probe_timeout` 2s 报 `probe timed out after 2.0s`
  并退出，无泄漏进程。
- **观察（独立问题，非本改动缺陷）**：对 `moss-shell log` 发 SIGTERM 不触发优雅 teardown
  （进程不处理 SIGTERM）→ 硬杀 → 挂死 probe 成孤儿（ppid=1）。probe 超时兜不住这个
  （超时只约束父进程存活期间）；硬杀场景的进程组回收是另一个问题。
- **失败事件广播**：脚本注入 bringup=[probe_fail]，`network.recent_events()` 出现 1 条
  `level=40`（ERROR）事件，content 带 target + reason（"bringup node failed: ... probe failed"）。

复现配方已留在 `system_test/HOST.md` 注释里（`bringup_nodes: []` + 注释掉的列表）。

## Current Consensus

### 四地址组合 — node 发现路径前缀

node 发现路径有四个语义锚，对应四个确认方：

| 前缀 | 解析到 | 确认方 |
|---|---|---|
| （无前缀）| `project_dir` | 使用者 |
| `$MOSS_WORKSPACE` | `workspace_path` | 管理者 |
| `$MODE` | `workspace/modes/<mode_name>` | mode 开发 |
| `$GHOST` | `workspace/ghosts/<ghost_name>` | ghost 自己 |

默认 `node_paths` 扩展为四组合。普通使用者无需看懂组合语义，能力就位、自现。

### Matrix.new 默认 persist=False

- `persist` 参数进 `Matrix.new` 表面，默认 `False`（脚本启动式 = 一次性 run-to-completion）。
- `event_level` 不暴露，由 `persist` 推导（persist=false → DEBUG 静默）。

### 账本 / singleton 单写记账链（2026-08-30 收敛 + review 修正；2026-09-05 补同步预检）

spawn 咽喉（`NodeManager.spawn_node`）：installed 校验 → launcher 打包 → probe 闸门 →
**singleton 预检**（read-only `is_locked`，撞锁抛 DuplicatedError）→ **写第一笔账本**
（launcher.runtime：uid/address/cell，pid/pgid 占位 0）→ execute 拉起。
不持有 singleton 锁、不回填 pid/pgid、不删账本——锁持有 / pid·pgid 回填 / 退出删账全归
node 自身 `enter_cell_lifecycle`。前提是 node 就是 matrix cell（cell 定义即"Matrix 网络中
运行的进程单元"；纯脚本不入网、不做服务发现，不该用 node 体系承载）。

单写记账链（关键机制）：spawner 写第一笔（身份 uid）→ node `discover_this_node` 从账本读回
身份（uid 一致，不 fallback）→ node `enter_cell_lifecycle` 回填 pid/pgid → 退出删账本。
缺失第一笔会让 node fallback `build_cell_from_node` 重新生成 uid，父/子身份发散（review
抓回的严重 bug）。spawner 写第一笔后 `CellHandle.runtime` 的 pid/pgid 仍是占位 0，真 pid
由 node 回填的账本提供。

### singleton 同步预检补丁（2026-09-05）

狗粮实测 screen（GUI node）可被并发/直接拉起多个实例：QML 窗口在 `main.py` 先 load、
Matrix daemon 线程后进 `enter_cell_lifecycle` 才抢锁，锁失败抛 DuplicatedError 落在 daemon
线程未捕获、窗口照开；且 spawn 路径（matrix.run_node）无同步 singleton 判定，
`matrix_channel.py` 的 `except DuplicatedError` 是死代码（错误发生在子进程）。

修：`spawn_node` 咽喉加同步 read-only 预检（`is_locked`，撞锁抛 DuplicatedError），让
matrix 路径调用方能同步感知顺序重复；CLI 原 read-only probe 收敛删除，改 catch
DuplicatedError。锁仍归 child 持有，本预检**不消除并发 TOCTOU**（`is_locked` 不持有锁）。

### 死文件观察垫（2026-08-29）

非优雅退出（crash / kill -9）时 node 的 `finally` 不执行，账本残留为 stale 记录。这份
残留**保留**（不在 spawn 时自动清），理由见 `node_manager.py` spawn_node 的观察垫注释：
① uid 动态，死文件是 crash 唯一可追溯痕迹；② jobs 已移除、无自动 respawn，反复 crash
目前不存在；③ 清账逻辑存在会抹掉"错误退出"的验证点。清理交由 host 启动/退出 +
CLI prune。若未来出现同 fullname 反复 crash 累积，再补 spawn 时只查本 fullname 的 done
callback（不做全目录轮询）。

### 收敛附带语义变化（2026-08-30）

- `kill_cell` 从"单发 SIGTERM fire-and-forget"（旧 `Project.kill_cell`，已删）变成
  "SIGTERM → 3s grace → SIGKILL（同步阻塞）"，host 清孤儿回调因此同步阻塞——观察垫。
- `spawn_node` 签名破坏性变更：capture 从 `CaptureSpec` 变 `Callable[[CellRuntimeInfo],
  CaptureSpec]`（落盘路径依赖 runtime.address），返回从 `ManagedProcess` 变
  `tuple[CellRuntimeInfo, ManagedProcess]`。

### reconcile review 有效（2026-08-30）

zero-context reconcile review（`moss features review` 遗忘测试）抓出 1 个严重 bug（uid
身份发散，根因是误删"启动方先写账单"）+ 4 个声明/交付 drift（DuplicatedError 契约、
kill_cell 语义、probe stdout、spawn 签名），已全部修复并同步进本 FEATURE。这是"声明 vs
交付"遗忘测试的实证价值。

## Open Questions

- **publish_event 级别（已解决 2026-09-13）**：`publish_event` 四层（Cell / Matrix /
  MatrixImpl / Presence）已加 `event_level` 覆盖参数（默认 None = 沿用 cell 级别），解决
  "默认静默但能喊"做不到的问题；本轮 bringup 失败通知复用同一参数。
- **probe 动态自描述进模型认知窗口**：probe stdout 作为动态 self-description 与
  instruction（静态）并列，如何进 open/read 面，后续评估，本次先做闸门主体。
- **singleton 并发 TOCTOU**：`is_locked` 预检不持有锁，快速连发（ghost 异步 `run_node`，
  GUI 节点 Qt 启动 ~1s 窗口）仍可多实例并发过检。要真正互斥需 spawner 持锁至 child 接管，
  或 child 提前到开窗前拿锁。暂接受。
- **GUI node 锁晚于开窗**：screen 的 flock 在 daemon 线程、QML load 之后；锁失败不关窗口。
  需把锁提前到主线程开窗前，或 daemon 失败时通知主线程 `app.quit()`。
- **phantom pid=0 账本**：并发 spawn 时失败的 child 在 `enter_cell_lifecycle` 里
  DuplicatedError 抛于 `_runtime_info_ctx` 进入之前，spawner 写的第一笔 pid=0 账本永不清理；
  且 `psutil.pid_exists(0)==True` → `nodes status` 显示成 alive 幽灵。观察垫只覆盖
  crash/kill-9，没覆盖 singleton 冲突 fast-fail 路径。