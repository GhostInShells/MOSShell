---
created: 2026-08-04
depends: []
description: Node 生命周期治理，从 node-migration 独立。四层方案已收敛为 node 最佳实践探索： 四地址发现、一次性 node、事件分级已落地；重新论证后决定
  drop node id、做 probe 预启动闸门。收尾中。
milestone: 0.1.0
priority: P1
status: completed
status_note: 人类判断可以 completed：reconcile review 抓回 uid 身份发散 bug 并修复，4 个声明/交付 drift
  已同步
title: Node Lifecycle — 身份、入口、验证与记忆
updated: '2026-09-05'
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

- **publish_event 级别**：persist=false 脚本主动 `publish_event` 目前被 cell.event_level
  锁死（`zenoh_presence.py:174`），"默认静默但能喊"做不到。是否加显式 event_level
  覆盖参数，待定。先不加。
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