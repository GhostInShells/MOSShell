# CellEventNucleus — M7.5 + M5.1 设计定案

CellEvent → Signal 链路，M9 telos 的必经环节 ("新器官 ready 送模型注意力候选")。
上游判决: cell-run-cycle FEATURE.md M7.5/M5.1 + matrix-cell-governance §WW-5/§WW-6。
与 channel 层的边界: **生产侧归 mesh channel** (matrix-channel.md §5.2 定,
2026-07-19 三 channel 拆分), 本 nucleus 独占消费侧 (signal → impulse 转换)。
一份订阅 (mesh channel on_startup) 无双写, unsub 跟随 channel 生命周期。

## 1. 形态

- MossRuntime 生命周期 aenter 时挂载专属 nucleus (工作命名 `CellEventNucleus`，
  与 mindflow 现有 nucleus 家族命名对齐后再定)，aexit 释放 —
  生命周期归 runtime，不归 matrix。
- 消费 `(await matrix.mesh()).on_event`，每条 CellEvent 转一条 Signal
  送 `runtime.mindflow`。
- **全部 background hint 姿态**: 低优 impulse，闲时才竞争到注意力。
  不做类型分档 (crash / new-ready / normal-exit 全同档) — 场景倒逼时
  单点加 `exit_code != 0 → high priority` override，一行代码 (Q3.2/Q3.3 定案)。

## 2. 层级判决 (否掉路径备查)

- **挂 MossRuntime，不挂 Host Matrix** — Matrix 承诺网络 primitives
  (Presence/Watcher/mesh)，Signal ↔ Mindflow ↔ Nucleus 是认知层语义。
  Matrix 保持纯网络门。
- **notify 姿态否掉** (强制入历史 + 触发动作) — 与 mindflow 闲时竞争哲学
  冲突且难拆；background hint 先行，升级可控 (四理由见 FEATURE.md M7.5)。

## 3. 无条件挂载 (2026-07-13 讨论增补)

signal 是 matrix/session 层**可不消费的协议动作**，不与 mindflow 耦合。
推论: nucleus 是无条件系统机制，**不按运行模式分支** —
MCP 模式下 signal 无人消费，无害；ghost 模式 mindflow 订阅即得。
不写 "if ghost mode then attach" 之类的分支。

## 4. payload 规范 (§WW-6)

胶囊薄快照: address/alias、生命周期跃迁、exit code、日志路径**指针**、
至多 stderr 尾部数行。日志本体不进 signal (拉取面归 WW-5 故事 8:
日志文件 + bash)。

- owner 的 dead 信号源 = Subprocesses done callback (进程真相，即时，
  带 exit code)，不走网络；网络 liveness 丢失对 owner 降级为对账。
- 正常退出 (exit 0) 也发 — 器官脱落，四弧之④。

## 5. M5.1 一体: run_cell 无 wait

`Matrix.run_cell` ABC 已无 wait 参数 (blueprint/matrix.py 已落)。
本 milestone 落地时核查:

- `matrix_impl.run_cell` 实现签名与 ABC 一致，无 wait 残迹。
- 全库 grep `run_cell(` 调用点，无人传 wait。
- 不留 "程序化 bootstrap 场景" 例外。

## 6. 位置

建议 `src/ghoshell_moss/matrix/nuclei/cell_event_nucleus.py`
(与 `matrix/networks/` 平级，为未来 audio/vision/topic nucleus 开位)。
最终位置执行时判断。

## 7. 风险监测点 (非 blocker)

crash 静默: 模型拉起 cell → 立刻崩 → 模型不知道继续调不存在的 channel。
可能在 M8.5 (L1 tutorial) 踩到。观察到就升分档，观察不到说明恐惧是空想。
注意 cells channel 的 context_messages (pull 面) 会显示 cell 状态，
是 signal 之外的第二道兜底。

## Tasks

### T1. nucleus 实现

**判据**: CellEventNucleus 按 §1/§4 实现，消费 on_event → Signal，
payload 守 WW-6 胶囊规范。

### T2. MossRuntime 挂载

**判据**: aenter 注册 / aexit 释放，无模式分支 (§3)。
MossRuntimeImpl (`host/moss_runtime.py`) 是归宿。

### T3. wait 残迹核查

**判据**: §5 三项全过。

### T4. 单测

**判据**: 伪造 CellEvent 流 → 断言 Signal 产出与 payload 形状;
无 mindflow 消费者时 nucleus 不报错 (§3 无条件挂载的验证)。
