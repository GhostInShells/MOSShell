---
title: Dolores Ghost Prototype Regression
version: 1
status: draft
priority: P1
created: 2026-10-03
updated: 2026-10-04
scope: subsystem
depends: [ghost-prototype-dolores]
description: >-
  dolores ghost prototype 的整体验收用例，按现状逐条累加；当前先装「turn/end 退出与中断」这一组。
---

# Dolores Ghost Prototype Regression

## 这个集合是什么、怎么长起来

- 它是 **prototype 的整体回归**，不是某处机制的窄表。用例从 `ghost-prototype-dolores/dolores-todo.md`
  按现状提取，**只收不能被 pytest 覆盖的命题**（纯自动化能覆盖的留在 `tests/`，不进这里）。
- **正式全量**：人类抽一天与模型录屏做一次 —— 清 stale 代码与注释 → 清 todo 表 → 提取命题 →
  写入本表 → 覆盖回 todo → 标记 "dolores ghost prototype 开发关闭"。在那之前本表按"一条条加"生长。
- **历史教训**：曾有一套 `.ai_partners/regressions/dolores/REGRESSION.md`（2026-09-01），从未跑过、
  没有 baseline，用例里还混着"跑 pytest"，后被删除。**没有 baseline 的回归等于计划**。本表每条用例
  都必须有外部可读的判据，且要一路跑到能写 baseline。
- `ghost-runtime` 是**另一套**回归，与本表无关。
- 本表当前对应 **dolores ghost prototype 0.1.0**。prototype 版本写在正文而不写进 frontmatter：
  那个字段按约定是"用例结构变更计数"，而 prototype 版本是另一根会独立前进的轴——两条轴挤一个
  字段，等它们错位那天它就开始说谎（baseline 文件名同理，跟 `version`）。

## Exploration Index

```
moss codex get-interface ghoshell_moss.ghosts.dolores._ego:DoloresEgo
moss codex get-interface ghoshell_moss.ghosts.dolores._run:DoloresRun
moss codex get-interface ghoshell_moss.core.blueprint.mindflow:Thinking
moss-ghost run --help
moss-ghost send --help
find .ai_partners/features/regressions -name REGRESSION.md
git log --oneline -10 -- src/ghoshell_moss/ghosts/dolores/
```

## Methodology

- **人在环**（human-in-the-loop），跑在**真实 dsh + 真实 mindflow** 上；不用替身——`test_dolores.py`
  里的 fake 曾发明过生产不存在的 API（单测全绿、生产必挂），那类失真正是回归要避开的。
- 两个执行面：
  - **观测**：`moss-ghost run --surface output` —— dsh 事件与 run 的收线都在这个面上。
  - **驱动**：`moss-ghost send` 驱动模型；**dsh web UI 由人手操作**（点中断、注入输入）。
- **用例准入判据**：每条用例的"预期结果"必须能被外部读到 —— 一行日志（收线 kind / cause / 本次采取的
  动作），或一个物理结果（语音是否停、会话能否继续输入）。写不出外部判据的，不进本表。
- 执行顺序：先 TC-000（判据可见性），其余按优先级；用例之间无强依赖。

## Prerequisites

- 完整克隆的 MOSS workspace；`.venv` 可用；`.moss` workspace 已初始化；provider 已配置（`.env`）。
- dsh web 可达（人手点中断用）。
- **TC-004 需要一个可复现的断连手法**，尚未确定（候选：强断 provider / 断开 dsh 连接 / kill dsh）。
  该条暂记 `BLOCKED`，先设计手法。

## 退出机制事实（turn/end）—— 本组用例的共同背景

> 当前代码事实，非结论。凡与此处冲突，**以源码与实测为准**。

四类收线：

| # | 场景 | 今天的代码行为 | 应有的行为 |
|---|------|----------------|------------|
| 1 | ego 侧发起 cancel（tool 结果带 `cancel=True`：`moss_wait_next` / `moss_react` / `moss_reasoning` / CTML 解析错） | plugin 在下一 pre-step `stepAgent.cancel({kind:'hook', reason:'moss cancel_turn'})` → turn/end → `logos()` return → run 收线；四个正常出口返回前已 `wait_actions_done()` | 自然退出；身体不被切（模型已交回、已 drain） |
| 2 | thinking 自己退出（mindflow 调度） | run 先退出 → `exit_thinking()` → plugin 见 agent 非 idle 再 `agent.cancel({kind:'hook', reason:'moss thinking/exit'})` → turn/end 在 MOSS 离开之后才到 | 该 turn/end 只是回声，不得触发任何"停身体" |
| 3 | dsh 界面点中断 | turn/end(aborted) → `logos()` return → `__aexit__` 只取消 enter task、解绑、`exit_thinking()`；无异常故不 abort；**Shell 侧没有任何停止** | `thinking.abort`（交互预期=整个体系停）+ **同一时刻显式停身体**（不能等下一帧） |
| 4 | 运行时错误 / 异常 | `turn/end kind=error/interrupted` 不是 python 异常 → 现在不 abort → 可能自续成环 | 必须中断 thinking，不进入死循环；可类比既有熔断 `_interpret_failure_limit = 2` |

**代码锚点**

- `src/ghoshell_moss/ghosts/dolores/_run.py`：`_note_turn_end`（收线判定）、`__aexit__`（仅异常才 abort，
  `CancelledError` 直接返回）、四个 tool handler 里 `wait_actions_done()` 的位置。
- `src/ghoshell_moss/ghosts/dolores/_ego.py`：`_on_turn_end`（只推进 last_turn + memento 阈值）、`exit_thinking()`。
- `src/ghoshell_moss/ghosts/dolores/dsh_plugin/moss-dolores-ghost-plugin.ts`：`cancel_turn` 两处 pre-step 判定、
  `thinking/exit` 触发的 cancel。
- `src/ghoshell_moss/core/mindflow/mindflow_in_shell.py`：`interrupt_first` / `previous_stop_reason()`
  → `shell.clear()`（既有"下一帧首帧清 body"路径）。
- `src/ghoshell_moss/core/mindflow/_mindflow.py`：`_last_abort_reason` → `moment.previous.stop_reason`。
- `src/ghoshell_moss/deepseek_harness/types/session_events.py`：`TurnEnd` / `TurnEndReason` / `AgentCancelCause`。

**已实测（2026-10-04）· `turn/end` 带 `cause`**：dsh 设计文档说 durable 的 turn/end 只有粗粒度
`aborted`、不携带 user/parent；实机证明**带** —— 人点停到达为 `aborted` + cause `user`
（日志 `turn 2 ended: aborted/user`），我们自己的 hook cancel 到达为 `hook`（`moss cancel_turn`）。
判据因此可以挂在 cause 上，不必依赖 turn 号。

**stale 记录已清理（2026-10-04）**：`_run.py` 中"turn/end 只用于观测、绝不驱动中断"的注释与
`_note_turn_end` 的 docstring 已删改 —— 它把第 1 类场景的判据写成了通性（人类裁定：错）。

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|-----------------|
| TC-000 | P0 | **判据可见性**：四类收线各自都能从日志读出 kind(+cause) 与本次采取的动作 | 1. 依次触发四类收线<br>2. 读 `moss-ghost run --surface output` | 每类一行、无歧义；cause 存在与否被记录成**事实**（无论 PASS/FAIL，这条都要有结论） |
| TC-001 | P0 | 第 1 类：tool 主动交回后的收线 | 1. 让 ghost 说一段较长的话后调用 `moss_wait_next`<br>2. 观察收线与语音 | 语音完整播完（cancel 前已 drain）；日志可见 `aborted` + `hook/moss cancel_turn`；身体未被打断 |
| TC-002 | P0 | 第 2 类：thinking 自退引发的 turn/end | 1. 构造 attention 自然结束（无自续帧） | turn/end 在 run 退出**之后**到达；不触发任何停身体；会话可继续 |
| TC-003 | P0 | 第 3 类：dsh 界面点中断（说话中点） | 1. 让 ghost 说一段较长的话<br>2. 说话中点 dsh web 的中断 | 语音**立即停**；下一帧带 `<stop_reason>`；会话可继续输入；日志可见 `user`（若 cause 不可得，按 TC-000 的结论处理） |
| TC-004 | P0 | 第 4 类：运行时错误 / 断连 | 1. 运行中制造断连（手法待定） | thinking 被中断；不进入死循环；日志可见 `error` / `interrupted` |
| TC-005 | P0 | 护栏：`completed` 自然收线不许回退 | 1. 正常完成一轮对话 | 无 abort；语音播完；attention 保留（下一帧可自续） |

> TC-003 已修并首测通过（2026-10-04，见 Execution Notes；`moss_react` 入口仍待补测）；
> TC-004 仍是已知缺陷：用例先立，修复后再跑。

## Execution Notes

- 判据读 `moss-ghost run --surface output` 的 `dsh/*` 事件与 run 侧收线日志；"语音是否停"只能靠人耳或录屏。
- TC-000 的日志前提已在 2026-10-04 落地（`_predispatch_abort` 的 `before the consumer runs` 那行，
  以及既有的 `turn N ended: kind/cause`），用例由此可执行。
- **无 baseline**。首次完整跑通后写 `baselines/YYYY-MM-DD_v1.md` 并转 `active`。

### 2026-10-04 · 首轮实测（半程）

**TC-003（dsh 界面点中断）跑两次。**

| # | 时刻 | 代码版本 | 结果 |
|---|------|----------|------|
| 1 | 00:29 | 上一版 | **FAIL** — 语音 41.3 秒播满；日志只有 `turn 2 ended: aborted/user`，无 abort 落地、无 clear |
| 2 | 01:06 | 修后（P1–P4） | **PASS** — 语音 3.1 秒处被掐断；探针早于消费者 20ms；整树 clear 22ms 内完成；下一帧带 `<stop_reason>turn/end aborted (user)</stop_reason>`；会话可继续输入 |

- **边界（未闭环）**：第 2 次走的是 `moss_interpret` + `moss_wait_next`，**`moss_react` tool 入口仍未实测**。
  死锁形状相同，P1 让 abort 绕开消费者，预期成立 —— 但未验就是未验。
- **TC-000（判据可见性）结论**：cause **可得**（`aborted/user`、`hook/moss cancel_turn`），
  因此 TC-003 里"若 cause 不可得"的分支不必走。
- **TC-001 / 002 / 004 / 005**：未跑。
