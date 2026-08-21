---
created: 2026-08-03
depends: []
description: 广播问答交换协议 — Asker 广播问题 / Watcher 发现并应答 / requester 持真相 / 先到先得裁定。 janus
  + zenoh 双实现，TUI 非阻塞交互，31+6 条行为测试，system_test 节点实机验证。
milestone: null
priority: P1
status: completed
status_note: answer-node CLI 消费面落地 (rich+prompt_toolkit), FEATURE.md 治理完成
title: QA Exchange Protocol
updated: '2026-08-22'
---

# QA Exchange Protocol

> Use `moss features set-status qa-exchange <status> -m "note"` to update state.

## Motivation

future-router (completed, P2) 建了进程内 Future 路由基建 (concurrent.futures + str 协议)，
但 ownership 模型是 "共享可变句柄" —— 任何人拿到 id 都能 settle。它在设计动机上没有满足要求，被冻结。

本 feature 做所有权反转：**requester 持真相，应答者无权变更终态。**
协议命名从 "future" 翻转为 "QA" (Question & Answer)，落在 concepts 体系内，
与 topic 并列 —— topic = 广播数据 (push)，QA = 广播问题 + 带回答案 (ask)。

QA 的聚合价值：用 namespace 构建不同对话空间，将各种场景的对话需求汇总在同一个界面内。

## Current Status (2026-08-06)

**已完成：**
- 概念层 ABC: `core/concepts/qa.py` — QA / Asker / Watcher / QAManager
- janus 进程内实现: `core/qa/janus_qa.py`
- zenoh 跨进程实现: `matrix/qa/zenoh_qa.py` + janus queue 线程隔离
- 生命周期: QAManager async context manager，Asker/Watcher 纯同步工厂
- 概念层行为测试 18 条 + janus 集成测试 13 条 + zenoh 测试 6 条
- safemode 审批场景评估 (结论: 当前实现已足够好)
- IoC Provider + manifests 注册 + default_providers 自动装配
- system_test_nodes: `qa_asker` / `qa_watcher` / `qa_pusher` — 实机验证
- `session.qa` 进入 Session ABC — 跨 cell 广播问答总线
- `Question.markdown` 字段 + `password` kind — 富文本展示 + 密码输入
- **TUI QA State** (`host/tui_entries/qa_state.py`):
  - C-q 切换 QA 视图，非模态，SafeMode 仍是唯一模态闸
  - list/detail 双模式，1-9 数字键直选、N+1 = reject
  - kind-aware 交互：confirm/choose/apply 一键提交，select 多选，input/password 文本输入 + Tab 补全
  - content + markdown 并行渲染
  - bottom toolbar `[N Questions]` 实时计数
  - 空列表 Enter = pop 回之前 state
  - QA state 不参与 C-t 循环
- `namespace or "default"` fallback — zenoh keyexpr 通配符友好
- `qa_pusher` system_test 节点 — confirm/input/choose/select 四问实机验证通过
- **CLI answer-node** (`cli/nodes_answer.py` + `cli/nodes_cli.py` 的 `answer-node` 子命令):
  - `moss nodes answer-node [--namespace NS]` — 无 GUI 的 headless QA 应答终端
  - 进程内 `Matrix.new()` 建矩阵，不拆 node / 不开子进程（开箱极简）
  - rich 渲染 question（panel + markdown + options 表）+ prompt_toolkit 交互（持久 PromptSession + patch_stdout）
  - completer 下拉列 option/reject（confirm/choose/apply）；select 数字多选；note 追加进 content
  - 底部 toolbar 实时 pending 计数；done-mid-display / first-wins 语义纳入

**未来迭代：**
- shell / channel 级默认 API
- Ghost 主动感知 QA namespace 并对问题发言
- GUI 消费面 — 参考 answer-node 的 rich+prompt_toolkit 交互，整合进 screen
- answer-node live-abort（done 时即时中断当前 prompt；当前为 submit 时复查）

## Design Index

- 概念层：`src/ghoshell_moss/core/concepts/qa.py`
- janus 实现：`src/ghoshell_moss/core/qa/janus_qa.py`
- zenoh 实现：`src/ghoshell_moss/matrix/qa/zenoh_qa.py`
- TUI QA State：`src/ghoshell_moss/host/tui_entries/qa_state.py`
- TUI 基类 QA 集成：`src/ghoshell_moss/host/tui.py` (MossHostTUI)
- CLI answer-node：`src/ghoshell_moss/cli/nodes_answer.py` + `src/ghoshell_moss/cli/nodes_cli.py` (`answer-node`)
- 验证节点：`.moss/system_test_nodes/qa_pusher/`
- 前身 (已冻结)：`src/ghoshell_moss/tools/future_router.py`
- 前身 feature：`workstreams/2026/06/future-router/` — completed

## Key Decisions

### KD1: requester 持真相，应答者无权变更终态

QA 是 requester 的真理源。应答者只能一次提交 (reply) ，不能置 done / cancel。
所有权是二阶段的：replied（应答提交，first-wins 锁定）→ accepted/done（仅 owner 可终态转移）。

### KD2: 三阶段生命周期

请求 (issue broadcast) → 应答 (reply，先到先得) → 裁定广播 (done/cancel verdict)。

### KD3: 角色拆分为 Asker / Watcher / QAManager

- **Asker** = issuer + 问题工厂 + undone 重建查询。
- **Watcher** = 发现 + 应答 + on_question 推。
- **QAManager** = asker/watch 工厂 + issuer 身份。自身为 async context manager。
- Asker/Watcher 为纯同步工厂，不暴露自身 context manager。

### KD4: QA 是纯抽象，广播/传输/角色强制是 impl 层

概念 ABC 定义契约面；janus 队列 / zenoh pub-sub / queryable 恢复在实现层。

### KD5: QA 自身不做持久化存储

requester 活着 = 真相活着。zenoh 版跨进程迟到 watcher 通过 Asker.undone() 经由 zenoh queryable 重建发现。

### KD6: 命名

协议 QA / 实体 Question Answer QAMeta / 角色 Asker Watcher QAManager /
审批域 kind='apply' 动词 approve reject / 关联 refer_to

### KD7: Question 自带 kind + 答案构造器

kind: input / confirm / apply / choose / select。Answer 自身带 match_question 校验。

### KD8: 先到先得 (first-wins)，应答者一次约束

### KD9: answer-node 是 CLI 消费面，rich + prompt_toolkit 混用

无 GUI 时也要能应答 QA。`moss nodes answer-node` 进程内 `Matrix.new()` 建矩阵
（不拆 node / 不开子进程，开箱极简）。交互策略：rich 渲染 question 正文 + markdown +
options 表，prompt_toolkit 用持久 PromptSession + patch_stdout 驱动交互，completer
下拉列 option/reject（单选类），select 走数字多选，note 追加进 Answer.content。
这是 GUI 消费面的参考实现。

## Implementation Notes

### 生命周期

QAManager.__aenter__ 开始，__aexit__ 清理全部 task/subscriber/queryable。
Asker/Watcher 由 QAManager 的 spawn 回调绑定任务追踪，不暴露自己的 with statement。

### zenoh keyexpr 布局

```
{prefix}/questions/{ns}   — question broadcast
{prefix}/replies/{ns}     — answer submissions
{prefix}/verdicts/{ns}    — accepted answer / cancel
{prefix}/query/{ns}       — late-join queryable → Asker.undone()
```

qid 始终在 payload (QAMeta.refer_to)，不在 keyexpr。

### 架构位置

| 层 | 路径 | 性质 |
|----|------|------|
| 概念 | `core.concepts.qa` | ABC contract, topic 兄弟 |
| janus 实现 | `core.qa.janus_qa` | 进程内 janus queue |
| zenoh 实现 | `matrix.qa.zenoh_qa` | 跨进程 zenoh pub/sub |
| manifest 注册 | `.moss/src/MOSS/manifests/providers/` | IoC Provider |
| 集成验证 | `.moss/system_test_nodes/` | 双节点交互 |
| CLI 消费面 | `cli/nodes_answer.py` | headless 应答终端 (rich + prompt_toolkit) |