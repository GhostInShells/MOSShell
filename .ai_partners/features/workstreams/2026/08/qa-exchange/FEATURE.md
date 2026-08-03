---
title: QA Exchange Protocol
status: draft
priority: P1
created: 2026-08-03
updated: 2026-08-03
depends: []
milestone:
description: >-
  广播问答交换协议 — Asker 广播问题 / Watcher 发现并应答 / requester 持真相 / 先到先得裁定 / janus 进程内实现。
---

# QA Exchange Protocol

> Use `moss features set-status qa-exchange <status> -m "note"` to update state.

## Motivation

future-router (completed, P2) 建了进程内 Future 路由基建 (concurrent.futures + str 协议)，
但 ownership 模型是 "共享可变句柄" —— 任何人拿到 id 都能 settle。它在设计动机上没有满足要求，被冻结。

本 feature 做所有权反转：**requester 持真相，应答者无权变更终态。**
协议命名从 "future" 翻转为 "QA" (Question & Answer)，落在 concepts 体系内，
与 topic 并列 —— topic = 广播数据 (push)，QA = 广播问题 + 带回答案 (ask)。

核心用例：(1) ghost-runtime-safemode 审批闸 (approval: apply kind, approve/reject)
(2) 模型发起的异步任务追踪 (answer = task result)。

## Design Index

- 概念层：`src/ghoshell_moss/core/concepts/qa.py` — QA / Asker / Watcher / QAManager ABC
- 实现层：`src/ghoshell_moss/core/qa/janus_qa.py` — JanusQA / JanusAsker / JanusQAManager (v0.1 骨架)
- 前身 (已冻结)：`src/ghoshell_moss/tools/future_router.py`
- 前身 feature：`workstreams/2026/06/future-router/` — completed
- 审批消费者：`workstreams/2026/07/ghost-runtime-safemode/` — in-progress

## Key Decisions

### KD1: requester 持真相，应答者无权变更终态

QA 是 requester 的真理源。应答者只能一次提交 (reply) ，不能置 done / cancel。
所有权是二阶段的：replied（应答提交，first-wins 锁定）→ accepted/done（仅 owner 可终态转移）。

### KD2: 三阶段生命周期

请求 (issue broadcast) → 应答 (reply，先到先得) → 裁定广播 (done/cancel verdict)。
阶段三在实现层 (core.qa) 通过广播完成，概念层 ABC 屏蔽传输细节。

### KD3: 角色拆分为 Asker / Watcher / QAManager

- **Asker** = issuer + 问题工厂 + undone 重建查询。对应提问方。
- **Watcher** = 发现 + 应答 + on_question 推。对应应答方。
- **QAManager** = asker/watch 工厂 + issuer 身份。
- Asker 和 Watcher 各有 namespace 生命周期。TUI gate 只造 Watcher，Ghost 造 Asker。

### KD4: QA 是纯抽象，广播/传输/角色强制是 impl 层

概念 ABC 定义契约面；janus 队列、跨进程 zenoh queryable 恢复、角色强制靠 `meta.issuer` 检查都在实现层。
`own` 标志从 `meta.issuer == identifier` 推导。

### KD5: QA 自身不做持久化存储

requester 活着 = 真相活着。跨进程迟到 watcher 通过 Asker.undone() 经由 zenoh queryable 重建发现。
不需要 sqlite3 真相源 —— requester 的内存状态即 truth。

### KD6: 命名

- 协议：QA (Question & Answer)
- 实体：Question / Answer / QAMeta
- 角色：Asker / Watcher / QAManager
- 审批域：kind='apply'，动词 approve / reject
- 关联：QAMeta.refer_to（答案引问题，中性链路）

### KD7: Question 自带 kind + 答案构造器

工厂方法与校验集中在 Question 上 (ask_approval / ask_confirm / ask_select / ask_choose)。
Answer 自身带 match_question 校验。

### KD8: 先到先得 (first-wins)，应答者一次约束

reply() 先到先得。应答锁 = 每个应答者只能操作一次，然后等裁定广播。

## Implementation Notes

- `src/ghoshell_moss/core/concepts/qa.py` — 概念层 ABC，hand-written by human architect (thirdgerb)
- `src/ghoshell_moss/core/qa/janus_qa.py` — janus 进程内实现骨架，v0.1 待补
- `src/ghoshell_moss/core/qa/__init__.py` — 新包入口
- 旧 `prompter.py` 已删除 (rename → qa.py)
- 与 future-router 关系：吸收替代；future-router 的消费者 (test only) 后续迁移
- TUI 集成路径：QA REPL state (TUIState) + Watcher.on_question 推入
- 跨进程路径：zenoh queryable → Asker.undone() 重建发现 (后续迭代)

### 架构位置

| 层 | 路径 | 性质 |
|----|------|------|
| 概念 (concepts) | `core.concepts.qa` | ABC contract, topic 兄弟 |
| 实现 (core.qa) | `core.qa.janus_qa` | janus in-process impl |
| 反射映射 | `core.concepts.qa` | 需加入 architecture.py |
