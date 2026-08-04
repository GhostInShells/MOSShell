---
title: QA Exchange Protocol
status: in-progress
priority: P1
created: 2026-08-03
updated: 2026-08-05
depends: []
milestone:
description: >-
  广播问答交换协议 — Asker 广播问题 / Watcher 发现并应答 / requester 持真相 / 先到先得裁定。
  janus (进程内) 和 zenoh (跨进程) 双实现，31+6 条行为测试全过。
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

## Current Status (2026-08-04)

**已完成：**
- 概念层 ABC: `core/concepts/qa.py` — QA / Asker / Watcher / QAManager
- janus 进程内实现: `core/qa/janus_qa.py` — JanusQA / JanusAsker / JanusWatcher / JanusQAManager (v0.1)
- zenoh 跨进程实现: `matrix/qa/zenoh_qa.py` — ZenohQA / ZenohAsker / ZenohWatcher / ZenohQAManager (v0.1)
- 生命周期: QAManager 为 async context manager，Asker/Watcher 为纯同步工厂，全部 task/subscriber 在 exit 时清理
- 概念层行为测试: 18 条 (tests/.../concepts/test_qa_concept.py)
- janus 集成测试: 13 条 (tests/.../qa/test_janus_qa.py)
- zenoh 集成测试: 6 条 (tests/.../matrix/qa/test_zenoh_qa.py)
- 测试风格指南: `tests/CLAUDE.md`
- safemode 审批场景评估: safemode 当前实现已经足够好，不再需要走 QA 体系接线
- IoC Provider: `matrix/providers/qa_provider.py` — ZenohQAManagerProvider，singleton，合约 QAManager → ZenohQAManager
- manifests 注册: `.moss/src/MOSS/manifests/providers/__init__.py` — qa_manager_provider
- default_providers: `zenoh_adapter.py` — ZenohAdapter.default_providers() 自动装配
- system_test_nodes: `qa_asker` / `qa_watcher` — 跨进程双节点实机验证通过 (2026-08-04)

**下一步：**
1. ~~QA 进入 manifests 体系~~ Done
2. ~~system_test_nodes 双节点交互验证~~ Done
3. ~~进入 matrix 默认 IoC~~ Done
4. 评估 shell / channel 级默认 API
5. **修复 zenoh 回调线程问题**: ZenohWatcher._on_question_sample / ZenohAsker._on_reply_sample 当前在 zenoh I/O 线程执行用户回调和状态变更，需补 janus queue 卸载到 event loop task（janus 实现已有此隔离，zenoh 缺失）

## Design Index

- 概念层：`src/ghoshell_moss/core/concepts/qa.py` — QA / Asker / Watcher / QAManager ABC
- janus 实现：`src/ghoshell_moss/core/qa/janus_qa.py`
- zenoh 实现：`src/ghoshell_moss/matrix/qa/zenoh_qa.py`
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
| manifest 注册 | `.moss/src/MOSS/manifests/providers/` | IoC Provider (待做) |
| 集成验证 | `.moss/system_test_nodes/` | 双节点交互 (待做) |
