# Tasks — First Ghost Prototype

## 当前阶段：实现

进度：Phase 1 设计讨论 7/7 完成。Phase 2 实现进行中。

## 任务分解

### Phase 1: 设计对齐

- [x] 01 — Ghost ABC 定位讨论
- [x] 02 — 最小原型技术目标
- [x] 03 — 基建依赖准备
- [x] 04 — Runtime 集成
- [x] 05 — Matrix / TUI 集成
- [x] 06 — 全链路测试方案
- [x] 07 — 文档准备

### Phase 2: 实现

- [x] 08 — Ghost ABC 微调 + 三层抽象落地
- [x] 09 — _meta / _runtime / _adapter + 24 tests
- [x] 10a — SystemPrompter tree 模型重构 + 迁至 blueprint.host
- [x] 10b — GhostRuntime ABC 定义 + 架构选型 (DESIGN.md)
- [x] 10c — GhostRuntimeImpl skeleton (wiring: providers → moss → ghost → mindflow)
- [x] 10d — action observe 回路 (流式 feed → interpreter → action.outcome) + 异常分级
- [x] 10e — session signal → mindflow 路由 + matrix 信息输出
- [x] 10f — task output 协议: `on_task_done` → session.output, 结算只发 status
