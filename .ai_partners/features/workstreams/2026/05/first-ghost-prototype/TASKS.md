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
- [x] 10g — output role 体系: 六 role 分离, command/error/system 已落地, logos 缓冲过渡
- [x] 10h — session stream logos: logos 走 session stream 流式输出, 替代缓冲检查点
- [x] 10i — ghost 可观测性体系: on_articulate_exit + inspect_state (Ghost), inspect_loop_health (GhostRuntime)

### Phase 3: 感知侧补齐 + 全链路测试

- [x] 11a — Mindflow 默认 input signal: GhostRuntime fallback 改用 new_default_mindflow() (InputSignalNucleus + PriorityProtectionAttention)
- [x] 11c — on_challenge 旁路观察: ChallengeObserver 类型 + Mindflow.on_challenge() + _challenge_attention 触发 (test_on_challenge.py 验证)
- [ ] 11b — Mindflow inspect + 自解释接口
- [ ] 12a — mock ghost + input signal 脚本测试
- [ ] 12b — TUI 全链路验证
