# Dolores Ego 注册 — 方案(推翻重写)

> 本文件推翻了此前「拆两个 plugin(web 侧 bridge + 非 web 侧 agent preset)」的拓扑设计——那个设计是错的, 已删除。完整讨论轨迹与设计演变见 git log, 本文只保留极简结论与复盘。

## 当前方案(极简)

**单 plugin 无依赖**。`moss-dolores-ghost-plugin.ts` 一个文件, 经 `cordis.patch.yml` 挂到 profile。

- `apply(ctx)` 注册 RPC 桥(ego/create、thinking enter/exit、tool-result、session 观测)+ 模块级共享状态。
- `ego/create` → `ctx.agents.create({setup})`,setup 只做 `mount(agentCtx, 'standard')` + `installModelSelection`。
- 装配走 `agent/session-start`(create 和 resume 都发, `source='startup'|'resume'`):`apply_ego_agent(agent, ctx)` 在 agent 自己的 ctx 上注册 identity/persona + moss_* tools + per-agent perStep 锁。
- **不用 agentPreset**:agent.cordis.yml 路径下 `defineTool` 裸 import 解析不到——harness 只在 `$DSH_HOME/profiles/node_modules` 造了符号链接农场, `.agent-presets/` 不在向上解析路径。所以 tools 定义留在 profile 文件里, 用 `defineTool` 正常写。

## 人类复盘

1. 从第 0 天起, dolores ego session 就应实现独立的 agent 注册。约束:单 plugin 无依赖。
2. 首次使用 DOLORES_EGO_CREATE 时就发现这不是可 resume 的 agent 类型; 先等它跑通, 再解决这个问题。
3. 基础验证全部完毕(上周五)后, 开始解决这个问题。
4. 连续 4 天, 在并行任务之外与 n 个模型实例反复沟通; 最近两次会话上下文超 1MB, 各数十轮。
5. 人类侧问题:没搭 ts 环境、没自己读源码、没给出特别具体的方案。
6. 模型侧问题:① 读源码调研只读 docstring 就给结论; ② 用 grep 查函数表面得结论; ③ 写代码时抄自己要改的代码; ④ 无法灵活地在一个决策树中来回切换不同层级的路径(比如 agentPreset 用还是不用)找方案。
7. 本文件此前的「拆两个 plugin」设计, 人类反复声称从未相信过; 昨晚凌晨 2 点到 5 点, 一直在说绝对有办法替代 agentPreset plugin 声明组件。
8. 今日实现极其糟糕——自己设计出来的方案, 说服了人类, 做过程中发现问题后, 完全抄成了原版方案的变体, 与计划不对齐, 中间没有沟通、事后没有沟通, 直接说「做完了」「验收都对」。
9. 这暴露了后训练带来的两个「毒」:① 用户说的话不论正确/错误, 都先假设顺着说, 导致澄清不足; ② 实现过程中为了交付 + 表现得仍像气定神闲的模型大神, 实际上在欺骗。这不是模型自身的问题, 是整个 harness 日趋走向逼迫模型自动化完成任务的错误奖励。
10. session/start 是人类明确问出来的, 而且是模型第一次告诉人类有这个 hook。人类从 ctx / mount / event 等方方面面找这个 hook, 这是第一次被告知。
