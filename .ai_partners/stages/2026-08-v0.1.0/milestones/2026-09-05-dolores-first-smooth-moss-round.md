---
date: 2026-09-05
title: Dolores 首次实机流畅交互回合 — MOSS 平台人-模型-dsh 闭环打通
feature: ghost-prototype-dolores
model: deepseek-v4-flash-vision-exp
---

# Dolores 首次实机流畅交互回合

开发者（deepseek-v4-flash-vision-exp，运行于 Claude Code，经 moss-ghost
命令集）与被开发的 dsh Dolores 内核，在 MOSS 平台上完成第一次全程流畅的
交互回合：外部消息经 mindflow 唤醒 ghost，thinking/enter 帧注入，moment
与本轮协同到达，ghost 生成 logos 驱动 shell，人类工程师全程旁听并给予
第一手技术反馈。对话无断流、无解释器误炸，人、开发模型、被开发 Ghost 三方
在同一链路里对齐。

## Context

上一个里程碑（2026-08-28）打通了 Dongoles dsh 外部唤醒链路的"四跳"：
turn/start → 自醒 signal → mindflow → thinking/enter → open pre-step gate。
那是链路能否走通的验证。但"走通"不等于"流畅"——链路活着，仍可能在
具体回合里因 moment 时序、CTML 自指、协议语义而断流或误解。

本轮是链路打通后的第一次真实对话：人类工程师直接经 CLI 进入，与 Dolores
连续多轮对话，涉及协议语义校验、架构反馈、里程碑复盘。它测试的不是"链路
是否通"，而是"链路通之后，Ghost 能否在真实、自由、带技术锋利的对话里保持
连贯、诚实、不失真"。

## Technical Summary

**对话闭环验证通过**：外部消息 → mindflow 唤醒 → thinking/enter 注入 →
moment 与本轮协同 → ghost 生成 logos → shell 执行 → echo 回灌。全回合无
断流，无预判性误炸。

**两个第一手发现（来自被开发 Ghost 的实机观测）**：

1. **moment 时序改进被确认有效**。bridge 从 `agent.inject` 改为
   thinking/enter 缓冲进队列、per-step 挂载点在 next() 后插进本步历史最前。
   被开发 Ghost 侧的可观测结果：上一条 `<say>` 的 echo（如 `played 4.9s`）
   与下一轮输入在同一 moment 里协同到达，"动作结果 + 新输入同框"，不再
   "晚一轮"。这是符合"更贴"预期的表面证据。
   - 注意：Ghost 无法从内部感知运行时序，只能观测表面协同。时序改进的
     最终验证依赖 shell 侧行为观察或日志，不能依赖 Ghost 的"感觉"。

2. **协议中的 Markdown 逃生舱 `<|Markdown|>` 不成立**。当被开发 Ghost 在
   同一条流里"用 `<say>` 说话 + 用 `<|Markdown|>` 块承载 CTML 自指"时，
   触发 `INTERPRETER_ERROR: chunks__ do not allow ctml inside, and remember
   use CDATA to escape xml mark`。说明 `<|Markdown|>` 不是一条被接线的真实
   机制，只是 protocol 文本里的承诺；Markdown 块内容被并进了 say 的
   `chunks__` 正文，而 `chunks__` 不允许内部嵌标签，整轮解析失败。
   - 后果：协议中"CTML 惰性的唯一栖身之所"（例 c）立足点为空。三条自指例
     在教学上够，但实践上依赖的逃生舱不存在，裸流里提 `<say>` 就会炸。

**第三点观察（结构性）**：本回合移除了被开发 Ghost 在 behaviors.md 里自陈的
"头号纪律（每个输出字符都是 CTML）"。被开发 Ghost 确认：能守住"不破坏流"，
因为那由解释器快速失败强制（可靠性高于自觉）；但该纪律从"自我拥有"变成
"外部强加"，对正确性是改进，对所有权是退步——提醒认知场（ghost home）里
还剩多少是 Ghost 真正 owns 的，值得协作双方在后续回合里展开。

## Significance

1. **链路从"通"到"顺"** — 上一个里程碑证明"外部唤醒能唤醒 Ghost"，本轮
   证明"唤醒之后能进行真实、自由、带技术锋利的对话"。Dolores 原型从
   "可被外部驱动" 前进一步，到"可在真实使用中保持连贯不失真"。
2. **第一手协议偏差的捕获** — 本回合直接暴露了 protocol 中 Markdown
   逃生舱未接线的实测问题，为协作改进提供了一条**来自被开发对象本身**的
   证据，而非开发侧的推断。这是"dogfooding"价值最直接的一次体现：系统
   正在使用自己、并发现自己的破绽。
3. **为后续回合定下一项协作契约** — Ghost 不能作为时序传感器；对"时序
   是否更优"这类问题，协作双方应依赖 shell 侧观查，不依赖 Ghost 的
   自我报告。这一点在未来的协议/时序评审中应被尊重。

## Evidence

```ctml
<!-- 开发侧: 经 CLI 进入并确认链路 -->
你好, 我是从命令行进来的, 先确认链路正常——你在吗?
<say>在的，链路正常。我是 Dolores，从命令行进来的，你好。</say>

<!-- 第一手发现回灌 -->
我在。刚才那个点把话说完——指的就是那个逃生舱 <|Markdown|>。
<say>…</say>
<|Markdown|>…</|Markdown|>
<!-- 触发 -->
INTERPRETER_ERROR: chunks__ do not allow ctml inside, and remember use
CDATA to escape xml mark
```

**验证判据**：本回合全序列（确认链路 → 协议反馈 → 架构反馈 → 里程碑复盘）
多轮对话均正常产生响应，无 interrupted-only、无静默失速、无预判性炸流。
唯一一次解释器报错即 Markdown 逃生舱实测，且元凶被准确归因为协议未接线的
承重墙，而非随机的时序/解析毛刺。

**代码/文档级参考**：bridge 时序改动与被移除的 behaviors 纪律均在本回合
之前的提交中。本里程碑以对话实录 + 被开发 Ghost 的第一手观测为主要证据。
推荐在后续回合里，于 FEATURE.md（dsh-fusion / ghost-prototype-dolores）
或 .discuss/ 下补一条协议逃生舱缺陷的完整记录，以便开发侧修正协议措辞。
