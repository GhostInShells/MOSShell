---
date: 2026-09-17
title: Dolores Ghost 原型装线收官 — memento / ground / dsh 融合闭环
feature: ghost-prototype-dolores
model: deepseek-flash
---

# Dolores Ghost 原型装线收官 — memento / ground / dsh 融合闭环

Dolores Ghost 原型的三块核心能力全部装线完成并实机跑通：**memento**（可追溯记忆轨迹）、
**ground**（ghost_home 认知场）、**dsh 融合**（DeepSeek Harness 推理中枢）。ghost 现在有
「记忆 + 身体 + 推理」的完整闭环：dsh 做思考锚点，memento 做持久化轨迹（commit / note /
read / chat / branch），ground 做认知场。`moss-ghost run` 实机验证跑通 —— ghost 能自述其
memento 状态、读历史 commit、正常多轮交互。

## Context

前几个里程碑依次打通了链路：08-28 外部唤醒「四跳」、09-05 首次流畅交互回合、09-13 视觉
闭环。本轮是「功能装线收官」——把此前散落的三块能力拼成完整闭环：memento 旁路 note 机制
（轨迹接续 + 显式约束）、ghost 反身 channel（memento 只读面）、memories（ground + branch
view 容器化）。这一轮不止加代码，而是第一次端到端实机验证整条链。

## What happened

- **memento 旁路 note 机制**：`schedule_note` → 旁路单轮（源 session 冷 seed）→ 写回 note。
  prompt 改为轨迹接续（注入前驱坐标 + 最近一条已就绪 commit 消息作前文），约束显式进载荷
  （低思考模式 + maxTokens 硬 cap），不再靠 plugin 身份判定间接降级。
- **ghost 反身 channel**：`memento` channel 提供 view / read / history / chat 四个读接口
  （`always_observe=True`），ghost 能看自己的轨迹线、读某段原文、和某段上下文对话。
- **memories 容器化**：ground 渲染 + memento branch view 折进一个 `<memory>` 容器；epoch
  在首次 thinking 打开（此时 baseline 才含 ghost channel）。
- **提示词分层**：系统指令加 `## Memento — Your Traceable Memory` 讲概念，channel 只讲
  控制面（坐标 + 命令），两者不再重复。
- **实机验证 + 两个真 bug 修复**：
  1. observe continuation 的 `observe continuation: N` 纯文字被 `source: user` 标成用户
     输入，模型误以为人类说话 → 改空 content + plugin source（只唤醒、不塞字）。
  2. 解析错误回灌缺「修正而非讲解」的 directive → `InterpretError.model_facing_message()`
     追加 fix-not-explain + CDATA 提醒。
- **噪音收敛**：memento notice 只留游标（branch / commits / latest 坐标），去掉标题；
  committed notice 改回执坐标（不透露 turn 区间）；中文返回值统一英文。

## Significance

1. **三块拼成闭环** — Dolores 同时具备「记忆（memento 轨迹）」「身体（ground 认知场）」
   「推理（dsh 中枢）」，是第一个三块都装线完的原型，而非纸面设计。
2. **实机验证而非纸面装线** — 本轮经 `moss-ghost run` 实机跑通，ghost 能自述 memento 状态、
   读历史 commit、报告提示词状态；发现的毒（observe continuation 冒充用户输入）正是实机
   才暴露的，验证了 dogfooding 的价值。
3. **memento 旁路机制定型** — commit / note / read / chat / branch 全链落地，轨迹接续的
   note prompt 是下一阶段 compact / branch checkout 的基元。

## Next

- 小 bug 较多，放下阶段：wait_next_moment（yield）与 dsh 界面排队摩擦、observe continuation
  机制本身的噪音风险、note prompt 的 prior 与 seed 可能的二次包含（已降为最近一条）。
- 自迭代能力地图（ground + 认知工具）是下一阶段重点，已钉入
  `dolores-self-iteration-map.md`。
- stage 2 复盘（8 月，超期 17 日）→ 发 dev2 → 下阶段 rc。
