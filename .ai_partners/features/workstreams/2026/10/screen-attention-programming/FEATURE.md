---
created: 2026-10-19
depends: []
description: screen 的注意力编程 —— 让 node 用 webview notify 举手，screen 以弹幕 + 时间序抽屉 把注意力送到该看的那块砖。模型只管用命令，通知的时机、地址、去重、摘要都由机制承担。
milestone: null
priority: P1
status: in-progress
status_note: D19 与 thirdgerb 对齐：notify 由命令自签发、带 address、screen 飞弹幕 + 时间序抽屉、限流 2-3
  秒、摘要分机械/动机两半、抽屉与 home 同池两读法
title: Screen Attention Programming
updated: '2026-10-19'
---

# Screen Attention Programming

> Use `moss features set-status screen-attention-programming <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss-os-control` 与 `screen-manager` 一路膨胀，新东西往里塞会把后续迭代埋掉：屏幕这一层已经
到了该"收尾定版 + 另起一条主线"的时候。而另起的这条主线不是加窗口语义，是加**注意力语义**。

要解决的问题是 thirdgerb 在 D19 说的那句：*"之前你有时候去改了 terminal，发了东西，我其实完全
不知道。"* 反过来也成立 —— 屏幕上发生的事，模型与人各自只看见自己那一半。缺的不是更多窗口，
是一条"谁刚动过"的通道。

## 哲学背景：注意力是有限的（D19 thirdgerb 口述）

- **屏幕是注意力的分发面。** 人的注意力是有限资源，屏幕上出现的每一样东西都在争夺它。
  所以母题是「怎么用屏幕去用好有限的注意力」，notify 只是这一章的第一个小节。
- **传统 Web 应用的病：注意力被压在一张静态平面上。** DSH、VS Code 以及这一类软件仍然
  用传统 Web 应用的思路构建：能力必须收口到平面上固定的位置，于是只能不断压缩注意力的信息，
  而且没法把注意力信息**分类管理**。
- **MOSS 的路线：注意力用动态的能力呈现。** 不把信息塞进固定位置，而是让「当下什么值得看」
  成为可以编排、随时间变化的东西 —— 这正是当初要做 CTML 这种复杂交互语言的目标之一。
- **真正要解决的命题：人机共享同一个物理世界。** 我看到的东西和你看到的东西，虽然你这边是
  文本加截屏，但要达到某种同构；同构成立，协作才成立。
- **本节范围：** 注意力编程是很大的一个章节，D19 只谈 notify 这一部分，其余几部分过几天再展开。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

D19 与 thirdgerb 对齐（现场直播录制，画板同步画图）。

1. **notify 挂在 webview service 协议上，由命令自己签发。** 模型照常用命令，不管理"什么时候
   该发通知"。screen 收到就出效果。模型侧零心智负担是这条的唯一目的。
2. **notify 必须带 address。** 地址由组件自己产出（terminal 的卡片 id、编辑器的 thread 名等），
   随命令返回给模型，模型原样传回。**不让模型拼地址** —— 拼必错，且错了不报错，只是点了没反应。
3. **效果一：弹幕。** screen 每收到一条 notify 飞一条弹幕（已有 danmaku 机制，不新造）。
4. **效果二：抽屉是排序，不是队列。** 右上角抽屉按"最近被通知过"排序，一个目标只占一条，
   时间戳刷新即位置刷新。等同于即时通讯软件的会话列表 —— 点击即把注意力送到那块砖。
5. **限流：同一目标 2–3 秒最多一条弹幕。** 被限流的通知不丢信息，仍然刷新排序与时间戳；
   弹幕只是视觉，抽屉才是状态。
6. **摘要分两半。** 机械那半（做了什么、改了几行）由命令自动生成，模型不写；动机那半来自
   模型给的 `label`。沿用 file_editor / terminal 已有的 card 规矩，不新造一套。
7. **抽屉与 home 是同池两读法。** home 是摆位（模型安排的构图），抽屉是时间序（谁刚动过）。
   点抽屉里的一条**不改摆位**，只送注意力 —— 因为人可能正看着另一组。
8. **砖的边框动效（转圈）= 伏笔接口。** thirdgerb 明说这个特效后面有个重要的东西要用上它，
   所以它要做成一个可复用的"这块砖在动"的表达，而不是一次性装饰。
9. **保留要有分级。** 报错类记录不得与普通记录同寿。
10. **点开一个已经关掉的窗口，必须有话说**，不能点了没反应。

## Exploration paths

- 这两个交互概念（notify→弹幕、抽屉→跳转）在 screen 早期就被设计过，因为当时实现成本高被砍。
  现在基线基本完成，重新加回来。第一版砍掉的判断不必推翻，但要记住"砍掉的是实现成本，不是设计"。
- 曾考虑把通知做成一个独立的"model 发通知"的命令。否决：那样模型要自己记着发，等价于把机制
  的成本转嫁给上下文，且必然漏。

## Methods

<!-- Non-obvious implementation patterns. -->

## Implementation Notes

- 弹幕与"我说话"的弹幕**在同一块屏上飞**，视觉上必须分得开：观众要能分辨哪句是 ghost 在说、
  哪件是机器在报。
- 地址的产出方是组件，不是屏幕；屏幕只做"按地址找砖"，找不到时要报告而不是静默。