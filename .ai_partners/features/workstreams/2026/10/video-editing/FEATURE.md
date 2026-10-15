---
title: Video Editing
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-10-15
updated: 2026-10-15
depends: []
milestone:
description: >-
  把素材（语音片段、画面）做成一条可对话、可回滚的时序计划，并把它播出来 ——
  兑现 CTML 最初的动机：剪视频。
---

# Video Editing

> Use `moss features set-status video-editing <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

**这不是一个新命题，是一笔没兑现的账。**

CTML 最初的动机就是剪视频：2023 年为了让模型能在对话过程中直接剪辑，围绕剪映的使用方式
设计了一版结构化语法 —— 一个时序规划语言，天然支持多轨与排序，且**不等剪完才呈现**，
而是动态安排数据结构、逐段播出。2024 年独立实现时意识到它是通用方案：把"时序拓扑规划"抽出来，
它就不只服务视频剪辑。于是它升格成了整个 MOSS 的控制语言。

**升格完成的部分不用再做。** 剩下的是最初那个具体问题：**用它去剪一次视频。**

**为什么现在建**：它服务 stage 3，先以 draft 落位，避免这块判断只留在对话里。

**起点判据（真实痛点）**：

- 通用剪辑工具（剪映一类）做不到"朴素"的效果 —— 它们的**时间轴是像素**。
- 想要的形态：用语音片段的 ASR 分句充当时间轴，中间片段**加速**而不是删掉，个别片段不加速；
  整条流程可以用语音交互。
- 通用工具的 AI 是**以工具形式嵌套**在像素时间轴上，而不是以模型为主的能力。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

- **（2026-10-15）名字取具体问题，不取抽象。**
  一度提议叫「时序编辑」，被否：时序规划已经是 CTML 自身的性质、是全项目的中心命题，
  再为它建一条 feature 属于 authoring debt + document debt。本 feature 的名字必须指向
  **视频剪辑这个应用**。

- **（2026-10-15）形态是"一边对话一边剪"，不是"一次剪完"。**
  视频在放，人和模型同时可以说话，可以指定位置播放，物料逐渐往系统上加。
  判据：**过程本身就是产品**，不是交付前的准备。

- **（2026-10-15）剪辑状态必须是数据，不是操作序列。**
  一份可读可写的计划（人和模型都能改、都能读回），而不是一串"删除 / 加速"的动作。
  否则对话结束后，没人说得清这条片子为什么长这样。
  （同形判例：屏幕池序从"插入序的副作用"改成了模型写的显式状态。）

- **（2026-10-15）播放器只是计划的一个渲染器，所以先后顺序很重要。**
  **先定计划的数据结构，再做播放器。** 反过来，播放器的天然接口是"seek 到第 N 秒"，
  会自然长回像素时间轴 —— 那正是要摆脱的东西。

## Exploration paths

<!-- 未开始。 -->

## Methods

<!-- 未开始。 -->

## Implementation Notes

- **待证（进入 in-progress 前第一件事）**：CTML 现有的"多轨"是 **channel 树的并行 + scope 的
  all / any**。它能不能直接当编辑的多轨用，**没有验证过**，不能当既成事实。
- **真正的技术分歧点**：CTML 是发射即调度、只往前跑；剪辑要能**回退**，要能对**已经播过的一段**
  重新决策。所以本 feature 的核心问题是 —— 一份 CTML 语义的、**可回滚的**、人和模型都能改的计划，
  长什么样。
