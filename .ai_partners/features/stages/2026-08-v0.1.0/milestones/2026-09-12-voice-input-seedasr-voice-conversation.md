---
date: 2026-09-12
title: 用语音和 Dolores 对话 — seedasr 重做版闭环
feature: voice-input-state-machine
model: deepseek-v4-flash
---

# 用语音和 Dolores 对话 — seedasr 重做版闭环

这一版用语音和 Dolores 对话了：说话 → 识别 → listener signal → mindflow 唤醒
Dolores → 它听见并回话。语音控制本身去年就已打通，本轮是**重做**——按官方 SDK 重写成 `volcengine_sauc`（豆包 2.0
接口），重新走到「能对话」这一步。

## Context

本轮用 seedasr（豆包 2.0 接口）重构 ASR：人类工程师拉取官方 SDK 与文档，
逐项对齐语义，模型执行。语音控制去年已打通，本轮是把 ASR 这一层换成按官方
协议建模的 seedasr。

## What happened

用语音和 Dolores 完成了一次对话。`moss audio listen -m once` 实机「说一句 →
自动识别 → 切段 → 退出」，signal 触达 Dolores，Dolores 听见并回话。

## Evidence

```text
$ moss audio listen -m once
  你好，这是我第一次尝试用语音给你发消息。
---
  [tail] 你好，这是我第一次尝试用语音给你发消息。
✓ session done: 5.5s, 1 turns, 1 clauses
```

Dolores 侧收到 `<inputs>你好，这是我第一次尝试用语音给你发消息。</inputs>`，
回话 `<say>你好呀，我第一次听见你的声音。</say>`。

## Significance

1. **用语音和 Dolores 对话了** — 这一版重做后，语音对话重新跑通。基于完整的 ListenerNucleus + DoloresGhost 实现语音对话. 

## Next

旧 `volcengine_asr/` 待退役；其余 CLI 命令待验证；类名/路径名待手动改为
seedasr。详见 voice-input-state-machine FEATURE.md「2026-09-12 会话决策」。
