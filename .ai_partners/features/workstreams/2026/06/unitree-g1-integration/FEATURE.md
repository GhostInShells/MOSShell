---
title: Unitree G1 Integration
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-04
depends: []
milestone:
description: >-
  将 Unitree G1 人形机器人通过 unitree_sdk2_python 集成到 MOSS，作为 bodies app 提供 CTML 可调用的全身运动控制、手臂操作和音频交互能力。
---

# Unitree G1 Integration

## Motivation

继 Reachy Mini 之后，G1 是 MOSS 接入的第二个人形机器人平台。与桌面级 Reachy Mini 不同，G1 是 1.3m 全尺寸人形机器人，拥有 23-43 个自由度、DDS 通讯总线、高低两级控制 API。这次集成验证 MOSS 的 app 模式在更大规模机器人平台上的可迁移性。

SDK (unitree_sdk2_python) 已 clone 到 app 的 `src/` 目录。

## Design Index

- 开发计划: `README.md` (app 根目录)
- 应用说明: `APP.md`
- SDK 分析: `.discuss/` (按需)

## Key Decisions

### 目录命名: `g1` 而非 `unitree_g1`

Unitree 是厂商名，G1 是型号。app 寻址为 `bodies/g1`，简洁且避免厂商绑定。同一厂商的其他型号 (H1, H2) 可以是 `bodies/h1`, `bodies/h2`。

### SDK 捆绑在 src/ 而非 submodule

当前阶段直接 clone 进 src/，随项目提交。不做盲开发——先理解 SDK 的接口、通讯模型和 macOS 兼容性。后续如果 SDK 需要长期跟随上游更新，再改造为 submodule。

### APP.md 是应用说明，README.md 是开发说明

APP.md 面向使用者/模型：这个 app 是什么、提供什么能力、怎么调用。README.md 面向开发者：当前阶段、设计决策、进度。

## Implementation Notes

- macOS 上 cyclonedds 需要从 C 源码编译，是第一个技术障碍
- 参考模式: Reachy Mini integration (FEATURE.md 已完成)
