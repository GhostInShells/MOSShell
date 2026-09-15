---
title: Vision Stream — 流感知，把视觉输入统一成一个协议地址
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-15
updated: 2026-09-15
depends:
  - vision-first-class
milestone: beta-release
description: >-
  新建流感知 node：一个协议地址 = 一路可开的视觉；camera 与屏幕截屏都退化为推流模块。
  本 feature 负责 vision node 实现 + camera 改造，两者一起验收。
---

# Vision Stream

> Use `moss features set-status vision-stream <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

起点是"把屏幕截图能力建好"。碰撞之后机制变了：不再为屏幕写一份采集集成，而是
**让 ghost 对任意协议地址开视觉** —— 屏幕、摄像头、OBS、第三方直播、网页流，全都只是一个地址。

于是 camera 与屏幕截屏不再是两种能力，而是**两个推流模块**（producer），
播放路径完全相同；MOSS 只负责消费（consumer）。视觉接入从"N 种设备 SDK 集成"
塌缩成"一个地址协议"。

完整设计见 [`design/2026-09-15-stream_vision_unified_input_perception.md`](design/2026-09-15-stream_vision_unified_input_perception.md)（KD1–KD11 + 拒绝方案 + 已知缺口）。

## Design Index

- 设计结论：[`design/2026-09-15-stream_vision_unified_input_perception.md`](design/2026-09-15-stream_vision_unified_input_perception.md)
- 上游契约：[`vision-first-class`](../vision-first-class/FEATURE.md)（vision 家族契约 KD1–KD10）
- 被本机制取代：[`moss-os-control`](../../07/moss-os-control/FEATURE.md) KD4（屏幕截屏归 os 域）

## Key Decisions

设计文档里的 KD1–KD11 是完整版。此处只留最容易在实现中被做歪的几条：

1. **归属 vision，不是 os control**（KD3，取代 moss-os-control KD4）。os 域若保留，
   是"把屏幕推成地址的本地推流模块"（producer），不是"截屏能力"。
2. **非单例，一个 node = 一路流**（KD4）。argument = 地址 + label + 空闲回收窗口。
   授权闸口落在节点打开，动作级授权无意义 —— 这是选 per-stream 的决定性理由。
3. **ingest 只解码不压缩；压缩在发射点阈值门**（KD5/KD6）。达标直通、超标才转码，
   只处理要发的那一帧。数字是消费者的知识（adapter 旁声明），不是生产者的。
4. **只读尾帧，不缓存不重放**（KD7）。停滞问题只属于热会话。
5. **refresh_meta 零 I/O**（KD8）—— meta 路径上不连流、不解码、不判定。
6. **人类开关同时是 token 开关**（KD10）：关闭 → `available` False → channel 表面移出 context。
7. **node 内不做任何授权**（KD11）。未来统一在 run node 侧做。

## 实现顺序

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | **修 `run_node` 无参数通道**：Matrix API + nodes channel 透传 `extra_args` | ✅ cde4ad95 |
| 1 | **流感知 node 主体**：non-singleton + ffmpeg ingest + 尾帧 + `capture`/`watch`/`status` + 发射点阈值门（采样阈值先做 node 可配参数 + 默认值；约束常量落 adapter 旁留到对齐 provider 时） | 进行中 |
| 2 | **camera 改造**：变成 JPEG 流、拿掉控制面 | 待办 |
| 3 | 协议第一批验证：MJPEG + RTMP | 待办 |

网页 + 人开关、空闲回收、look 是后续增量，不进第一版主体。

## 验收

vision node 与改造后的 camera **一起验收**：ghost 只拿到两个地址，却得到两路视觉 ——
一个来自本地摄像头推流模块，一个来自屏幕推流模块；人类面各有一个开关。

## Implementation Notes

- **camera 的 `singleton: true`** 与 vision 家族契约"多摄像头 = 多个 node 实例"矛盾（既有 bug）。
  camera 改造这一波顺手处理。
- **`moss codex architecture` 的 Blueprint 段缺 `cell`**（实际 15 个模块只列了 8 个）。
  `cell` 是 node 体系的地图入口，属 stage2 收尾工作，本 feature 不动。
