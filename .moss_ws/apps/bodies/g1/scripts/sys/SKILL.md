---
name: g1-sys-investigation
description: G1 系统线实验 — 硬件事实摸底。在 PC2 (Jetson Orin NX) 上本地执行，每个技能验证一组原子命题，结论直接影响架构决策。
---

# G1 系统线实验

回答一个问题：**PC2 上有什么，能做什么？** — 在写任何 SDK 代码之前。

## 技能索引

| 技能 | 目录 | 调查命题 | 关键决策 |
|------|------|---------|---------|
| network | `network/` | 网络接口、三节点连通、WiFi 状态 | 通讯路径选择 |
| audio | `audio/` | 播放/录音设备、蓝牙硬件、实际发声 | PC2 蓝牙 vs PC1 API |
| usb_camera | `usb_camera/` | USB 接口暴露、摄像头驱动 | 视觉方案选择 |
| system | `system/` | Jetson 型号、资源、Python/moss 环境 | 运行环境基准确认 |
| dds | `dds/` | cyclonedds 包、环境变量、共享内存 | DDS 通讯就绪状态 |
| performance | `performance/` | 进程树、CPU/内存/IO、idle 基线 | 资源约束判断 |

## 使用

```bash
# 每个技能独立执行
cd network && bash 01_interfaces.sh

# 全量执行（按依赖顺序）
for skill in network audio usb_camera system dds performance; do
  for script in $skill/*.sh; do bash "$script"; done
done
```

## 执行环境

所有脚本在 G1 PC2 (Jetson Orin NX, 192.168.123.164) 上本地执行。不经过 MOSS channel 体系。

## 与 SDK 线的关系

系统线先于 SDK 线。系统线回答"PC2 有什么"→ SDK 线回答"SDK 提供了什么 API"→ 两条线交汇在阶段 E 的验证脚本。
