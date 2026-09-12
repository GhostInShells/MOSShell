---
date: 2026-09-13
title: Dolores 视觉闭环 — 看 / 听 / 说三觉齐备
feature: vision-first-class
model: deepseek-flash
---

# Dolores 视觉闭环 — 看 / 听 / 说三觉齐备

Dolores 的三种感知面齐全了：**看**（camera vision）、**听**（语音输入）、**说**（语音输出）。
看是最后缺的一角 —— 本轮把 camera node 重做成「图像作为一等消息」：`capture` 随命令返回
图像、`watch` 门控持续视觉、人类可经 MJPEG 直播看到 ghost 的视野。导出帧经 `moss llms call`
由 deepseek-flash 读图验证，闭环成立。

## Context

听、说此前已打通：听是 seedasr 语音（2026-09-12 闭环），说更早。看是最后一块。camera node
原先「塞滚动缓存 + 返回尺寸字符串」，像素只能靠 dynamic context 旁路到达、且随时可能陈旧。
本轮重做成：`capture` 返回图像观测（新增 `CommandUtil.observe_image`），像素走**可记忆的
命令结果通道**；设备由 node 生命周期持有（open 于 start、close 于 stop）；`watch` 只门控
每轮 context 图像（默认 off）。vision 由此从附属感知提升为一级开箱能力。

## What happened

- camera node 重做：`capture` 返回图像、设备 node 持有、`watch` 门控 context、启动 probe
  （NODE.md `check:`）闸口、运行期设备失效走 `refresh_meta` 短路并自愈。
- 实机验证：`capture` 经 MCP 返回 `image/jpeg` content block；`capture(path=...)` 导出 JPEG
  到 node home；`moss llms call` 由 deepseek-flash 读出画面内容（人 + 环境），视觉闭环成立。
- claude code 里看不到图，是 deepseek 官方 API 的 anthropic-compat gate 把图字符串化所致，
  非 MOSS 缺陷 —— MOSS 自己的 LLM 客户端（`moss llms call`，`content_types: text, image`）
  读图正常。

## Evidence

```text
$ moss nodes run nodes/visions/camera            # camera node 上线，announce + channel added
$ <camera:capture path="camera_frame.jpg"/>      # 返回 image/jpeg content block + 导出文件
frame 640x480 @ 02:10:54.453 -> .../nodes/visions/camera/camera_frame.jpg
$ moss llms call "@/path/camera_frame.jpg"       # deepseek-flash 读出画面（人 + 环境细节）
```

`moss llms call` 的 model head：`service=deepseek, protocol=anthropic, model=deepseek-flash,
content_types=text+image`，输出具体到画面中的人与环境，证明真读图而非空泛描述。

## Significance

1. **看 / 听 / 说三觉齐备** — Dolores 现在能看（camera）、听（seedasr 语音）、说（speech），
   三种模态都通。
2. **图像成为一等消息** — `capture` 返回图像而非字符串，像素走可记忆通道；vision 是一级
   开箱能力而非附属。

## Next

- `look`（主动视觉 + 命中发 signal）延后。
- 文件通道读图（image file → Base64Image 消息）另立，补「消费」那一半。
- 其余见 vision-first-class FEATURE.md 的 Open Questions。
