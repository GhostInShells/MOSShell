---
title: Reflex Gui Channel
status: draft
priority: P2
created: 2026-06-09
updated: 2026-06-09
depends: []
milestone:
description: >-
  Brief one-line summary of what this feature is about.
---

# Reflex Gui Channel

> Use `moss features set-status reflex-gui-channel <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

将现有的 `moss-in-reflex` 项目迁移并集成为 MOSS 的原生 GUI Channel，使 Ghost 能够通过 Reflex 构建的 Web GUI 进行实时交互，并在此基础上构建多媒体生成（图/视频）和备课讲课全链路能力。

## Tasks

| # | 任务 | 状态 |
|---|---|---|
| 1 | 移动 moss-in-reflex 项目到本项目的 app 里 | completed |
| 2 | 适配运行 reflex 和 text-to-image | completed，待 ghost 端到端测试 |
| 3 | 完成 text-to-video 和 image-to-video | genkits/video 已新建，未开始 |
| 4 | 完成备课讲课全链路能力 | pending |
| 5 | 新建 mode reflex，待人类工程师端到端测试 | webview mode 已新建 |

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### 1. 移动 moss-in-reflex 项目到本项目的 app 里

- 将外部 `moss-in-reflex` 仓库整合进 `.moss_ws/apps/ui/reflex/`
- 目录结构、依赖（`pyproject.toml`）、manifest 注册需与 MOSS app 规范对齐
- 目标：Ghost 退出时优雅结束 reflex 进程，无端口残留和孤儿进程

### 2. 适配运行 reflex 和 text-to-image

- 确保 reflex 前端在 MOSS 上下文中正常启动和路由
- 实现文生图 CTML 命令，绑定图像生成后端（如 Diffusers / API）
- 前端展示组件与后端异步生成状态同步

### 3. 完成 text-to-video 和 image-to-video

- 在 reflex app 中扩展视频生成能力
- 支持文生视频和图生视频两种模式
- 处理视频模型的异步调用、进度反馈、结果展示

### 4. 完成备课讲课全链路能力

- 构建备课 → 讲课 → 归档的完整闭环
- 课程数据模型设计（课件内容、章节、素材）
- 讲课交互：实时演示、GUI 控制、Ghost 语音/文本联动
- 课后归档：记录、回放或导出

### 5. 新建 mode reflex，待人类工程师端到端测试

- 创建独立的 `reflex` mode，隔离 GUI 开发环境的配置与依赖
- 该 mode 需经人类工程师完整跑通端到端流程后标记完成

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->
