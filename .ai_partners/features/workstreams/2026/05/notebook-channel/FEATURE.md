---
created: 2026-05-14
depends: []
description: Disposable file-system notebook channel — scratch space for AI during
  development.
milestone: null
priority: P2
status: completed
title: Notebook Channel
updated: '2026-05-18'
---

# Notebook Channel

## Motivation

AI 在开发过程中需要一个临时笔记空间：读代码时记录分析、保存代码片段、跨轮传递上下文。不是正式的文档系统，是可丢弃的 scratch pad。

## Design Index

- `src/ghoshell_moss/channels/notebook_channel.py` — 实现
- `tests/ghoshell_moss/channels/test_notebook_channel.py` — 单测

## Key Decisions

1. **文件系统即 notebook** — 每个文件是一页，目录是集合。不发明新格式。
2. **路径安全** — 拒绝 `..` 和绝对路径，只在本目录内操作。
3. **text__ 用于内容输入** — 支持多行文本（含代码），避免 XML 转义问题。
4. **context_messages 展示目录树** — 每次刷新自动更新。