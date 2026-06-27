---
arguments: ''
description: "show_moshi 导演：管理章节推进"
executable: uv
respawn: false
script: main.py
workers: 1
---

Moshi — show_moshi 导演 App。启动时扫描可用课程，
Ghost 渐进式进入：课程列表 → _meta 概述 → 逐章推进。
通过 context_messages 被动推送上下文，无需 Ghost 主动查询。