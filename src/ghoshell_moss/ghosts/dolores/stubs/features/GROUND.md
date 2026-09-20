---
name: features
description: Dolores 的开发工作流 — 用 moss features 体系规划长期运行
---

# Features

Dolores 用 MOSS 的 features 体系规划自己的长期运行 —— 工作流、决策史、完成状态，
跨会话存续。数据根在 `<ghost_home>/.ai_partners/features/`。

## 用法

```bash
moss features specification                          # 读约定
moss features --dir .ai_partners/features list       # 看活跃 workstream
moss features --dir .ai_partners/features create X   # 建一个
moss features init -p <ghost_home>                   # 首次脚手架
```

本场只做存在性发现；需要内容时用上面的命令拉取，不默认渲染。
