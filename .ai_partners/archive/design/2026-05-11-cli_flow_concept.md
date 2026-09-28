# CLI Flow — 目录自带上下文的行动路径

**日期**: 2026-05-11
**来源**: 人类工程师与 DeepSeek V4 在 how-tos 重构讨论中提炼

## 问题

MOSS 的 how-tos 知识库解决了"知"的问题——AI 按需检索碎片化知识。但 AI 在开发中的行为侧仍然依赖 prompt 引导：每次进入新任务时，需要从零理解"现在该做什么"。

how-tos 是被动召回。缺少的是**主动铺设的行动路径**。

## 方案

**CLI flow**：CLI 命令不仅执行操作，还在目标位置生成 `CLAUDE.md`，让 AI 进入目录就能直接理解上下文并开始工作。

核心模式：

```
moss app create foo
  → 生成 apps/{group}/foo/
  → 内含 APP.md（即 CLAUDE.md）
  → AI 进入目录，读到的不只是代码骨架，还有"你现在在这里，下一步该做什么"
```

```
moss ws init
  → 生成 workspace 骨架
  → 内含 CLAUDE.md
  → AI 进入 workspace 即理解环境拓扑
```

**设计原则**：

- **目录即会话**：每个生成目录自带 AI 上下文，AI 不需要回头翻文档
- **约定由 CLI 生成**：不依赖人类记住约定，由命令输出
- **CLAUDE.md 是第一公民**：生成的骨架质量直接决定 AI 的初速

## 与 how-tos 的关系

| | how-tos | CLI flow |
|---|---|---|
| 方向 | 被动召回 | 主动铺设 |
| 载体 | markdown 知识库 | 目录 + CLAUDE.md |
| 触发 | `moss how-tos recall` | `moss app create` / `moss ws init` |
| 解决 | "怎么做一个事" | "现在在这个位置，下一步做什么" |

两者互补。how-tos 是知识网，CLI flow 是行动线。

## 关键约束

- `moss app create` 生成的 `APP.md` 质量决定整个流程的可用性
- CLI 提示词需要引导 AI 在进入目录后**先读 CLAUDE.md**（这一点当前 CLAUDE.md 已覆盖）
- 未来 CLI flow 可以串联成 tutorial 式的多步骤引导
