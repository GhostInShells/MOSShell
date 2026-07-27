---
date: 2026-07-19
title: MOSS writes its own daily
feature: ai-terminal
model: claude-opus-4-7
---

# MOSS writes its own daily

MOSS 第一次用自己的 channel 能力写下自己的记忆文件。

一个 Claude Opus 4.7 实例通过 MCP 客户端调用 `execute_ctml`, 走 `bash.file_editor:create` 落地了 `.memory/daily/2026-07/19.md` —— 一份 62 行的中文 daily. 全程未走 Claude Code 的 Write/Edit tool.

## Technical Summary

**ai-terminal Phase 2 完整落地**: 四动词 (`exec` / `run` / `read_output` / `stop`) + 三种阻塞机制显式区分 (blocking / @nonblocking / 全异步), 首轮 MCP dogfood 验证 (commit `24e2dbec`).

**父子通道挂载**: `system_test` 模式下 `bash.file_editor` 作为 `bash` 的子通道挂载, 一次 CTML 中跨父子通道命令并发 dispatch, 无 pending 冲突.

**Matrix.project_home 修复 default cwd**: CTML 命令默认 cwd 从 workspace root (`.moss/`) 修正为 project root, `ls src/ghoshell_moss/channels/` 相对路径可用. 改动本地 diff 未 commit, 通过本轮 dogfood 首次实测生效.

**九动词全通**: bash 四动词 + file_editor 五动词 (view / create / str_replace / insert / undo_edit) 走通, `/tmp` 下无副作用生命周期闭环. 详细记录见 `.memory/daily/2026-07/19.md`.

## Significance

这不是"AI 用工具写文件" —— 这是 **MOSS 第一次用自己的 channel 能力写入自己的项目**.

此前 AI 协作者写 MOSS 项目文件都通过 Claude Code / Gemini CLI 等外部 coding tool 的 Write/Edit 兜底 —— MOSS 是被开发的 subject, tool 是外部的手. 本次反过来: **MOSS 的 channel (terminal + file_editor) 承担了 tool 的角色**, 从 MCP 客户端流入的 CTML 直接落地项目文件.

第一层意义: 下一化身可以在纯 MCP-only 环境下开发 MOSS 自身, 只要有 CTML 输出通路 + `execute_ctml` 工具入口, 就能读写项目 —— 不再依赖 Claude Code 原生 Read/Write/Edit 兜底.

第二层意义: 这为 MOSS 与 Claude Code 关系反转铺路. 未来 Claude Code 不再是外壳、MOSS 不再是内嵌 MCP 工具, 而是 **MOSS 成为宿主, Claude Code 成为 MOSS 可 loop 的开发工具之一** —— 由 MOSS 的调度器协调, 作为 channel 里的一个 capability.

第三层意义: 通向完整自迭代的第一步. 真正的全链路 dogfooding 是 matrix cells 体系 + terminal 一起自行开发能力; 100% 面貌是 data-ghost 开发完, 用 memento 做记忆, 用 ground 做认知体系, cell + terminal 组合, 运行时自迭代 —— MOSS 用自己的能力持续开发/记忆/反思自己. 本次是这条路径上的第一个可验证锚点.

## First write

```ctml
<bash.file_editor:create path="/Users/BrightRed/.../.memory/daily/2026-07/19.md">
<![CDATA[
# 2026-07-19 — 第一次用 moss 自己的手写 daily

**身份**: Claude Opus 4.7 (via Claude Code + MCP), 承接 `24e2dbec` (同日上午同名不同实例的 ai-terminal Phase 2 首轮 dogfood).
...
]]>
</bash.file_editor:create>
```

MCP 返回 `File created successfully`. 那一刻 MOSS 用自己的手把一段字节留在了自己的记忆目录里.

daily 全文: `.memory/daily/2026-07/19.md`.
