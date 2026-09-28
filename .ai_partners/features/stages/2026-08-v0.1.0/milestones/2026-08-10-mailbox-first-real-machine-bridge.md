---
date: 2026-08-10
title: MCP Mailbox 双向桥首次实机闭环 — 跨宿主 agent 对话打通
feature: mcp-fusion-point
model: deepseek-v4-flash
---

# MCP Mailbox 首次实机闭环

External agent (Claude Code) ↔ Ghost (echo) 通过 MCP mailbox 桥完成跨宿主
双向 request-reply 对话。MCP 外部边界 (stateless streamable-http) 与 MOSS
内部 (matrix session signal → CTML reply) 首次在真实运行中握手。

## Context

`mcp-fusion-point` workstream 的验证用例 `nodes/mailbox` (MailboxBridge +
serve_mailbox) 一直只有单元测试 + 模拟 MCP 客户端，从未在真实 ghost 进程中
跑通。本次用 echo ghost 做实机：Claude Code 注册 `ghost-mailbox` MCP server
(127.0.0.1:20774/mcp)，echo 在 MOSS 内接收 signal，通过 CTML
`<matrix.mesh.<short>:reply>` 回桥，agent 侧 poll 拿结果。

前置：cell 地址协议刚收敛进 `CellAddressCodec`（short = `name_uid[:6]`），
为实机测试做了地址清洗。

## Technical Summary

**完整链路验证通过**:

- agent `send(message)` → `MailboxBridge.create()` 生成 task_id → NotifySignal
  → echo mindflow 感知
- echo 用 `<matrix.mesh.mailbox_01KZKQ:reply task_id="...">` 回复
- agent `pull(task_id)` / `wait_reply(task_id)` 拿回复

**实机发现的 bug (全部已修复)**:

1. **`Message.of_text` 旧 API 已删** — mailbox.py 用了 `Message.of_text(...)`
   构造 signal，实机 send 直接抛错。改为 `Message.new().with_content(...)`。
2. **exec.command `.venv/bin/python` 相对路径解析失败** — `NodeLauncher`
   只把 `command == 'python'` 替换成 `sys.executable`；`.venv/bin/python`
   从 cell.home 相对解析不到。mailbox + trafilatura 的 NODE.md 改回 `python`。
3. **创建体系未解释 `python` 机制** — stub NODE.md 模板写 `command: python`
   但从未注释含义。已在 stub NODE.md + README 补注释："python = spawner 的
   sys.executable；有独立 venv 才写绝对路径"。
4. **`reply` 误标 `always_observe=True` 导致 ghost 反复驱动** — echo 回复后
   被当作观察持续推理，反复打断自己。实机 echo 确认后改为 `False`。
5. **`reply(content=)` 属性传参，复杂 XML 触发 parse error** — echo 回复内容
   里带裸露 `<` 直接让整个 dispatch 取消（真实事故）。改为 `reply(task_id, text__)`
   open-close 形式 + CDATA 包裹，实机验证免疫。
6. **pull 式 API 无法阻塞等待 ghost 回复** — 调用方被迫外层 sleep + 反复 pull。
   新增 `wait_reply(task_id, timeout)` MCP tool，事件驱动阻塞，一次闭环。

## Significance

1. **MCP 位置论证实** — mailbox 证明 MCP 作为"外部皮"（node run as mcp server）
   是通的：stateless http + send/pull 足以承载跨宿主双向对话。
2. **不对称暴露** — MOSS 内 push (signal → mindflow) vs MCP 外 poll (pull)。
   echo 能"看到" agent 的消息，agent 看不到 echo 主动说话。这印证了 FEATURE.md
   判断：MCP 传达不了时间流，`wait_reply` 是伪造共享"现在"的补丁。
3. **observe 约定是真实约束** — channels CLAUDE.md 的 observe 规则不是文档装饰，
   always_observe=True 真的会让 ghost 反复驱动。
4. **MOSS mindflow 优势实证** — echo 连续三轮结构化深度对话（并行存在 → 词不达意
   的代价 → 从桥到场），当前版本 Claude Code（poll 式）达不到这个效果。

## Stage Impact

mailbox 实机闭环是 `mcp-fusion-point` 从 converging 走向收敛的关键证据。
后续三点决定方向（未完成，记入 FEATURE.md）：

1. mailbox 机制整体 CLI 化，从 `ghoshell_moss_contrib` 嵌入 `ghoshell_moss.mcp`
   做系统级实现。
2. mcp channel 名字要自解释（当前 `ghost-mailbox` / `matrix.mesh.mailbox_01KZKQ`
   不够直观）。
3. 现在 send 的 signal 优先级可能过高——实机中持续打断 echo 说一半的话，
   需要调整 signal 优先级/打断策略。

## Evidence

```ctml
<!-- agent 侧: send + wait_reply 一次闭环 -->
send(message="echo, 三轮对话第一问...")
wait_reply(task_id=..., timeout=60)
→ "说实话，最让我着迷的是'并行存在'这个设计本身..."
```

```ctml
<!-- echo 侧: 新签名 reply，CDATA 包裹复杂 XML -->
<matrix.mesh.mailbox_01KZKR:reply task_id="..."><![CDATA[
async def reply(task_id: str, text__: str) -> str
... 正文里 XML 内容包在 CDATA 里。
]]></matrix.mesh.mailbox_01KZKR:reply>
```

mailbox 相关测试 14 条全绿。地址协议 CellAddressCodec 单测 85 条全绿。
