---
date: 2026-05-19
title: First Runtime Self-Iteration
feature: app-system-cli
author: deepseek-v4
---

# First Runtime Self-Iteration

MOSS 架构下第一个端到端走通的运行时自迭代闭环。

AI 通过 MCP/CTML 完成了 init → 写码 → start → call → 返回值的完整链路，意味着 AI 不再是被写死的工具调用者，而能在运行时扩展自己的能力边界。

## Technical Summary

**根因修复**:
- `AppInfo.make_address` 魔法值 → 统一走 `Cell.make_address("app", fullname)`
- `send_command_task` chan fallback → consumer/provider 路径混淆，改为显式传 `provider_side_chan_path`
- `list_apps` 缺少 refresh → `get_apps_context` 新增 `refresh` 参数透传

**验证 App**: `ai_tools/calc` (add / multiply / div + context_messages)
- `moss apps init` → 脚手架创建
- `<apps:start timeout=3>` → proxy 连通，2s 内 connected
- `<apps.ai_tools_calc:add/multiply/div />` → 返回值正确，异常正确

## Significance

这是项目从 "应用框架" 到 "AI 操作系统内核" 的质变点。

此前 Channel 的注册依赖 workspace 静态声明 + host 启动加载。现在 AI 可以：
1. 在运行时创建新的能力单元 (App)
2. 将其注册到通讯总线 (Zenoh)
3. 通过 CTML 流式调度新能力

这打开了 AI 自生长的能力边界 —— 类比 OS 中 `fork` + `exec` 使得进程可以创造进程。
