---
created: 2026-06-03
depends: []
description: typer_channel 包装 moss 自身 CLI 为 Channel，注册为 App，通过 MCP 暴露给 Ghost。 验收：Claude
  Code 通过 MCP 连接 moss-as-mcp，调用 moss codex get-interface 成功。
milestone: null
priority: P1
status: draft
status_note: delegating to other collaborators for implementation
title: Moss Self Channel — Typer CLI 反射为 Channel，实现 moss 命令自举
updated: '2026-06-04'
---

# Moss Self Channel

> 用 typer_channel 把 moss 自己包成 channel → 注册为 App → MCP 暴露 → Ghost 能用 moss 命令开发 moss。

## Motivation

moss CLI 已经是一套完整的 typer 命令树（codex, features, manifests, ctml...），
typer_channel 已经有原型能把任意 typer app 反射为 channel。二者拼在一起就是自举：
ghost 通过 MCP 调 moss 命令，用 moss 的工具体系开发 moss 自身。

这不是新能力，是已有零件的一次组装验证。验证通过后，Atom ghost 原型也有了一个
完整的 CLI 工具链作为"身体"。

## Design Index

- `typer_channel.py`：现有原型，`src/ghoshell_moss/channels/typer_channel.py`
- moss CLI 入口：`src/ghoshell_moss/cli/main.py`
- App 系统：`moss codex blueprint app` / `moss apps --help`
- MCP 暴露：`moss-as-mcp`

## Key Decisions

### 1. 两层架构：build-time 反射 + runtime subprocess

**选择**：用一个子进程完成 `get_group()` 反射，将命令树 dump 到文件。
Channel 读取文件做 instruction，执行时走 subprocess。

**拒绝**：进程内 `CliRunner().invoke()`。

**Why**：
- 依赖隔离：channel 进程不 import moss CLI 的完整依赖树
- 崩溃隔离：命令执行 crash 不影响 channel
- 安全天然：subprocess 边界 + typer 自身类型校验 = 双重保险
- `CliRunner` 本身不扔 SystemExit（它内部捕获了），但进程内方案没有上述隔离收益

### 2. text__ 参数接受命令字符串

**选择**：`async def exec(text__: str) -> str`，模型用开放-闭合标签传命令：

```ctml
<moss:exec>codex get-interface ghoshell_moss.channels.typer_channel</moss:exec>
```

**拒绝**：`cmd: str` 作为 XML 属性。

**Why**：text__ 无转义问题，命令含引号、特殊字符时不会被 XML 属性解析破坏。
CTML 的 text__ 就是为这个场景设计的。

### 3. instruction 用 all-commands 输出

**选择**：instruction 内容 = `moss --ai all-commands --depth 3` 的输出（或等价的 get_group 反射结果）。

**Why**：模型不需要多轮探索。instruction 就是完整命令树，一轮看完。
这比让模型自己跑 `--help` 高效几个数量级。

### 4. 安全边界

零审批命令体系的安全分层：

| 层 | 机制 |
|----|------|
| 命令白名单 | typer 只认识注册过的命令，不存在任意命令注入 |
| 参数类型校验 | typer/click 在 dispatch 前校验所有参数类型 |
| 进程隔离 | subprocess 执行，不共享内存 |
| 后续可加 | workspace 路径限制、timeout、危险命令注册表 |

typer 层已经消除了 shell 注入面——没有 `shell=True`，没有字符串解析为命令。

## Implementation Notes

### 入口适配

moss CLI 的 console_script 入口是 `moss`，内部通过 typer app 分发。
typer_channel 当前假设 `python -m typer module_path run`，需要适配为
直接 `moss --ai <command>` 或进程内 typer invoke。

### get_group 反射深度

当前 `get_instruction()` 只遍历一级命令名 + help。对于 moss 的命令树（两级深度），
需要递归 `group.commands` 到子组。`all-commands --depth 3` 的输出格式可以作为基准。

### 与现有 typer_channel 的关系

当前 `typer_channel.py` 是 alpha 原型。本 feature 不需要重构它——先在 moss CLI
这个具体场景上跑通，验证两层架构和 text__ 模式。跑通后再考虑泛化回 typer_channel。

### 验收标准

> 启动 moss-as-mcp → Claude Code 连接 → 调用 moss codex get-interface ghoshell_moss.channels.typer_channel → 返回 typer_channel 的接口信息。

这个单次调用验证：ghost → MCP → moss App → typer_channel → moss CLI → 反射自身源码 → 返回。
全链路闭环。