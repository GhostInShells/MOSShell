---
created: 2026-06-03
depends: []
description: 'moss 自举 channel 重构: bundled moss_cli channel (python -m ghoshell_moss.cli + matrix
  Subprocesses) + 受限 file view 配套 + default mode 降级 wireup。原 typer 反射方案废弃。'
milestone: null
priority: P1
status: in-progress
status_note: '重开 2026-08-06: typer 反射方案废弃(typer_channel.py 删除, 无独立 feature); 新方向 =
  bundled channel moss_cli (python -m + matrix Subprocesses) + 受限 file view 配套 + default
  mode 降级 wireup'
title: Moss Self Channel — 自举 channel: moss_cli (python -m + Subprocesses) + default mode 降级
updated: '2026-08-06'
---

# Moss Self Channel

> 一个 bundled channel `moss_cli`,把 moss 自身 CLI 以"去授权"形态暴露给 ghost:`python -m ghoshell_moss.cli` + matrix Subprocesses。
> 配套受限 file view channel,构成 default mode 降级后的最小开箱面。

## 当前方向 (2026-08-06 重开)

原 typer 反射方案废弃。apps → nodes 是**计划内升级**(`.moss_ws/apps` 0.1 删除,`moss apps` → `moss nodes`),
moss_self 本就在 node-migration 第一梯队迁移清单里,原设计讨论时已考虑 nodes。但本轮重定义选择了**不同形态**:
不做 node,做 bundled channel,服务于 default mode 降级。
驱动因素:typer_channel 泛化方案质量演进未达预期(维护一个未被清晰理解的方案成本高),且本 feature
重新定位为开箱模式的去授权 CLI 暴露——开箱模式不含 bash,以"channel 即授权"暴露 CLI。

**新决策**:

1. **channel 名 `moss_cli`**,单命令 `exec(text__)`,极简提示。
   - **拒绝**"按 import path 反射、把 CLI 子命令逐个注册成 channel command"——反射复杂度回潮,与极简矛盾。
   - **拒绝** node 形态——它是 bundled channel,定义在 `src/ghoshell_moss/channels/`。
2. **执行**:`python -m ghoshell_moss.cli --ai <args>`(`cli/__main__.py` → `main_entry`,无 PATH 依赖),
   经 matrix `Subprocesses`(IoC `CommandUtil.get_contract(Subprocesses)`),**不裸 asyncio subprocess**。
   已确认 Subprocesses 不传 `cell_address`(`_build_env` = `os.environ.copy()` + extra_env)。
3. **cwd**:project 根,经 Project 抽象寻址。不用 `MOSS_WORKSPACE` 环境变量推导(环境变量因果未治理)。
4. **instruction**:嵌入 `src/ghoshell_moss/cli/start.md`——同步安全,无 build-time subprocess。
   原方案同步函数里跑 `subprocess.run` 有阻塞事件循环风险。备选:first-frame 提示 `exec moss start`。
5. **`codex eval` 命令级拒绝**(不注册),CLI 代码保留。
6. **受限 file view 配套 channel**(default mode 用):基于 file_editor 层(file_editor 正在增加
   `glob`/`list_dir`/`grep`),只读 + project-cwd 授权范围,命令级路径拒绝即足够,不做沙箱。
7. **default mode 降级 wireup**:stub 模板 + 本地同步,剥离 Speech/AppStore/MCP/terminal/fractal,
   只剩系统原语 + `moss_cli` + file view。降级后仅需两个 anthropic 配置即可运行。

**访问路径两轨并存**:MCP(`moss-as-mcp` 由外部 agent 触达)是**开发期验证**;bundled channel 挂
default mode 主树是 ghost 的**运行时使用**。二者不冲突,是不同相位。终局:meta-mode 成熟后
ghost 自我运行时开发,外部 agent 路径自然退出,问题溶解。

**待定**:命令名(暂定 `exec`);受限 file view 依赖 file_editor 的 list_dir/glob/grep 落地。

**连带治理**:`app-system-cli` FEATURE 标 dropped(apps → nodes)。

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

> ⚠️ 以下 1-4 是**原始设计决策**(2026-06-03),已被上方"当前方向"推翻,保留作继承记录。
> 核心反转:build-time get_group 反射(→ `--ai all-commands` 权威树)、typer_channel 泛化(→ 删除)、
> 交付形态 App 外置(→ bundled channel)。访问路径不是切换,是两轨并存:MCP 开发期验证 + 进程内运行时使用。

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

## Implementation

App 路径：`.moss_ws/apps/tools/moss_self/`

| 文件 | 职责 |
|------|------|
| `main.py` | Channel 定义 + runtime subprocess 执行 |
| `reflect_cli.py` | Build-time 反射 moss CLI 命令树为 markdown instruction |
| `APP.md` | App 元数据 |

关键实现点：
- 入口适配：直接调用 `moss --ai`（console_script），不走 `python -m typer`
- 反射深度：`reflect_cli.py` 递归遍历 `registered_groups` + `registered_commands`，参数解析到 `--depth 3` 级别
- cwd 绑定：通过 `MOSS_WORKSPACE` 环境变量推导项目根目录，确保子进程在正确目录执行

## 验收标准

> 启动 moss-as-mcp → Claude Code 连接 → 调用 moss codex get-interface ghoshell_moss.channels.typer_channel → 返回 typer_channel 的接口信息。

这个单次调用验证：ghost → MCP → moss App → typer_channel → moss CLI → 反射自身源码 → 返回。
全链路闭环。

**验收结果**：2026-06-11 通过 Claude Code MCP 连接验证，命令 `<apps.tools_moss_self:exec>codex get-interface ghoshell_moss.channels.typer_channel</apps.tools_moss_self:exec>` 正常返回源码与依赖接口。