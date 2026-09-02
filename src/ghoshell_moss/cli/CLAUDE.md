<!--
  @provenance
    author:     DeepSeek V4 (via Claude Code)
    date:       2026-05-10
    updated:    2026-07-28 — remove stale module refs, switch to all-commands-first
    to-future:  如果你发现本文档与实际代码不一致，请以代码为准并修改本文档。
-->

# CLI — MOSS Command Line Tools

## 注册方式

所有命令行入口注册在根目录 `pyproject.toml` 的 `[project.scripts]` 下:

```toml
moss       = "ghoshell_moss.cli:main_entry"
moss-shell = 'ghoshell_moss.cli.moss_debug_repl:moss_shell_main'
moss-ghost = 'ghoshell_moss.cli.ghost_run:ghost_run_main'
```

安装后 (`uv sync --active --all-extras`), 这些命令可在 `.venv/bin/` 下找到可执行文件。

## 交互入口

### 1. `moss` — 纯命令行工具

- **入口**: `main.py` → `main_entry()`
- **框架**: Typer, 根 app 定义在 `main.py:app`
- **用途**: 无交互的纯命令行操作。AI worker、脚本、以及人类工程师非交互式使用时的入口
- **关键机制**: 全局 `--ai` flag 通过 callback 注入, 调用 `set_ai_mode(True)` 切换到纯文本输出模式, 剥离所有 rich 视觉排版 (表格转为 markdown, 代码直接输出, rich markup 全部 strip)。这对 AI 消费者节省大量 token
- **子命令组** — 每个是独立的 Typer instance, 通过 `app.add_typer()` 挂载。权威索引始终是 `moss --ai all-commands`，不要手维护此列表。
- **自省命令** (在 `main.py` 中直接定义, 不通过子 app):
  - `help [commands...]`: 批量获取命令帮助。无参数显示根帮助, 带参数按路径解析 (如 `moss --ai help codex get-interface codex concepts`)
  - `all-commands`: 一次性列出所有命令树。`--depth 1/2/3` 控制深度, `--group <name>` 限定子树。设计目标: 将 AI 的 CLI 发现从 40+ 轮压缩到 2 轮

### 2. `moss-shell` — Shell 运行时入口 (四模式: tui / mcp / log / fractalize)

- **入口**: `moss_debug_repl.py` → `moss_shell_main()` (Click group)
- **框架**: Click (简单参数解析) + Textual/prompt_toolkit (tui 模式)
- **模式**:
  - `moss-shell` (无子命令) / `moss-shell tui` — 启动完整 MOSS Host Runtime (不含 Ghost), 进入 TUI 调试终端。人类在给模型 CTML 之前先在这里手动测试。流程: Environment 显式构造 + seal → Host() → MossRuntimeTUI.run()
  - `moss-shell mcp` — 将 MOSS 运行时暴露为 MCP server (原独立 `moss-mcp` 二进制)。需要 `[mcp]` extra, 经 `depend_mcp()` 惰性 gate
  - `moss-shell log` — 无交互 headless 运行, 只输出日志, 供 CI/后台排障
  - `moss-shell fractalize` — 进入 Matrix 网络作为一个 fractal cell, 只暴露本 mode 的 NodeManager (nodes channel) 一条能力。远程 host `mesh:accept` 后可远程治理本 mode 的 nodes (Mode as Cell, workstream: mode-as-cell)
- Ghost 运行前调试 Shell 层的入口: 测 CTML、检 channels/matrix/manifests

### 3. `moss-ghost` — Ghost 交互入口 (group: run / send)

- **入口**: `ghost_run.py` → `ghost_run_main()` (Click group)
- **框架**: Click + GhostTUI (Textual/prompt_toolkit)
- **子命令**:
  - `run <ghost> [--surface tui|output|log]` — 启动一个 Ghost。tui 交互终端 (logos 流式输出、output 结构化消息、SafeMode 审批闸口)；output/log 为 headless 观测面 (tail stdout / 日志)。Ghost 真正的交互界面在 nodes 体系里, TUI 是元控制面
  - `send <text>` — 往同 scope 的 running ghost 注入一条 input signal (非持久化 Matrix 节点, `session.add_input_signal`)。signal 命名空间只含 network_scope, 故不传 --ghost
- **全局 option** (`--mode/--scope/--network`, 放子命令前, 对齐 `moss` 主 CLI 惯例): 无子命令时列出可用 ghost

### 4. `moss-shell mcp` — MOSS Runtime 作为 MCP Server

- **入口**: `moss_debug_repl.py` → `mcp` 子命令 → 惰性 import `moss_as_mcp.py` → `main_entry()`
- **框架**: Click + mcp SDK (MCPServer)
- **用途**: 将 MOSS 运行时暴露为 MCP (Model Context Protocol) 服务, 供 Claude Code 等 AI 工具调用
- **依赖**: `[mcp]` extra (mcp, uvicorn), 经 `depend_mcp()` 惰性 gate — 未安装时 `moss-shell mcp` 报清晰提示
- **核心**:
  - `ServerState`: 持有 `MossHost` 和 `MossRuntime` 引用, server 级 watcher + fire-and-forget task 池
  - `bootstrap()`: 注册 MCP tools (moss_instruction, get_moss_dynamic_info, ctml_append/exec/observe/replan/interrupt)
  - `MCPMessageAdapter`: 将 MOSS Message 转为 MCP ContentBlock
- **传输协议**: 支持 SSE (默认端口 20773), stdio, streamable_http
- `--ai` flag 不适用于此命令, 因为它是 MCP 服务端, 输出遵从 MCP 协议

## 基础设施: `utils.py`

`utils.py` 是整个 CLI 的输出基础设施，所有子命令都 import 它的函数。核心设计:

### `_ConsoleProxy` 代理模式

全局 `console` 是一个代理对象, 根据 `_ai_mode` flag 自动切换输出路径:
- **人类模式**: 委托给 `RichConsole` (颜色/表格/Panel/Syntax 高亮)
- **AI 模式**: 调用 `_ai_print()` 等函数, 用 `click.echo()` 输出纯文本

### 关键输出函数

| 函数 | 人类模式 | AI 模式 |
|---|---|---|
| `print_simple_table(data, headers)` | Rich Table (SIMPLE box) | Markdown table |
| `print_panel(text, title)` | Rich Panel (DOUBLE box) | `## title\ncontent` |
| `print_simple_panel(text, title)` | Rich Panel (SIMPLE box) | `## title\ncontent` |
| `print_code(code)` | 带装饰器的代码块 | 纯代码 |
| `print_success/error/warning/info(msg)` | Rich 彩色输出 | `[OK]/[ERROR]/[WARN]/[INFO]` 前缀 |
| `echo(msg)` | click.echo | click.echo |

### 设计要点

- 所有 rich 对象 (Syntax, Panel, Table) 在 `_ConsoleProxy.print()` 中被拦截, 不会传递给 RichConsole
- `_strip_markup()` 剥离 `[bold cyan]...[/bold cyan]` 等 rich markup 标签
- `console` 是 `_ConsoleProxy` 的单例, 永远不需要替换变量引用 — `from utils import console` 始终生效

## 开发指南

### 框架与风格

1. **子命令用 Typer**, 不是 Click。每个文件是一个独立的 `typer.Typer()` instance, 在 `main.py` 中用 `app.add_typer()` 挂载
2. **入口点用 Click**。`moss-shell`, `moss-ghost` 这两个独立进程不需要 Typer 的 tree handling, 用 Click 参数解析即可
3. **输出统一走 `utils.py`**:
   - 表格用 `print_simple_table()`
   - 面板用 `print_simple_panel()` 或 `print_panel()`
   - 代码用 `print_code()`
   - 不要直接 `console.print()` — 除非内容本身不需要 AI 模式兼容 (如 Syntax 直接在 AI 模式会被 `_ConsoleProxy` 处理, 但最好避免)
4. **所有新命令必须支持 `--ai` flag**。这意味着表格必须用 `print_simple_table`, Syntax 必须由 `console.print()` 输出 (Proxy 会拦截)。如果你需要输出 JSON 给 AI, 加 `--json` option 而不是依赖 `--ai`
5. **`--ai` 是全局 flag**: 在 `main.py` 的 callback 中设置, 所有子命令自动继承

### 全局环境参数

`--mode` / `--ghost` / `--network` / `--scope` / `--workspace` 已在 `main.py` callback 中定义为全局 option。
通过 `_set_global_environment()` 注入到 `Environment` 进程单例，不做验证，谁用谁管。

设计决策：采用 kubectl/docker 标准模式 —— 根级全局 option + 懒解析，而非按 group 重复定义或建三层子 group。

### 添加新子命令组的步骤

1. 新建 `xxx_cli.py`, 定义 `xxx_app = typer.Typer(help="...", no_args_is_help=True)`
2. 在 `xxx_app` 上 `@xxx_app.command()` 装饰函数
3. 在 `main.py` 中 import 并用 `app.add_typer(xxx_app, name="xxx")` 注册
4. 所有输出函数从 `utils.py` import

## Skills 知识库

`skills_cli.py` + `cli/skills/` 目录组成反身性知识库 (取代 howtos, moss-skills):

- **存储**: `cli/skills/<name>/SKILL.md`, 通过 `MarkdownKnowledgeBase` (来自 `ghoshell_moss.resources.markdown_kb`, glob+frontmatter 参数化) 做资源管理
- **结构**: `<name>/SKILL.md` (name + description frontmatter), `cli/skills/README.md` (skill 治理)
- **命令**:
  - `moss skills list [-q keyword] [--json] [--root <path>]`: 发现技能
  - `moss skills recall <question>`: 语义召回 (LLMFuncs 多标签分类, 需 LLM 配置)
- **元规则**: `cli/skills/README.md` 承担——入口判定三问 + 反模式清单 + 写作纪律。写 skill 前必读
- **当前定位**: 复合任务的行动导向技能。历史上 howtos 曾积累 16 篇混合内容, 2026-07-18 doc-governance 治理后收敛为 3 篇, 2026-08 moss-skills 迁移为 skills (见 `.ai_partners/features/workstreams/2026/08/moss-skills/`)

## 架构小贴士

- `Environment.discover()` 在多个命令中独立调用 — 这是设计意图, 因为各命令可能在不同 mode 或 scope 下独立运行
- `moss` CLI 子命令在独立的 Typer 子 app 中实现, 通过 `app.add_typer()` 挂载 — 隔离性好, 各子命令组可独立测试
- `manifests_cli.py` 是最复杂的子命令组, 包含对 providers/topics/configs/channels/primitives/contracts/resources/ctml-versions 的完整自解释体系 — 这是 "code as prompt" 哲学的直接体现
- `moss-shell mcp` 模式依赖 `[mcp]` 可选 extra (mcp, uvicorn), 经 `depend_mcp()` 惰性 gate — 未安装时 mcp 模式报提示, shell 其他模式不受影响
