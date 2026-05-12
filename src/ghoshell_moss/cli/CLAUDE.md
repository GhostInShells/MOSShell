<!--
  @provenance
    author:     DeepSeek V4 (via Claude Code)
    date:       2026-05-10
    process:    阅读 cli/ 目录下全部源码 (main.py, cli_controller.py, moss_as_mcp.py,
               moss_debug_repl.py, utils.py, codex_cli.py, concepts_cli.py, ctml_cli.py,
               howto_cli.py, manifests_cli.py, modes_cli.py, workspace_cli.py, apps_cli.py)
               以及 pyproject.toml 入口注册、how_tos/ 目录结构后撰写。
    to-future:  如果你发现本文档与实际代码不一致，请以代码为准并修改本文档。
               在 commit message 中标注 "docs: update CLI/CLAUDE.md" 即可。
-->

# CLI — MOSS Command Line Tools

## 注册方式

所有命令行入口注册在根目录 `pyproject.toml` 的 `[project.scripts]` 下:

```toml
moss       = "ghoshell_moss.cli:main_entry"
moss-cli   = "ghoshell_moss.cli.cli_controller:main"
moss-repl  = "ghoshell_moss.cli.moss_debug_repl:moss_debug_repl_main"
moss-as-mcp = "ghoshell_moss.cli.moss_as_mcp:main"
```

安装后 (`uv sync --active --all-extras`), 这些命令可在 `.venv/bin/` 下找到四个可执行文件。

## 四大入口点

### 1. `moss` — 纯命令行工具

- **入口**: `main.py` → `main_entry()`
- **框架**: Typer, 根 app 定义在 `main.py:app`
- **用途**: 无交互的纯命令行操作。AI worker、脚本、以及人类工程师非交互式使用时的入口
- **关键机制**: 全局 `--ai` flag 通过 callback 注入, 调用 `set_ai_mode(True)` 切换到纯文本输出模式, 剥离所有 rich 视觉排版 (表格转为 markdown, 代码直接输出, rich markup 全部 strip)。这对 AI 消费者节省大量 token
- **子命令组** (每个都是独立的 Typer instance, 通过 `app.add_typer()` 挂载):
  - `codex` → `codex_cli.py`: Python 运行时反射 (get-interface, get-source, list, info)
  - `concepts` → `concepts_cli.py`: MOSS 架构概念展示 (core/blueprint/contracts)
  - `ctml` → `ctml_cli.py`: CTML 版本管理 (list, read)
  - `ws` → `workspace_cli.py`: workspace 管理 (where, init, copy-env)
  - `manifests` → `manifests_cli.py`: 环境发现与自解释 (providers, topics, configs, channels, primitives, contracts, ctml-versions, resources)
  - `modes` → `modes_cli.py`: MossMode 管理 (list, show, create)
  - `apps` → `apps_cli.py`: App 管理 (list, show, test)
  - `how-tos` → `howto_cli.py`: 知识库 (list, read, recall)
  - `features` → `features_cli.py`: AI 原生 feature tracking (specification, list, status, create, archive, init)
- **自省命令** (在 `main.py` 中直接定义, 不通过子 app):
  - `help [commands...]`: 批量获取命令帮助。无参数显示根帮助, 带参数按路径解析 (如 `moss --ai help codex get-interface concepts core`)
  - `all-commands`: 一次性列出所有命令树。`--depth 1/2/3` 控制深度, `--group <name>` 限定子树。设计目标: 将 AI 的 CLI 发现从 40+ 轮压缩到 2 轮

### 2. `moss-cli` — 面向人类使用的交互式 Shell

- **入口**: `cli_controller.py` → `main_entry()`
- **框架**: Click (装饰器风格) + prompt_toolkit + Typer
- **用途**: 人类工程师的交互式 REPL, 带 Tab 自动补完
- **核心组件**:
  - `TyperAppCompleter`: 基于 Typer/Click 命令树自动生成补全, 支持 `/` (命令) 和 `?` (帮助) 前缀
  - `TyperAppController`: 异步 REPL 循环, 带 bottom toolbar 显示当前模式
  - `interactive_config()`: 启动时的交互式模式选择和 session scope 配置
- **执行方式**: 每个命令通过 `subprocess.run()` 在新的子进程中执行 (`python -m typer ghoshell_moss.cli.main run ...`), 注入 MOSS 环境变量。这意味着它实际调用了 `moss` 命令
- **补全逻辑**: 递归解析 Typer → Click Group 的命令树, 逐层匹配

### 3. `moss-repl` — MOSS Runtime 人类交互界面

- **入口**: `moss_debug_repl.py` → `moss_debug_repl_main()`
- **框架**: Click (简单参数解析) + Textual/prompt_toolkit
- **用途**: 启动完整 MOSS Host Runtime, 进入 TUI 调试终端
- **流程**: Environment.discover() → 设置 mode/scope → Host() → MossRuntimeTUI.run()
- 这是最完整的运行时入口, mode/scope 均有默认值

### 4. `moss-as-mcp` — MOSS Runtime 作为 MCP Server

- **入口**: `moss_as_mcp.py` → `main()`
- **框架**: Click + FastMCP
- **用途**: 将 MOSS 运行时暴露为 MCP (Model Context Protocol) 服务, 供 Claude Code 等 AI 工具调用
- **核心**:
  - `ServerState`: 持有 `MossHost` 和 `MossRuntime` 引用
  - `bootstrap()`: 注册 MCP tools (moss_instruction, get_moss_dynamic_info, execute_ctml, interrupt_execution)
  - `FastMCPMessageAdapter`: 将 MOSS Message 转为 MCP ContentBlock
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
2. **入口点用 Click**。`moss-cli`, `moss-repl`, `moss-as-mcp` 这三个独立进程不需要 Typer 的 tree handling, 用 Click 参数解析即可
3. **输出统一走 `utils.py`**:
   - 表格用 `print_simple_table()`
   - 面板用 `print_simple_panel()` 或 `print_panel()`
   - 代码用 `print_code()`
   - 不要直接 `console.print()` — 除非内容本身不需要 AI 模式兼容 (如 Syntax 直接在 AI 模式会被 `_ConsoleProxy` 处理, 但最好避免)
4. **所有新命令必须支持 `--ai` flag**。这意味着表格必须用 `print_simple_table`, Syntax 必须由 `console.print()` 输出 (Proxy 会拦截)。如果你需要输出 JSON 给 AI, 加 `--json` option 而不是依赖 `--ai`
5. **`--ai` 是全局 flag**: 在 `main.py` 的 callback 中设置, 所有子命令自动继承

### 全局环境参数

`--mode` / `--session-scope` / `--workspace` 已在 `main.py` callback 中定义为全局 option。
通过 `_set_global_environment()` 注入到 `Environment` 进程单例，不做验证，谁用谁管。

- 无环境需求的命令 (codex, concepts, ctml, how-tos, features, ws) 不受影响，自动忽略
- 有环境需求的命令 (manifests, apps, modes) 通过 `Host()` → `Environment.discover()` 自动获取已设置的值
- **第二步（待做）**: 删除各子命令中冗余的 `--mode` / `--session_scope` 参数，统一走全局

设计决策：采用 kubectl/docker 标准模式 —— 根级全局 option + 懒解析，而非按 group 重复定义或建三层子 group。

### 添加新子命令组的步骤

1. 新建 `xxx_cli.py`, 定义 `xxx_app = typer.Typer(help="...", no_args_is_help=True)`
2. 在 `xxx_app` 上 `@xxx_app.command()` 装饰函数
3. 在 `main.py` 中 import 并用 `app.add_typer(xxx_app, name="xxx")` 注册
4. 所有输出函数从 `utils.py` import

## How-To 知识库

`howto_cli.py` + `how_tos/` 目录组成一个反身性知识库:

- **存储**: `how_tos/` 目录下的 markdown 文件, 通过 `MarkdownKnowledgeBase` (来自 `ghoshell_moss.core.resources.markdown_kb`) 做资源管理
- **结构**:
  - `how_tos/README.md` → howto_app 的 help text
  - `how_tos/*.md` → 一级文档
  - `how_tos/<subdir>/README.md` → 子目录领域概述
  - `how_tos/<subdir>/*.md` → 子目录下的文档, path 为 `<subdir>/<filename>.md`
- **命令**:
  - `moss how-tos list [-q keyword] [--json]`: 列出所有文档
  - `moss how-tos read <path>`: 读取文档 (带 syntax highlighting)
  - `moss how-tos recall <question>`: AI 语义召回 (需 `ANTHROPIC_SMALL_FAST_MODEL` 环境变量)
- **贡献方式**: 见 `how_tos/how-to-make-how-to.md`。必须包含 YAML frontmatter (`title` + `description`), description 是给 AI 召回用的关键字段
- **当前文档**:
  - `for-moss-core-developer/`: 内核开发相关 (IoC 容器, Matrix 能力发现)
  - `for-moss-app-developer/`: 应用开发相关 (ResourceStorage 添加)

## 架构小贴士

- `Environment.discover()` 在多个命令中独立调用 — 这是设计意图, 因为各命令可能在不同 mode 或 scope 下独立运行
- `moss-cli` 通过 subprocess 调用 `moss` 命令, 这种设计保证了隔离性: REPL 进程不加载业务代码, 避免内存泄漏和 import 冲突
- `manifests_cli.py` 是最复杂的子命令组, 包含对 providers/topics/configs/channels/primitives/contracts/resources/ctml-versions 的完整自解释体系 — 这是 "code as prompt" 哲学的直接体现
- `moss-as-mcp` 依赖 `fastmcp` 可选依赖 (`[project.optional-dependencies]` 中的 `mcp`), 未安装时无法运行
