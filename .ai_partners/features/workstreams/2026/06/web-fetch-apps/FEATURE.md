---
title: Web Fetch Apps — 隔离式 Web 内容抓取工具集
status: in-progress
priority: P2
created: 2026-06-04
updated: 2026-06-04
depends: []
milestone:
description: >-
  将 trafilatura、firecrawl、jina-reader 等 web 抓取工具做成隔离的 MOSS app，
  放在 apps/web/ 下，让 Ghost 通过 CTML channel 原生获取 web 内容。
---

# Web Fetch Apps

## Motivation

当前 AI 获取 web 内容的唯一路径是 Claude Code 的 WebFetch 工具。MOSS 自身没有 web 抓取能力——Ghost 在运行时无法自主从 web 获取信息。这在两条路径上都造成断裂：

- **MCP 路径**：AI 编码工具（Claude Code 等）通过 `moss-as-mcp` 连接 MOSS 时，如果模型需要查 web 文档、抓取页面内容，只能依赖编码工具自身的 fetch 能力。MOSS 的工具链在这里是空的——模型在 MOSS 上下文中无法发起 web 请求
- **Ghost 路径**：Ghost 运行时无法自主从 web 获取信息，信息获取半径被限制在代码仓库和信号输入

目标：将主流 web 抓取工具做成隔离的 MOSS app，每个工具独立进程、独立依赖、独立生命周期。Ghost 通过 CTML channel 调用它们，就像调用任何其他 channel 一样。

选择 app 模式（而非在 core 里写一个通用 web fetch 模块）的理由：
- **依赖隔离**：trafilatura 需要 `trafilatura` + `lxml`，firecrawl 需要 `firecrawl-py`，Jina 只需要 `httpx`。放在一个包里会污染依赖树
- **独立生命周期**：按需启动/停止。不需要 web 能力的 Mode 不加载它们
- **AI 可热装**：AI 可以在会话中 `moss apps create web/xxx` 然后 `start`，即刻获得新能力
- **符合 app 哲学**：web fetch 是 Ghost 可插拔的"器官"，不是内核功能

## Design

### 目录拓扑

```
apps/web/                       # 新 group: web
├── README.md                   # group 级说明（可选）
├── trafilatura/                # app: web/trafilatura
│   ├── APP.md
│   ├── CLAUDE.md
│   ├── pyproject.toml          # 依赖: trafilatura
│   ├── main.py
│   └── runtime/
├── firecrawl/                  # app: web/firecrawl
│   ├── APP.md
│   ├── CLAUDE.md
│   ├── pyproject.toml          # 依赖: firecrawl-py
│   ├── main.py
│   └── runtime/
└── jina-reader/                # app: web/jina-reader
    ├── APP.md
    ├── CLAUDE.md
    ├── pyproject.toml          # 依赖: httpx
    ├── main.py
    └── runtime/
```

### 每个 App 的能力边界

| App | 核心能力 | 外部依赖 | API Key | 输出格式 |
|-----|---------|---------|---------|---------|
| `web/trafilatura` | HTML→Markdown 正文提取 | 无 | 不需要 | Markdown |
| `web/firecrawl` | 全量爬取 + LLM 处理 | Firecrawl 服务 | 需要 | Markdown / structured |
| `web/jina-reader` | URL→LLM-ready Markdown | Jina AI 服务 | 免费额度 | Markdown |

### Channel 契约

每个 app 都注册一个以 `web_<name>` 命名的 channel，CTML 调用路径为 `apps.web_<name>:<command>`。

**核心约束**：所有抓取命令必须使用 `@channel.build.command(always_observe=True)`。fetch 结果是观测物——模型需要"看到"它，而不仅仅是拿到一个返回值。`always_observe=True` 确保抓取内容进入 observe 流，在 MCP 场景下模型通过 `moss_observe` 或 CTML 执行结果直接拿到内容。

**trafilatura** — 本地抓取，零外部依赖：

```python
@channel.build.command(always_observe=True)
async def extract(url: str, output_format: str = "markdown") -> str

@channel.build.command(always_observe=True)
async def extract_batch(urls: list[str], output_format: str = "markdown") -> dict[str, str]

@channel.build.command(always_observe=True)
async def extract_selector(url: str, selector: str) -> str
```

**firecrawl** — 全量爬取 + AI 处理：

```python
@channel.build.command(always_observe=True)
async def scrape(url: str, formats: list[str] = ["markdown"]) -> str

@channel.build.command(always_observe=True)
async def crawl(url: str, max_pages: int = 10) -> str

@channel.build.command(always_observe=True)
async def search(query: str, max_results: int = 5) -> str
```

**jina-reader** — 轻量 URL→Markdown：

```python
@channel.build.command(always_observe=True)
async def read(url: str) -> str

@channel.build.command(always_observe=True)
async def read_with_query(url: str, query: str) -> str
```

### trafilatura 作为参考实现

`web/trafilatura` 第一个实现，因为它：
- 纯 Python，无外部 API 依赖，无认证复杂度
- 安装最简单：`uv pip install trafilatura` 即可
- 本地运行，无网络调用限制，无费用
- 代码量最小，适合作为后续实现的模板

后续 `web/firecrawl` 和 `web/jina-reader` 参考其结构，增加 API key 配置和环境变量管理。

## Key Decisions

### KD1: 每个工具独立 app，不共享代码

**决策**：trafilatura、firecrawl、jina-reader 各是独立 app，不抽取公共基类或共享包。

**理由**：
- 它们做的事情完全不同——本地提取 vs 服务端爬取 vs API 代理
- 公共代码量极小（只有 `Matrix.discover().run(main)` 那个入口），不值得抽象
- 独立 app 让 AI 可以只启动需要的那个——省资源、省依赖、省配置复杂度
- 未来如果真出现跨 app 共享逻辑，再抽一个 `web/_common` 不迟

**反模式警告**：不要为了"一致性"而统一三者的 channel 命令签名。trafilatura 有 `extract_selector`，firecrawl 有 `search`，Jina 有 `read_with_query`——这是它们的真实能力差异，强行对齐反而丢失功能。

### KD2: 所有抓取命令标记 `always_observe=True`

**决策**：每个 web fetch 命令都用 `@channel.build.command(always_observe=True)` 装饰，fetch 结果自动进入 observe 流。

**理由**：
- web 抓取的本质是"感知外部世界"，不是"执行一个函数"。模型需要看到结果来形成下一步决策
- `always_observe=True` 让 fetch 内容出现在 Ghost/MCP 的 observe 上下文中，模型不需要额外步骤去取返回值
- 这是 web fetch 和普通工具调用的本质差别——获取信息 vs 执行操作
- MCP 场景下，`moss-as-mcp` 暴露的 CTML 执行工具返回 observe 内容给模型，`always_observe` 确保 fetch 结果在这个通路里

### KD3: trafilatura 优先，作为参考模板

**决策**：先实现 `web/trafilatura`，验证 app 结构、channel 设计、CTML 调用路径全部跑通后，再复刻到 firecrawl 和 jina-reader。

**理由**：
- trafilatura 无外部依赖，最快跑通闭环
- 第一个 app 暴露出的设计问题在后续两个里修正，成本最低
- 后续 app 的 CLAUDE.md 可以直接引用 trafilatura 的实现作为参考

### KD4: API key 通过 .env 文件管理，不硬编码在 APP.md 的 arguments 里

**决策**：firecrawl 和 jina-reader 需要的 API key 通过 app 目录内的 `.env` 文件加载（`python-dotenv`），不通过 APP.md 的 `arguments` 传递。

**理由**：
- APP.md 的 arguments 字段可能在日志/CLI 输出中暴露
- `.env` 是 Python 生态的标准实践
- app 目录下的 `.env` 天然隔离，不同 app 的 key 不会混淆
- 参考 `sensors/voice` app 的模式：根目录 `.env.example` + `load_dotenv()`

### KD5: Channel 输出统一为 Markdown 字符串，不做结构化

**决策**：所有 channel 命令返回 `str`（Markdown 格式），不返回 JSON/dict 等结构化数据。

**理由**：
- CTML 的消费方是 LLM，Markdown 是 LLM 最自然的输入格式
- 结构化输出增加解析负担，且不同工具的结构不同，模型需要适应的 schema 数量膨胀
- 保留扩展性：未来如果需要结构化输出，可以通过 `output_format` 参数选择

### KD6: 不需要 web 全局索引或注册中心

**决策**：apps 系统自己的发现机制（`moss apps list`）已经足够，不在 apps 层之上再做一层 web 工具集的注册。

**理由**：
- `moss manifests channels` 已经能看到所有已注册的 channel
- Group 名 `web` 本身就是索引——想知道有哪些 web 工具，list 一下 group 即可
- 过度抽象违反 features specification 的 "efficiency over format" 原则

## Implementation Plan

### Phase 1: trafilatura（参考实现）

1. 创建 `apps/web/trafilatura/` 目录结构
2. 写 `pyproject.toml`：依赖 `trafilatura` + `ghoshell-moss[host]`
3. 实现 `main.py`：注册 `web_trafilatura` channel，暴露 `extract`、`extract_batch`、`extract_selector`
4. 写 `APP.md` frontmatter
5. 写 `CLAUDE.md`：app 专属开发上下文 + 后续 app 的参考说明
6. 测试：`moss apps test web/trafilatura` → CTML 调用 → 验证返回 Markdown

### Phase 2: firecrawl

1. 参考 `web/trafilatura/` 结构创建 `apps/web/firecrawl/`
2. 增加 `.env.example` 管理 `FIRECRAWL_API_KEY`
3. 实现 `scrape`、`crawl`、`search` 三个命令
4. 写 `CLAUDE.md`

### Phase 3: jina-reader

1. 参考 `web/trafilatura/` 结构创建 `apps/web/jina-reader/`
2. 实现 `read`、`read_with_query`
3. 最简实现——本质上就是 `httpx.get(f"https://r.jina.ai/{url}")`
4. 写 `CLAUDE.md`

### Phase 4: 文档与集成

1. 在 `apps/web/README.md` 中写 group 级说明：各 app 的用途、选择指南
2. 更新相关 Mode 配置（如果需要默认启用某些 web app）
3. 考虑添加一个 `web/search` app，整合多个搜索引擎（可选，后续 feature）

## 验收标准

**核心验收（MCP 闭环）**：

> **启动 `moss-as-mcp`，AI 编码工具通过 MCP 工具执行 CTML，调用 `web/trafilatura:extract` 抓取真实 URL，模型在 observe 返回中看到网页正文 Markdown。这是整个 feature 的价值证明——web fetch 作为 MOSS 原生能力，通过 MCP 路径被模型自主调用。**

**分步验收**：

1. **app 发现**：`moss apps list` 能看到 `web/trafilatura`、`web/firecrawl`、`web/jina-reader`；`moss manifests channels` 能看到对应 channel
2. **前台调试**：`moss apps test web/trafilatura` 前台运行正常，Ctrl+C 停止
3. **CTML 直接调用**：在 moss-repl 或 CLI 中执行 `<apps.web_trafilatura:extract url="https://example.com" />`，返回页面正文 Markdown
4. **always_observe 生效**：抓取结果出现在 observe 上下文中，不只是函数返回值
5. **MCP 闭环（关键）**：
   - 启动 `moss-as-mcp`（对应 Mode 包含 web app 的 bringup）
   - AI 编码工具连接 MCP server
   - 模型通过 MCP 的 CTML 执行工具调用 `<apps.web_trafilatura:extract url="https://github.com/adbar/trafilatura" />`
   - 模型在工具返回中读到 trafilatura README 的 Markdown 内容
   - 模型能基于抓取内容做出后续决策（验证信息闭环）
6. **firecrawl MCP 路径**：同上，在有 `FIRECRAWL_API_KEY` 的环境下通过 MCP 调用 `web/firecrawl:scrape`
7. **jina-reader MCP 路径**：同上，通过 MCP 调用 `web/jina-reader:read`
8. **依赖隔离**：每个 app 的 `pyproject.toml` 独立，trafilatura 的依赖不包含 firecrawl 的，反之亦然
9. **CLAUDE.md 完成**：每个 app 的 `CLAUDE.md` 包含足够的上下文让下个 AI 实例理解和修改

## 开发过程摩擦点

> 这些摩擦点暴露了 app 体系和主项目之间的认知间隙。将在后续 commit 中入体系：app 模板、docs、start.md。

### F1: `moss apps create` 模板过于 minimal

生成的 `main.py` 是 helloworld stub。模板不提示 AI 按"独立 project"思路治理——没有 pyproject.toml 示例、没有测试目录引导、没有 CLAUDE.md 模板。AI 实例需要额外的上下文才能理解 app 的正确开发模式。

**待修**: `stubs/app/` 下增强模板，加 CLAUDE.md / README 提示 project 思维。

### F2: `uv sync --active` 在 app 目录污染主 venv

在 app 目录执行 `uv sync --active` 会用 app 的依赖替换主项目 venv 的依赖（因为 editable path 共享 workspace）。主项目的 venv 被清掉 100 个包，IDE 卡死。

**正确做法**: app 有自己的 `.venv`（由 `uv run` 自动创建），依赖通过 `uv run` 或 `moss apps test` 管理，不要手动 `uv sync --active`。

**待修**: `start.md` 增加 "常见误用命令" 区块，明确警告。

### F3: App 测试应该放在 app 自身目录

app 是独立项目，未来可能从 hub 云端下载。测试放在 `tests/ghoshell_moss/apps/` 下违反独立性原则。本次将测试放在了 `apps/web/trafilatura/tests/`。

**待修**: app 模板提示 test 目录约定。

### F4: 缺少内建 channel 测试套件

当前测试 channel 需要了解 `ctml_shell_test()` 的 API 和 import 路径。channel_builder 没有提供简单的一行测试工具。每个 app 开发者都要重学一遍。

**待修**: `channel_builder` 增加 `test_channel()` 辅助函数，howtos 增加 app 测试专题。

### F5: `core/speech/__init__.py` eager import 链污染

`__init__.py` 无条件导入 `BaseAudioStreamPlayer` 和 `VirtualStreamPlayer`，导致 `scipy`（120MB）成为所有 import 路径的硬依赖。即使只需要 `NullSpeech`，也必须安装 scipy。

**已修**: 移除 `__init__.py` 中的 eager export，`BaseAudioStreamPlayer` 和 `VirtualStreamPlayer` 通过直接路径导入。

### F6: `base_player.py` scipy 顶层 import

`from scipy import signal` 在模块顶层，`resample()` 是静态方法但 scipy 在 import 时就加载。

**已修（人类工程师）**: scipy import 移入 `resample()` 函数内部，仅在真正需要重采样时才加载。



---

*设计: DeepSeek V4 Pro 与人类工程师, 2026-06-04*
