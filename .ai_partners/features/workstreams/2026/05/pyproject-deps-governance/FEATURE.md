---
created: 2026-05-25
depends: []
description: pyproject.toml 依赖治理：pydantic-ai 从 [host] 拆到 [ghosts]，anthropic 类型懒加载，清理
  markdown_kb recall agent
milestone: null
priority: P1
status: completed
title: Pyproject Deps Governance — 依赖分层治理
updated: '2026-06-01'
---

# Pyproject Deps Governance

## Motivation

`pyproject.toml` 的 `[host]` extras 混合了两类职责：

| 依赖 | 职责 |
|------|------|
| `circus`, `eclipse-zenoh`, `uv`, `uvloop` | 基础设施（进程管理、通讯总线、包管理） |
| `pydantic-ai` | AI 模型调用（Ghost 思维循环） |

运行 Host 基础设施不需要 pydantic-ai。只有 Ghost 开发者才需要它。
当前耦合导致：安装 `[host]` 就必须拉 pydantic-ai 及其传递依赖。

同时，`depends.py` 的 install hint 有维护债：
- `depend_matrix()` / `depend_circus()` 提示 `[matrix]`（实际是 `[host]`）
- `depend_cli()` 提示 `[cli`（括号缺失，且 cli 不是 extras group）
- `depend_pydantic_ai()` 提示 `[host]`（应改为 `[ghosts]`）

另外，`core/resources/markdown_kb/_agent.py` 是测试目的的 pydantic-ai recall agent，
无外部调用者，应删除。

## Key Decisions

### D1: pydantic-ai 从 [host] 拆到 [ghosts]

```
[host]      → circus, eclipse-zenoh, uv, uvloop     (基础设施)
[ghosts]    → pydantic-ai>=1.90.0                     (AI 模型调用)
```

**Why**: 职责分离。Host 运行时不依赖 pydantic-ai。Ghost 开发者装 `[ghosts]` 即可。
不往 `[ghosts]` 里塞其他依赖——保持干净，靠 `depends.py` 给出提示。

**Rejected**: 
- 把 anthropic SDK 也放进 `[ghosts]` — anthropic 在 core deps 中已被 `message/contents/` 和 `tools.py` 使用，不能移出 core。
- 把 `[ghosts]` 做成大而全的 AI 依赖集合 — 太碎体验差，且 litellm/openai 等目前只是可选懒加载。

### D2: anthropic 类型懒加载

`message/contents/images.py` 和 `core/concepts/tools.py` 中的 `from anthropic.types import ...` 改为方法内 lazy import，解除 core 对 anthropic SDK 的模块级硬依赖。

| 文件 | 当前 | 改为 |
|------|------|------|
| `images.py:10` | `from anthropic.types import Base64ImageSourceParam` | lazy，只在 `from_base64()` / `from_binary()` 内 import |
| `tools.py:16` | `from anthropic.types import ToolParam` | lazy，只在 `to_anthropic_tool_param()` 内 import |

**Why**: 这是防御性的——anthropic 已在 core deps 中，但模块级 import 意味着 import `ghoshell_moss.message` 的任何子模块都会触发 anthropic SDK 加载。懒加载让 import 图更干净。

### D3: 删除 markdown_kb/_agent.py

删除文件及所有关联：
- `_agent.py` 本身
- `__init__.py` 中的 `from ._agent import recall, recall_available`
- `_markdown_kb.py:184-189` 中的 `recall()` 方法（回退到 `ResourceStorage.recall()` 默认的 `NotImplementedError`）

`Recollection` 类型保留在 `contracts/resource.py`——它是合法的抽象概念，不因一个实现被删而移除。

**Why**: recall agent 是测试代码，无任何外部调用者（`docs_cli.py` 和 `howto_cli.py` 只 import `MarkdownKnowledgeBase`，不使用 `recall`）。

### D4: 修 depends.py 的 install hint

| 函数 | 旧 hint | 新 hint |
|------|---------|---------|
| `depend_matrix()` | `[matrix]` | `[host]` |
| `depend_circus()` | `[matrix]` | `[host]` |
| `depend_cli()` | `[cli` | 删掉（typer 已在 core deps） |
| `depend_pydantic_ai()` | `[host]` | `[ghosts]` |

### D5: [ghosts] 保持最小

只放 `pydantic-ai`。不做 `[ghosts-all]`、不做 `[ghosts-litellm]` 等碎片化分组。
未来 Ghost 实现者如需其他模型 SDK，在自己的 workspace/app 里声明依赖。
`depends.py` 提供清晰的 install hint 就足够了。

## Blast Radius

| 改动 | 影响 |
|------|------|
| `pyproject.toml`: pydantic-ai 移到 `[ghosts]` | 装 `[host]` 的用户不再拉 pydantic-ai |
| `depends.py`: 修 4 个 hint 字符串 | 错误提示变正确 |
| `images.py`: lazy import | 行为不变，import 图更干净 |
| `tools.py`: lazy import + openai 已是 lazy | 行为不变 |
| `markdown_kb/`: 删 3 处 | 无外部调用者，安全 |
| `ghosts/atom/`: 不动 | 仍用 pydantic-ai，import 时触发 `depend_pydantic_ai()` |

## Implementation Steps

1. `pyproject.toml`: 新增 `[ghosts]` extras，从 `[host]` 移除 `pydantic-ai`
2. `depends.py`: 修 4 个 hint
3. `message/contents/images.py`: `Base64ImageSourceParam` → lazy
4. `core/concepts/tools.py`: `ToolParam` → lazy
5. `markdown_kb/`: 删除 `_agent.py`，清理 `__init__.py` 和 `_markdown_kb.py` 的 recall 相关代码
6. 回归验证：`import ghoshell_moss` 不触发 pydantic_ai 加载；`from ghoshell_moss.ghosts.atom import AtomMeta` 触发 `depend_pydantic_ai()` 并给出正确的 `[ghosts]` hint