# L1. Create a Node — 从零创建 MOSS Node 的完整流程

> Written by deepseek-v4-pro, 2026-07-21

**时间估计**: 15 分钟  |  **学习目标**: 用 CLI 创建 Node、设计目录、配置依赖、安装环境

## 你需要知道什么

- `moss --ai all-commands --group nodes` — Node CLI 完整命令
- `moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest` — NODE.md 的 schema
- `moss codex get-interface ghoshell_moss.core.blueprint.cell:ExecSpec` — exec 段的定义

## 你要做什么

创建一个可被 MOSS Matrix 发现和拉起的 Node cell。Node 是独立进程，自带依赖、膜接口、声明文件。

本 tutorial 以 desktop-gui 为例（一个 Reflex web GUI），整个过程聚焦 Node 骨架和环境安装，不涉及具体实现代码。

---

## 第一步：创建 Node 骨架

```bash
moss nodes create nodes/skins/desktop-gui
```

输出：
```
✓ Node 'desktop-gui' created at .../nodes/skins/desktop-gui

ℹ  Read .../README.md — what to fill in before running or sharing.
ℹ  Edit .../NODE.md — name, exec, instruction body.
ℹ  Read .../INSTALL.md — declares install steps.
ℹ       Delete it if no install is needed (then the node is installed by default).
ℹ       Otherwise run the steps, then: moss nodes install nodes/skins/desktop-gui
ℹ  Run: moss nodes run nodes/skins/desktop-gui
```

生成的文件：

```
nodes/skins/desktop-gui/
├── NODE.md           # manifest: name, singleton, exec, body
├── INSTALL.md        # 安装步骤（可选，不需要则删）
├── README.md         # 人类文档（可选）
├── main.py           # entry point
├── .gitignore
└── runtime/          # 运行时文件（gitignored）
```

## 第二步：设计目录结构

Node 支持两种目录风格：

| 风格 | 适用场景 | Python 包位置 |
|------|---------|-------------|
| Flat | 单文件工具节点，不跨项目复用 | `main.py` 同级 |
| src layout | 独立发版、跨项目复用 | `src/<package>/` |

**desktop-gui 选择 src layout**——它有自己的 pyproject.toml、独立依赖、未来可能跨项目复用。

完整目录：

```
nodes/skins/desktop-gui/
├── NODE.md                    # exec: python main.py
├── INSTALL.md                 # uv sync + nodes install
├── pyproject.toml             # 独立依赖声明
├── rxconfig.py                # Reflex 配置（放根目录，Reflex 从 CWD 找）
├── main.py                    # 薄入口，import 并 run package
├── src/
│   └── ghoshell_desktop_gui/  # Python 包（独立命名空间）
│       ├── __init__.py
│       ├── app.py
│       ├── state.py
│       ├── pages/
│       └── components/
└── runtime/                   # 运行时文件（gitignored）
```

要点：
- `rxconfig.py` 放根目录——Reflex 从 CWD 发现配置
- `main.py` 只有两行：`from pkg.app import main; main()`
- 包名用独立命名空间（`ghoshell_desktop_gui`），不和 MOSS 核心冲突

## 第三步：配置 NODE.md

```markdown
---
name: 'desktop-gui'
description: 'Desktop GUI — human observation and approval interface for the desktop channel'
singleton: true
exec:
  command: python
  args: main.py
---

Body describes what this node does — the model reads this when the channel
is accepted. Include capability summary + CTML invocation examples.
```

关键字段：
- `singleton: true` — 同一网络内只允许一个实例，重复拉起抛 `DuplicatedError`
- `exec.command: python` — 用什么解释器
- `exec.args: main.py` — 传给解释器的参数
- Body 是给模型读的——描述能力、给出 CTML 示例

## 第四步：配置 pyproject.toml

```toml
[project]
name = "ghoshell-desktop-gui"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "ghoshell-moss[matrix]",
    "reflex>=0.9.0",
]

[tool.uv.sources]
ghoshell-moss = { path = "../../..", editable = true }

[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"
```

要点：
- `ghoshell-moss[matrix]` — matrix extra 提供 zenoh 通讯依赖
- `path = "../../.."` — 相对路径指向 MOSS 项目根。计算方式：node 目录深度 = 从项目根向下的层数。`nodes/skins/desktop-gui` = 3 层，所以 `../` × 3
- 用 `editable = true` 开发期实时同步 MOSS 变更

## 第五步：写 entry point

`main.py`（根目录，薄入口）：

```python
"""Desktop GUI node entry point.

Start:  moss nodes run nodes/skins/desktop-gui
Debug:  python main.py
"""

from ghoshell_desktop_gui.app import main

if __name__ == "__main__":
    main()
```

`src/ghoshell_desktop_gui/app.py`（真正的启动逻辑）：

```python
import reflex as rx
from ghoshell_desktop_gui.pages.index import index

def main():
    app = rx.App()
    app.add_page(index, route="/")
    app.run()
```

保持 `main.py` 极简——一行 import 一行 run。所有逻辑在 package 里。

## 第六步：安装环境

```bash
cd nodes/skins/desktop-gui

# 重要：先 unset VIRTUAL_ENV，避免 uv sync 误用根项目的 venv
unset VIRTUAL_ENV
uv sync

# 安装完成后标记
moss nodes install nodes/skins/desktop-gui
```

`uv sync` 在 node 目录下创建独立的 `.venv`，根项目环境不受影响。
`moss nodes install` 建 `.installed` 标记文件——无标记的 node 不能被 `moss nodes run` 拉起。

## 第七步：验证

```bash
moss nodes list    # 应看到 desktop-gui
moss nodes show nodes/skins/desktop-gui   # 查看 NODE.md 原文 + 目录内容
```

`moss nodes show` 输出包括 frontmatter 和 body，以及目录下的文件列表。确认 `pyproject.toml`、`main.py`、`src/` 都在。

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `uv sync` 失败 "does not appear to be a Python project" | `path` 层数不对 | 从 node 目录逐层向上数到项目根，确认 `../` 数量 |
| `uv sync` 装到了根 .venv | `VIRTUAL_ENV` 未 unset | 先 `unset VIRTUAL_ENV` 再 sync |
| `moss nodes run` 提示未安装 | 缺少 `.installed` | 跑 `moss nodes install` |
| Node 不在 list 里 | 发现路径不覆盖 | 确认 node 目录在 project 的 nodes 发现路径下 |

## 相关文档

```bash
moss --ai all-commands --group nodes                          # Node CLI 完整命令
moss codex get-interface ghoshell_moss.core.blueprint.cell     # Cell/Node 核心定义
moss codex blueprint matrix                                    # Matrix 入网机制
moss codex blueprint channel_builder                           # 下一步：写 Channel
```

| 时间 | 模型 | 备注 |
|------|------|------|
| 2026-07-21 | deepseek-v4-pro | 基于 desktop-gui node 创建流程撰写，node 创建 + src layout + 环境安装全链路 |
