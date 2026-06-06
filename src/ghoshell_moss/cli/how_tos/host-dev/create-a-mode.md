---
title: Create a Mode
description: 如何创建一个 MOSS mode 来隔离 app 权限、定制能力视图。覆盖创建、配置、添加 manifest、验证的完整流程。面向 app 开发者。
---

# How to Create a Mode

## 背景

Mode 通过显式继承定义自己的能力视图。它解决两个问题：

1. **权限隔离** — 通过 apps 白名单控制该 mode 下可访问哪些 app
2. **能力定制** — mode 可以构建独立的 main channel 或继承全局能力后扩展

什么时候需要创建 mode：
- 为一个特定任务或场景限定可用 app 范围
- 为实验性功能提供隔离的运行环境，不影响 default mode
- 为不同硬件配置（不同机器人、不同外设）提供各自的能力组合

什么时候**不需要**创建 mode：
- 只是在 default mode 下开发新 app — 直接加，无需新 mode
- 只是临时调试 — 用 `--mode system_test`（如果可用）

### 深入理解 Mode 模型

```bash
moss codex get-interface ghoshell_moss.core.blueprint.matrix:Mode
moss docs read workspace-and-mode.md
```

## 步骤

### 1. 创建 mode

```bash
moss modes create <name> -d "one-line description" -a "group/*" -u "group/app"
```

参数：
- `-d` — 一行描述
- `-a` — 允许的 app 白名单，可重复。默认 `*`（全部公开 app）
- `-u` — 启动时自动 bringup 的 app，可重复。默认空

创建后 CLI 会列出生成的文件清单。

### 2. 编辑 MODE.md 配置权限边界

```bash
# 找到 mode 目录
moss modes show <name>   # 看 File Path 那行
```

编辑 `MODE.md` 的 YAML frontmatter：

- `apps` — app 白名单。`*/*` 全部公开，`group/*` 某 group 下全部，`_` 前缀禁止
- `bringup_apps` — 启动时自动拉起哪些 app。通常留空，按需添加
- 正文区写 mode 的使用说明，会显示在 `moss modes show` 中

### 3. 按需添加 manifest 文件

Mode 目录下可选的 manifest 文件：

| 文件 | 何时添加 |
|------|---------|
| `channels.py` | 需要定制 main channel（继承全局或从零构建） |
| `providers.py` | 默认继承全局（`from MOSS.manifests.providers import *`），追加 mode 专属的 |
| `configs.py` | 默认继承全局，追加 mode 专属配置 |
| `topics.py` | 默认继承全局，追加 mode 专属事件协议 |
| `resources.py` | 默认继承全局，追加 mode 专属资源存储 |
| `nuclei.py` | 默认继承全局，追加 mode 专属感知核 |
| `contracts.py` | 需要声明 mode 专属的 contract 绑定 |

大多数场景只改 `channels.py` 就够了。模板文件里已有注释说明怎么改写。

manifest 文件的显式继承模式：
```python
from MOSS.manifests.providers import *
# mode 专属追加
my_provider = MyProvider()
```

channels.py 的两种构建模式：

```python
# 独立 mode（推荐）：从零构建标准 shell main channel
from ghoshell_moss import new_default_shell_main_channel
main = new_default_shell_main_channel(description="...")
# 在 main 上追加自己的 command 或 compose sub-channel...

# 复用模式：继承全局 main 后增量改造
from MOSS.manifests.channels import main
# 在 main 上追加改造...
```

查看 `new_default_shell_main_channel` 源码来理解它默认组装了什么：

```bash
moss codex get-source ghoshell_moss.core.blueprint.states_channel:new_default_shell_main_channel
```

### 4. 验证 mode

```bash
# 查看 mode 详情和 manifest 文件清单
moss modes show <name>

# 查看显式继承后的能力视图
moss --mode <name> manifests explain

# 查看 main channel 的命令树
moss --mode <name> manifests channels

# 查看可用 provider 和 contract
moss --mode <name> manifests providers
moss --mode <name> manifests contracts
```

`moss manifests explain` 是验证 mode 能力的最快方式 — 它会列出 mode manifest 文件的内容。对比 default mode 和自己的 mode 的 explain 输出，差异来自 mode 目录下的 manifest 文件 + MODE.md 的 apps 白名单。

### 5. 使用 mode

大部分 moss 命令支持 `--mode` flag：

```bash
moss --mode <name> manifests explain
moss --mode <name> apps list
```

moss-repl 启动时选择 mode：

```bash
moss-repl  # 交互式选择 mode 和 session scope
```

## 示例

### 参考已有 mode

```bash
# workspace stub 中的 default mode（最小示例）
ls $(python -c "import ghoshell_moss.host.stubs.workspace; print(ghoshell_moss.host.stubs.workspace.__file__.rsplit('/', 1)[0])")/src/MOSS/modes/default/

# system_test mode（带 channels.py 定制）
moss modes show system_test
```

### 查看 Mode 能力视图

```bash
# 对比 default mode 和自定义 mode 的能力差异
moss --mode default manifests explain
moss --mode <name> manifests explain
```

差异来自 mode 目录下的 manifest 文件 + MODE.md 的 apps 白名单。

## 常见问题

### 问题：创建 mode 后 `moss modes list` 看不到

确认 mode 目录下有 `__init__.py`（`moss modes create` 会自动创建）。如果手动创建目录，需要自己加。

### 问题：mode 的 manifest import 报错

检查 import 路径。在 mode 的 Python package 上下文中：
- `from ghoshell_moss import ...` — 正确，从已安装的包导入
- `from MOSS.manifests.channels import main` — 正确，从 workspace 的全局 manifests 导入
- `from .channels import ...` — 相对导入在 mode 包内也可能工作，但推荐用绝对路径

### 问题：`moss --mode <name> manifests explain` 没有期望的效果

确认 mode 目录下的 manifest 文件有正确的 `from MOSS.manifests.xxx import *`。用 `moss modes show <name>` 看文件是否被识别为 `present`。

### 问题：apps 白名单不生效

检查 MODE.md frontmatter 的 `apps` 字段格式。每个 pattern 一行，用 `- 'pattern'` 语法。注意 YAML 缩进必须用空格。

## 文档目标

读者按照本文档操作，应该能够：
1. 用 `moss modes create` 创建一个新 mode
2. 编辑 MODE.md 配置 apps 白名单和 bringup 策略
3. 在 mode 目录下按需添加 manifest 文件（至少 channels.py）
4. 用 `moss modes show` 和 `moss --mode <name> manifests explain` 验证 mode 显式继承效果
5. 用 `moss --mode <name>` 在指定 mode 下运行命令
