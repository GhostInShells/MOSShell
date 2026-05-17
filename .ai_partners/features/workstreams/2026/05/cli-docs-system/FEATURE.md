---
created: 2026-05-17
depends: []
description: '建立 moss CLI 的 docs 命令组，唯一命令为 list。 扫描约定路径下的 .md 文件，尊重 .gitignore，提取第一个
  # 标题，按 git mtime 排序输出树形索引。 同时建立项目正式文档结构 docs/（en/zh/ai 三路），docs/ 随 CLI 打包发布。'
priority: P2
status: completed
status_note: docs command + directory skeleton delivered; content production is ongoing,
  not part of this feature
title: CLI Docs System
updated: '2026-05-18'
---

# CLI Docs System

## Motivation

当前项目有五种沉淀物，各自有不同的真实性和打包策略：

| 沉淀物 | 性质 | 打包 |
|--------|------|------|
| `features/` | 开发轨迹，git commit 反向索引 | 不进包 |
| `.design/` | 现场设计，高度关联但非 truth | 不进包（pyproject 已忽略） |
| `.discuss/` | 对话锚点，服务于意识体轨迹还原 | 不进包 |
| `.memory/` | 人格化第一人称记忆 | 不进包 |
| `docs/` | **正式文档，面向人类和 AI 的 truth** | **随 CLI 打包** |

没有一个命令能回答"项目有哪些设计文档"。`docs list` 填补这个空白——只做一件事：扫描 + 树形索引。

## What This Is

`docs list` — 扫描约定路径下的 `.md` 文件，尊重 `.gitignore`，提取第一个 `#` 标题，按 git mtime 排序输出树形索引。

### What It Is NOT

- 不提供 `read` 命令（工具替代即可）
- 不提供 `search` 命令（grep 替代即可）
- 不建立元数据层（不要求 frontmatter）
- 不收纳 howto（how-tos 是另一个体系）

## Design Index

- CLI 命令树设计参考: `src/ghoshell_moss/cli/.design/2026-05-17-cli_command_tree_redesign.md`
- 参考实现: GhostOS 2024 的 Directory ABC（`/Users/BrightRed/Develop/github.com/ghost-in-moss/GhostOS/libs/ghostos/ghostos/libraries/project/`）

## Key Decisions

### 1. docs 是唯一打包的正式文档（2026-05-17）

**决策**: 五种项目沉淀物中，只有 `docs/` 随 CLI 打包发布。其余四种（features/.design/.discuss/.memory）是过程性资产，不打包。

**理由**:
- docs 是对外承诺，人读了能理解系统，AI 读了能进入工作状态
- 其余四种的真实性是轨迹性的，不是 formal truth
- pyproject 已将 `.design` 等忽略

### 2. docs 随 CLI 发布，不随项目（2026-05-17）

**决策**: docs 放在 `src/ghoshell_moss/cli/docs/`，随框架安装，属于 AI startup context 的一部分。

**理由**:
- MOSS 类似 ROS2，是环境建立框架
- AI 进入新 workspace 时，需要框架级的知识体系而非散落的项目笔记
- 类比 `concepts` / `how-tos` ——都是随框架走

**`--path` 参数**: 预留 `--path` 指向项目本地 docs，覆盖默认路径。

### 3. 三路结构：en / zh / ai（2026-05-17）

**决策**: docs 分为三个平行路径：

```
docs/
  README.md             # en 入口，GitHub 直显
  en/                   # 面向人类（英文），渐进式叙事
    getting-started.md
    core-concepts.md
    architecture.md
    cli-reference.md
    development.md
    faq.md
  zh/                   # 面向人类（中文），按 en 结构翻译
    README.md
    ...
  ai/                   # 面向 AI 模型，结构独立
    README.md
    ...
```

**理由**:
- `en/zh` 是语言轴，`ai/` 是受众轴
- `ai/` 不承担人类可读性——追求上下文压缩密度，code as prompt + CLI flow
- `en/` 追求渐进叙事，有例子有故事
- 分开写避免两者互相拖累

### 4. `docs list` 只做一个命令（2026-05-17）

**决策**: `docs` 命令组只有一个 `list` 命令，不设 `read` / `search`。

**理由**:
- `read` 是多余的——任何工具都能读（AI 用 Read tool，人用编辑器）
- `search` 是 grep 能做的事，除非理解 frontmatter 和文档结构（那需要另一个工程）
- `docs list` 给地图，不替人走路

### 5. 零元数据负担（2026-05-17）

**决策**: 不强制 frontmatter。`docs list` 的输出就是文件名 + 第一个 `#` 标题 + git mtime。

**理由**:
- 文件中第一个 `#` 标题足够
- 文件名按项目约定已是自解释的
- 不给现有 .design/ 和 .discuss/ 增加维护负担

### 6. 尊重 .gitignore（2026-05-17）

**决策**: `docs list` 扫描时读 `.gitignore` 并过滤。

**理由**: 参考 GhostOS 2024 的 DirectoryImpl 实现——读 `.gitignore` 合并进 ignores list。所有其他文件系统工具都遵循此约定。

### 7. 按 git mtime 排序（2026-05-17）

**决策**: 列表按最近修改时间排序，活跃文档浮到上面。

**理由**: 和 `features list` 一致的行为。`git log -1 --format=%ci` 拿到时间戳，零元数据。

### 8. docs 与 how-tos 的区别（2026-05-17）

| | how-tos | docs |
|---|---|---|
| 受众 | 需要完成任务的开发者 | 需要理解设计的协作者 |
| 结构 | 按领域分目录，有 frontmatter | 树形发现，无元数据要求 |
| 内容 | 操作步骤、代码示例 | 设计理由、架构决策 |
| 维护 | 需要有人写，有模板约束 | 随代码 commit，零额外维护 |
| AI 用 | "我要做 X，查一下怎么弄" | "这个系统有哪些设计资产" |

## Implementation Notes

- 实现位置: `src/ghoshell_moss/cli/docs_cli.py` (单函数 `docs_cmd`)
- 注册为 root command (`app.command(name="docs")`)，非 Typer group（docs 永远不需要子命令）
- `--path` 参数覆盖所有默认行为，直接扫指定路径
- AI 模式 (`--ai`): 默认扫 `docs/ai/`
- 人类模式 + `--lang zh`: 默认扫 `docs/zh/`
- 人类模式默认 (en): 显示 root README + 可用 doc set 提示，不展开树
- 空白子目录不显示（`_build_tree` 过滤空目录）
- `.gitignore` 读取 + `fnmatch` 匹配
- git mtime 通过 `git log -1 --format=%ct` 获取，fallback 到文件系统 mtime
- 树形输出使用 Unicode box-drawing 字符（`├──` / `└──` / `│`）

## Scope

### 本轮

- [x] 创建 `docs/` 目录结构（en/zh/ai 子目录 + 顶层 README）
- [x] 实现 `moss docs` root command（默认 README + hints / --ai → ai/ / --lang → zh/ / --path → custom）
- [x] 在 `main.py` 注册为 root command

### 后续

- [ ] zh/ 中文翻译
- [ ] en/ 人类文档渐进填充
- [ ] ai/ 文档开始生产
- [ ] CLI 命令树重构（将 docs 正式纳入 workspace→codex 体系）