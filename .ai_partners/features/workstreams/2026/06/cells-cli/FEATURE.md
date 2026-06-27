---
title: Cells CLI
status: in-progress
priority: P0
created: 2026-06-28
updated: 2026-06-28
depends:
  - matrix-cell-governance
milestone: 0.1.0
description: >-
  moss cells 命令行体系 — 统一取代 apps/script/runtime 三个旧 CLI 组。
  静态发现、生命周期管理、运行时治理、认知引导全部归入 cells 命令组。
---

# Cells CLI

> 人类架构师 + claude-opus-4-7。从 matrix-cell-governance 的设计结论出发，
> 定义 `moss cells` 的完整命令体系。

## Motivation

旧 CLI 有三个独立命令组 (`moss apps`, `moss script`, `moss runtime`)，
对应三种不同的 cell 管理模型。matrix-cell-governance 将它们统一为 cell
概念后，需要一个对应的命令行入口：静态发现、进程启动、运行时治理、认知引导。

核心原则：**模型是第一开发者**。CLI 不需要覆盖所有操作——specification 定义约定，
模型读 specification 后自己改 CELL.md。CLI 只做最薄的约定执行层。

## Key Decisions

### 命令树 (10 个命令)

```
moss cells
├── specification     # 认知入口: CELL.md 格式 + 五类 cell + launcher + 依赖策略
├── create <name>     # 从 stub 生成最简模板 (CELL.md + main.py + README.md + INSTALL.md)
│   --group <group>
├── list              # CELL.md 扫描，mode-aware
│   --installed       #   只显示已安装
│   --include/exclude #   fnmatch 筛选
├── show <path>       # manifest 全部字段 + instruction 全文
├── register <file>   # 外部脚本快捷方式 → workspace/cells/{group}/{name}/
│   --name --group
├── run <target>      # 目录/脚本/名称 三种模式 → spawn + 启动前信息反馈
│   [args...]         #   透传
├── install <name>    # 标记已安装 (touch .installed)，不做自动化
├── status [address]  # 运行时: 无参=列表, 带 address=单 cell 详情
├── kill <address>    # 强杀 + 清 runtime file
└── kill-all          # 强杀全部
```

### 五类 cell (launcher 配置差异，不是类型差异)

| 标签 | interpreter | 依赖 | 例子 |
|------|------------|------|------|
| standalone | python (sys.executable) | 零外部依赖 | moss_self |
| project | python (sys.executable) | 依赖 project (import MOSS.*) | playwright |
| isolated | .venv/bin/python | 独立 pyproject.toml, 最小依赖 ghoshell_moss[matrix] | feishu |
| script | /bin/bash | 无关 ghoshell，纯 shell | install 脚本 |
| remote | N/A | 独立 .moss workspace，靠 network/scope 通讯 | G1 PC2 |

全部是 CellType.worker。差异在 CELL.md 的 launcher 字段。

### specification 是认知入口

`moss cells specification` 输出 cell 开发完整约定，等同 cell 世界的 `moss start`。
涵盖: CELL.md 格式、五类 cell 详解、launcher 配置、依赖隔离策略、cell 发现规则。
每个 CELL.md stub 的 instruction body 首行写 `moss cells specification — cell development guide.`
作为必然被读到的 hint。

### CELL.md + README.md 文档分工

- CELL.md body (instruction): 模型入口 — 能力清单 + CTML 命令 + 怎么调用
- README.md: 人类开发者入口 — 环境准备 + 依赖 + 开发流程 + 已知问题
- 不创建 CLAUDE.md (旧 app 的 AI 开发者上下文)

### create 不加 --type

specification 定义约定，模型读后自己改 CELL.md。CLI 只生成一种最简 stub。
独立 venv、pyproject.toml、复杂依赖——模型自己加。

### enable 概念删除

无 enable/disable 命令。筛选通过 CellRegistry.list_cell_manifests(include=, exclude=)
的 fnmatch 模式 + 模式的 exclude_cell_paths。

### install 是标记，不是自动化

`moss cells install <name>` 只创建 .installed 空文件。INSTALL.md 存在表示需要
安装——Ghost/模型读 INSTALL.md 用 bash:exec 跑安装步骤。CLI 不替 cell 跑脚本。

### run 反馈启动信息，不替代 python main.py

`moss cells run` 走完整 spawn 协议: resolve manifest → 写 runtime file → spawn。
`python main.py` 是开发者的快捷路径 (Cell.from_proc 自发现)。CLI 不对后者负责。

## Implementation Notes

- 复用 `Project.discover()` → `project.cells` (CellRegistry) 做静态发现
- 复用 `CellRegistry.spawn_cell()` 做进程启动
- 复用 `CellRegistry.local_runtime_cells()` + `Cell.is_alive()` 做运行时查询
- 复用 `CellRegistry.recursively_kill_process()` + `kill_all_runtime_cells()` 做进程杀死
- `show` 的入参是 project-relative 目录路径，不做向上查找
- `register` 创建的 CELL.md 放在 workspace/cells/ 下
- 命令间无状态的 CLI 模式——每个命令独立 `Project.discover()`，不跨命令共享
