---
title: Desktop Channel — Shell 的 OS 交互工具集
status: in-progress
priority: P0
created: 2026-07-20
updated: 2026-07-20
depends: []
milestone:
description: >-
  建立 desktop channel 作为 MOSS Shell 操作面的工具组织层，挂载 bash / file_editor / ground，
  与 matrix (cell 治理) 平级。matrix 回归纯粹，不再直接挂 OS 工具。
---

# Desktop Channel

## Motivation

matrix 当前通过 `with_terminal` / `with_file_editor` 参数同时挂 cell 治理和 OS 工具，
两种不同语义混在一个集成点。system_test mode 被迫用 `_build_bash_with_file_editor`
在外面又挂一遍，说明组织方式自陷摩擦。desktop channel 作为 Shell 操作面工具的组织层，
与 matrix (cell 治理) 平级挂 main 下，各司其职。

## Key Decisions

### K1. desktop 是三工具的组织容器

desktop 是极简集成 channel，无 own commands。静态挂载:

- `bash` (terminal_channel) — 子进程执行
- `file_editor` (file_editor_channel) — 文件读写
- `ground` (未来) — 认知场；本轮不挂，就绪时加

三个是平级 sub-channel，寻址: `desktop.bash:exec`、`desktop.file_editor:view`。

### K2. matrix 不再挂 OS 工具

`build_matrix_channel` 的 `with_terminal` / `with_file_editor` 参数移除。
matrix 回归 cell 治理集成点: nodes / mesh 及其 virtual children。

### K3. desktop 命名不冲突 ground

K49 全库 `desktop→ground` 重命名针对的是 Ground 抽象层 (文件系统认知面)，
desktop channel 是另一个概念面 (Shell 操作面的工具集合)。命名空间不重合。

## Implementation Notes

- `_build_bash_with_file_editor` in system_test channels.py 删除，替换为 `build_desktop_channel()`
- `build_matrix_channel(with_terminal=True, with_file_editor=True)` 的 kwargs 移除，caller 不再传工具参数
- desktop_channel.py 参考 matrix_channel.py 的集成模式: `new_channel()` + `import_channels` 平级挂载
