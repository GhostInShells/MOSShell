---
created: 2026-05-22
depends:
- first-ghost-prototype
description: 原计划为 Ghost 提供统一多级隔离 Storage 入口。后因 Matrix 自身接管整套环境存储声明，缩并为 GhostWorkspace
  dataclass。
milestone: null
priority: P0
status: dropped
status_note: 2026-07 转向 ground (原 project-manager) 的实现，ghost-ground 取代了 playground 的认知定位。
step: superseded_by_matrix_storage_declaration
title: Ghost Playground — 多级隔离文件空间 → 缩并为 GhostWorkspace
updated: '2026-08-11'
---

# Ghost Playground → GhostWorkspace

> 2026-06-02 修订：GhostPlayground 被缩并为 GhostWorkspace。Matrix 接管了所有存储声明。

## 原始设计（已废弃）

原计划通过 GhostPlayground ABC 将 workspace/session/home 三种 Storage 组织为单一入口，
让 ghost 子功能（memory, personality, scratchpad...）有明确的 scope 选择。
ABC 在 `blueprint/host.py`，默认实现 + Provider 在 `host/ghost_playground.py`。

## 为什么缩并

在后续开发中发现：**Matrix 自身应该做整套环境声明。** Matrix 是通讯拓扑和资源管理的单一事实源，
它已经持有 Workspace + Session，所有存储路径约定应该由 Matrix 统一声明，而不是再建一个
GhostPlayground 中间层去桥接。

Matrix 在 staged 改动中新增了完整的存储树声明：

```python
# Matrix ABC 上新增
ghosts_storage   → workspace/ghosts/
modes_storage    → workspace/modes/
ghost_home       → workspace/ghosts/{ghost_name}/
mode_home        → workspace/modes/{mode_name}/
```

既然 Matrix 已经声明了 `ghost_home`，GhostPlayground 的存在就是多余的了——它只是在 Matrix
和 Ghost 之间做了一次无意义的转发。

## 当前形态：GhostWorkspace

GhostPlayground 缩并为 `GhostWorkspace` — 一个只有两个字段的 dataclass，定义在
`blueprint/ghost.py`：

```python
@dataclass(frozen=True)
class GhostWorkspace:
    home: pathlib.Path      # host 为 ghost 分配的持久化存储区域
    source: Optional[pathlib.Path]  # ghost 源代码所处环境
```

- `home`: 来自 `matrix.ghost_home`（路径约定 `workspace/ghosts/{name}/`）
- `source`: ghost 模块的源码目录，用于 soul 文件加载等

`GhostWorkspaceProvider` 在 `host/ghost_runtime.py`，是 Provider[GhostWorkspace] 的简单适配器。
Ghost 和 GhostMeta 通过 `container.get(GhostWorkspace)` 获取。

## 变更清单

| 操作 | 文件 |
|------|------|
| 删除 | `host/ghost_playground.py`（GhostPlaygroundImpl + GhostPlaygroundProvider） |
| 从 ABC 移除 | `blueprint/host.py` — GhostPlayground class |
| 新增 | `blueprint/ghost.py` — GhostWorkspace dataclass |
| 新增 | `host/ghost_runtime.py` — GhostWorkspaceProvider |
| Matrix 新增 | `blueprint/matrix.py` — ghost_home / ghosts_storage / modes_storage 等存储声明 |
| Stub 新增 | `stubs/workspace/ghosts/` — ghosts/None/, ghosts/echo/ 目录结构 + soul.md |

## 后续影响

- `ghosts/atom/_meta.py` 的 `_load_soul()` 和 `build_agent()` 改用 GhostWorkspace（随本次 unstaged 改动完成）
- Ghost 子功能需要存储时，直接从 IoC 拿 `GhostWorkspace` 或直接 `matrix.ghost_home`