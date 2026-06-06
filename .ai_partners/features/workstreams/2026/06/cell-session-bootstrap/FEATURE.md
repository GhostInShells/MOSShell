---
title: Cell Session Bootstrap
status: draft
priority: P0
created: 2026-06-07
updated: 2026-06-07
depends: []
milestone:
description: >-
  Cell 入网协议——独立进程启动时从 workspace 恢复 session scope、分配唯一 cell 地址、注册到 Matrix 总线。替代 circusd-daemon-management 方向，MOSS 回归轻量总线定位。
---

# Cell Session Bootstrap

> 此 feature 是 `circusd-daemon-management` 的替代方向，来自 2026-06-07 与人类工程师的架构校正。

## Motivation

当前 MOSS 通过 App 体系（`HostAppStore` → circusd 子进程）跑通了 AIOS PoC，但方向有走得过远的倾向：开始在 Python 层做进程生命周期治理——circusd daemon 化、bringup 编排、不同类型节点分类。

**这是错误的**。MOSS 是 Shell/Bus 层，不是 kernel/init。进程生命周期治理是 OS 的事，每个操作系统都有自己的 init 系统。MOSS 不应在 Python 里再实现一个 init。

**真问题**：基于 Zenoh 组网 + Cell Discovery + Channel 生命周期，任何安装了 `moss[host]` 的 Python 脚本都可以组网。真正需要解决的问题很窄——独立进程启动时如何正确并网：

1. 从 workspace 恢复 session context（哪个 workspace？哪个 mode？哪个 ghost？哪个 session_id？）
2. 在 Zenoh key space 中获得不冲突的唯一 cell 地址
3. 注册到 Matrix 总线，让其他 cell 可发现

答案在 workspace 文件系统里。Workspace 是治理根——持有 session 状态、cell 注册表、地址分配记录。不需要框架级进程管理器。

## 核心架构决策

### 1. Matrix 是 bus，不是 kernel

Matrix 传递信号，不管理进程生命周期。进程怎么启动、用什么进程管理器（circusd/systemd/手动后台运行）、什么时候退出——都是外部选择。Matrix 只关心：进程启动后能否接入总线。

### 2. Workspace 是唯一的治理根

Workspace 文件系统持有：
- Session scope（mode/ghost/session_id 绑定）
- Cell 注册表（已分配的地址、租约、状态）
- 环境变量模板（进程启动时恢复上下文用）

不需要额外的注册中心服务——文件系统就是状态。

### 3. Process orchestration is a Channel

进程编排和视觉感知、屏幕截图一样——是能力，不是框架基建。如果系统中运行了 circusd 或 systemd，有权限的 Channel 可以通过它们的接口做 bringup。MOSS 不特殊对待进程管理——它只是一个普通的 Python 函数签名 Channel。

### 4. Bringup = 启动脚本，不是 app

Mode 的 bringup 逻辑本质是启动脚本。当前 `MODE.md` 中的 `bringup_apps` 字段（指定 mode 启动时自动拉起哪些 app）长期应回归为外部启动脚本或直接被外部系统托管。MOSS 不实现 bringup 编排。

## 与现有实现的关系

- App 体系现有实现不动。能 work 就继续 work，保持 PoC 验证能力。
- `AppStoreChannel` 已是可选。在注释中标记正确方向。
- 长期 App 体系会被本 feature 定义的 cell 入网协议替代——届时进程编排回归为普通 Channel。
- 本 feature 不改变 `matrix-channel-hub` 的接口，只收缩 AppStore 的职责范围。

## 设计要点

### Cell 入网协议

一个独立进程要从"裸 Python 脚本"变成"Matrix 上的 cell"，需要：

1. **Workspace 发现**：从环境变量或父进程继承 workspace 路径
2. **Session scope 恢复**：从 workspace 读取当前 mode/ghost/session_id → 注入 env
3. **Cell 地址分配**：在 workspace 的 cell 注册表中原子分配唯一地址
4. **Matrix 接入**：用分配的地址启动 Zenoh session，注册到总线
5. **租约维护**（可选）：周期性续约，进程死亡后地址自动回收

### Cell 注册表

`workspace/runtime/cells/` 目录：

```
cells/
  0001.json  # {address, session_scope, mode, ghost, pid, created_at, lease_until}
  0002.json
  ...
```

- 文件名 = 序号，内容 = cell 元信息
- 分配策略：扫描已有文件，取最大序号 + 1；或复用已过期租约的序号
- 租约过期 → 地址可回收，下一个进程启动时清理

### 启动脚本生成

`moss cell bootstrap` 命令生成一段 shell 脚本：

```bash
# 从 workspace 恢复 session context
export MOSS_WORKSPACE=/path/to/workspace
export MOSS_MODE=desktop
export MOSS_CELL_ADDRESS=cell/0003
export MOSS_SESSION_ID=abc123

# 启动进程
python my_script.py
```

或直接 `eval "$(moss cell bootstrap --mode desktop)"`。

## Open Questions

1. **Cell 注册表的并发安全**：多进程同时分配地址时如何避免竞态？文件锁（workspace lock）是最小可行方案。不需要分布式协调。

2. **租约粒度**：cell 应该多频繁续约？对于开发场景（REPL/TUI），30s 足够。对于生产场景（长期运行的感知 app），应支持更长的租约或"永久"模式。

3. **与现有 `MOSS_CELL_ADDRESS` 的关系**：当前 `Environment` 通过环境变量传递 cell address。本 feature 的注册表机制是对此的规范化——从"手动设置 env"变为"workspace 自动分配"。

4. **App 体系何时替换**：本 feature 先定义协议和注册表格式。App 体系继续工作。替换时机：当 matrix-channel-hub 足够稳定、cell 注册表格式验证通过后，逐步迁移 App 的启动路径到本协议。
