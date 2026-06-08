---
title: Matrix Cell Governance
status: draft
priority: P0
created: 2026-06-09
updated: 2026-06-09
depends:
  - cell-discovery-refactor
  - cell-session-bootstrap
milestone: 0.1.0
description: >-
  Matrix cell 体系治理总任务。从 circusd 死胡同到 node 体系：
  重新定义 cell 类型、发现模型、最小依赖闭包。Node 取代 app/script，
  fractal 回归 cell 身份，host 成为拓扑聚合者。环境变量三个变一个。
  进程管理三件套：start_new_session + pipe fencing + polling。
  不碰现有 apps 代码，先建 parallel node 线。
---

# Matrix Cell Governance

> 人类做代码架构，模型 review。此 feature 记录动机、共识和推进方法。

## Motivation

当前 MOSS 的 cell 体系有三套互相竞争的机制：circusd 进程守护（AppStore）、
zenoh queryable 发现（cell discovery）、以及裸 asyncio subprocess（ManagedProcess）。
加之 cell 类型混乱——app/script/fractal/host 的边界从未被严格定义——
导致开发者在"如何启动一个 cell"这件事上各自为政。

circusd 被设计为独立系统守护进程，它的核心能力（重启、监控、web dashboard）
面向部署场景，而非 per-session 的进程图。把它塞进 host 进程的子进程位置，
造成双层通讯（host → ZMQ → circusd → watcher → 子进程）、
孤儿泄漏（host SIGKILL 后 circusd 不知道宿主已死）、
以及两套监控体系（circus status vs zenoh queryable）。

**真命题**：MOSS 是通讯总线，不是进程管理器。Cell 的"存活"由网络查询决定，
不由本地守护进程决定。需要一套统一的 cell 治理框架：身份定义、发现机制、
生命周期契约、最小依赖闭包。

## 共识结论

### Cell 类型的重新定义

旧四类 → 新三类：

- **host** — 保留。由 MODE.md 定义，是 session 入口。负责聚合拓扑、提供 queryable。
- **node** — 新增。worker cell。App 和 script 归一为此类型。
  被 host spawn → 角色是 app（host 管理生命周期）。
  外部进程 → 角色是 node（外部管理生命周期）。
  同一份 CELL.md / NODE.md 描述，启动者决定角色。
- **fractal** — 保留为 cell 类型。不是"不是 cell"——它是拥有 address、
  可在 Matrix 总线上被寻址的参与者。只是自己不提供 channel，
  由 host 通过 FractalHub 注入到 queryable。

`CellType.app` 消失。App 是 node 被 host spawn 后的运行时角色，不是静态类型。
`CellType.script` 消失。被 node 取代。

### Cell 的唯一定义

拥有 address、可在 Matrix 总线上被寻址的参与者。
不问"你是什么类型"，只问"你的 address 是什么"。

### 发现模型

host 是拓扑聚合者。三种路径注入 cell 到 queryable：
1. node 自宣告（自己的 queryable）
2. host spawn（host 自然知道它的存在）
3. fractal peer（FractalHub 注入）

在 host 的 queryable 里，所有 cell 扁平化为 `address → Cell`。
观察者（UI 面板、REPL）看到统一的网络拓扑。

### 两轴状态模型

活跃轴和可用轴正交：

```
活跃轴（liveness）           可用轴（availability）
──────────────────          ──────────────────────
在线 / 离线                  已安装 / 未安装 / 停用
Zenoh queryable 判定         文件系统状态
"进程在跑吗"                 "模型该看见它吗"
```

可用轴的三个状态：

- **未安装** — 注册表里有记录（`moss nodes register` 写入），代码未在环境里。
  模型不可见。只是一段元数据。
- **已安装** — 代码就位。模型可见，可被 bringup。
- **停用** — 代码在但标记 disabled。文件在，cell 可能在跑，但模型不可见。
  `moss nodes disable/enable` 切换。

cell meta 文件（`cells/cell-{address}.json`）已经承载 PID 验活和僵尸杀灭——
它本质上是**离线声明**。加一个 `enabled: bool` 字段即可同时表达可用状态。
host 在构建 queryable 结果时按可用轴过滤。

当前 apps 的痛点：`moss apps list` 列出的是静态目录遍历，代码未安装的 app
模型也能看见甚至尝试调用。这是混淆了注册（存在）和可用（可调用）。

### 最小依赖闭包

node 加入 Matrix 网络的最小依赖：`ghoshell_moss[cell]` → `eclipse-zenoh>=1.8.0`。
core（blueprint + concepts + contracts + duplex）+ bridges/zenoh_bridge。
不需要 host、cli、speech、circus。

### 环境变量回归一个

`MOSS_WORKSPACE` 是唯一必须的环境变量。
`MOSS_SESSION_SCOPE` 默认 `"default"`，需要隔离时显式传。
`MOSS_CELL_ADDRESS` 从 CELL.md 推导或自动生成。
`MOSS_PARENT_PID` 由启动器自动注入，node 用 `== 0` 判断是否需要跟随生命周期。

### 进程生命周期

三件套（全覆盖正常退出和 SIGKILL）：
1. `start_new_session` + `killpg` — 优雅退出时父进程一刀切
2. Pipe fencing — 父进程 SIGKILL 时子进程零延迟自检
3. Polling（`_ensure_parent_process_exists`）— 已有的兜底，不留

不做进程编排（restart / health check / bringup）。那是 Channel 的能力，不是框架的职责。

### 启动 node 的三种方式

1. 纯 Python：`eval $(moss shell-init) && python my_cell.py`
2. moss 命令：`moss nodes run my_node`
3. cell 间：`matrix.run_cell("my_node")`（待建 Process Nursery）

### CELL.md 最小格式

```yaml
name, group       → cell_address = app/{group}/{name}
executable         → 启动器（python / uv / bash）
script             → 入口文件
arguments          → 额外 CLI 参数
description        → 人类可读
```

type 字段不在 CELL.md 中。类型由启动者赋予。

### Skills 与 Meta Channel

Node 的 dev/debug 脚本不需要反射为 virtual channel。方案极简：

- meta channel 接受 `paths` 配置项，指向已安装 node 的 skills 目录
- 扫描 `--help` 输出，聚合到 channel instruction 里
- ghost 看到 skill 列表，通过 `bash:exec`（AI Terminal 已提供）调用

这本质上是 `moss all-commands` 的 channel 化——meta channel 就是 CTML 语境下的
能力目录。无需 virtual channel、动态命令挂载。

## 迭代路径

1. **不碰 apps**：现有 App 体系保留，circusd 不去。bug fix 照常。
2. **建 node 并行线**：`CellType.node`、NODE.md、`moss nodes` CLI、注册表。
3. **Matrix 接口补齐**：Process Nursery、`matrix.run_cell()`、pipe fencing。
4. **环境变量契约文档化**：`moss shell-init`、最小启动示例。
5. **验证完毕后才讨论 apps 迁移**：当 node 机制覆盖了 apps 的所有场景。

## Open Questions

- 注册表格式：目录约定 vs 显式 registry yml？
- Pipe fencing 的 asyncio 集成：`loop.add_reader` vs 独立 watchdog task？
- fractal 连接到 host 的 channel proxy 语义：透明转发还是显式声明？
- 可用轴状态存储在 cell meta 文件还是独立注册表？倾向于 cell meta 加 `enabled` 字段。
