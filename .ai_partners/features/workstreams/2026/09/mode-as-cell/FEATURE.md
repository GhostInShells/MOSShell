---
title: Mode as Cell — 把 mode 作为 fractal cell 挂到 Matrix 网络
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-02
updated: 2026-09-02
depends: []
milestone: alpha
description: >-
  拉起一个进程, 把本 mode 的 NodeManager 作为唯一 channel 挂到 Matrix 网络,
  供远端 host accept 后远程治理本 mode 的 nodes 集合. 消解老 fractal 的
  Hub/Provider/独立 transport 抽象, 完全复用 Matrix CellNetwork.
---

# Mode as Cell — 把 mode 作为 fractal cell 挂到 Matrix 网络

> Use `moss features set-status mode-as-cell <status> -m "note"` to update state.

## Motivation

### 问题

MOSS 目前的 host 是"本地治理面 + 网络投影"的组合: `matrix` channel 内含 `nodes`
(本地 NodeManager) + `mesh` (远端 accepted cells). 但当前网络里只有 host 拉起的
node 子进程 (通过 `matrix.run_node()`), 缺一类"远端整 mode 作为可远程治理的
cell"—— 即, 一台设备把自己的一套 mode (含 terminal / motor / 摄像头等
nodes) 完整暴露给另一台设备的 ghost, 让远端 ghost 能远程 `list / read / run
/ stop` 这套 nodes.

典型场景: g1 机器人板子上跑一个独立 mode, Mac 上的 ghost accept 它后, 通过
`fractal.g1_dev.list()` / `run("terminal")` 直接治理板子上的 nodes.

### 为什么现在

- Matrix 层已经落地 (matrix-operator testing, matrix-manifest-layers completed,
  node-lifecycle completed, cells-cli completed): CellNetwork 的 accept / mesh
  projection / virtual_children 已跑通.
- `mesh` channel 已经把 accepted cells 挂成 `virtual_children`. 剩下缺的只是
  "远端 cell 提供的是什么 channel".
- 老 fractal 体系 (zenoh-fractal / zmq-fractal, 2026-05 completed 后在
  `f0d6d2cf` / `affdfd58` massive refact 期间被清理) 遗留的正是这个能力空位.

### 为什么值得重做, 又为什么这次简单

老 fractal 的复杂度来自: FractalHub / FractalNodeProvider 双抽象 + 独立
zenoh session + 独立 key space (`FractalKeyExpressions`) + Matrix 检查
`container.bound(FractalHub)` 集成路径 + REPL state 从创建者退化为观测者 +
shell 缓存失效连锁 bug.

这次全部消解: Matrix CellNetwork 已经会做"cell 入网 + 远端 accept + virtual
children 挂载". 我们要做的只是 "拉起一个进程, 让它 providing 一个 channel,
这个 channel 就是本 mode 的 nodes channel". 一切走现有基建, 不引入任何新
抽象.

## Design

### 概念: Mode as Cell

一个 mode 不只是本地 host 内部的治理域, 它可以作为独立 cell 拉起, 把自己的
`NodeManager` 治理面作为唯一 channel 挂到 Matrix 网络. 别的 host accept 后就
能在 `fractal.<short>.*` 下调用本 mode 的 nodes 命令.

对比老 fractal: 不再有 Hub/Provider 双抽象, 不再有独立 transport session, 不
再有独立 key space. Matrix CellNetwork 自己就是 transport, mesh 自己就是发
现, cell providing channel 自己就是"上线".

### Cell 身份

| 字段 | 值 | 理由 |
|---|---|---|
| `role` | `NODE_ROLE` (`'node'`) | 定义就是功能性节点, 非 host |
| `category` | `'fractal'` | 具体分类, 承载"整 mode 作为可远程 accept 的 cell"语义. 需扩展 `Cell.category` 的 Literal 集合 |
| `name` | `{project_name}_{mode_name}` (经 `normalize()`) | 承载"谁家的哪个 mode", 例: `mosshell_default` / `g1robot_dev` |
| `uid` | 默认 | address 已含 uid 保证唯一性, name 只承载可读信息 |
| `singleton` | `True` | 同机同 project+mode 拉两个 fractal 语义混乱 (谁治理谁的 nodes). DuplicatedError 是正确表达 |
| `persist` | `True` | 长期驻留, 等 host accept, 非一次性 |
| `event_level` | `None` (=INFO) | host 需感知上线/下线, 默认可感知档 |
| `providing` | `['channel']` | 单一 channel = nodes channel |

### 启动命令

**`moss-shell fractalize`** — moss-shell 的新子命令, 全局 `--mode` 参数选目
标 mode.

- 复用 `moss-shell` 现有的 mode 解析 / workspace 绑定 / matrix boot 路径
- 语义直白: 一个 shell 运行时, 以 cell 身份对外只暴露 nodes 面
- 区别 human debug 的 `moss-shell` 主命令 (暴露完整 main_channel + desktop +
  matrix 等) —— `fractalize` 是"把自己交出去"

内部草图 (细节开工时确认):

```python
matrix = Matrix.new(
    node_name=f"{normalize(project_name)}_{normalize(mode_name)}",
    category='fractal',
    persist=True,
    singleton=True,
)
async with matrix:
    nodes_channel = new_nodes_channel(matrix)
    await matrix.provide_channel(nodes_channel)  # 具体 API 开工时看 service.ServiceOperator
    await matrix.wait_forever()                  # 阻塞至外部 stop
```

### Host 端

无需新命令. 现有:

- `mesh:accept <address>` (或 auto_accept) 批准 fractal cell
- Matrix 现有 CellNetwork 自动把它挂进 `matrix.mesh` 作 virtual_children
- 命令空间自动出现在 `fractal.<short>:list / read / run / stop / status /
  read_output`

## Scope

### In

- `moss-shell fractalize` CLI 子命令 (子命令注册 + mode 解析复用)
- `Cell.category` Literal 集合追加 `'fractal'`
- 复用 `new_nodes_channel(matrix)` 作为唯一 providing channel — 不改一行
  nodes channel 逻辑
- 端到端本地验证: 同机两进程 (host + fractal), host `mesh:accept` 后能通过
  `fractal.<short>:list / run` 治理 fractal 侧 mode 的 nodes
- 至少一条 how-to (README 段落 / 短 doc): 场景说明 + 启动步骤 + 验证方法

### Out

- **跨机器验证**: 同机联通即算 done. 跨机器是网络配置事 (防火墙 / NAT /
  transport endpoint), 非本 feature 逻辑
- **嵌套 fractal**: fractal cell 不再暴露 mesh, 不做多级嵌套. 如需, 加另一
  台真 host
- **权限 / 鉴权**: 走 Warrant 体系后续演进, 本 feature 走 mesh 现有
  `set_auto_accept` 默认策略
- **老 fractal 词汇**: 不引入 Hub / Provider / KeyExpressions / 独立
  transport 任何一个概念
- **transport 层修改**: Matrix CellNetwork 底层实现零改动

## Key Decisions

### 1. category = 'fractal' 而非 'moss' (2026-09-02)

**决策**: 新增 `Cell.category` Literal 值 `'fractal'`, 承载本 cell 类型.

**理由**:
- category 是分类语义, 越具体越承载信息
- `'moss'` 太泛 —— 所有 host / node 本质都是 moss 实例, 当分类反而空转
- `'fractal'` 一词准确表达角色 (整 mode 作为可被远端 accept 的 cell), 且延续
  用户脑海中的历史概念词 (虽然实现全变), 词义连续对未来 review 者更友好

### 2. name = {project_name}_{mode_name} (2026-09-02)

**决策**: cell name 由 project_name + mode_name 拼接, 经 `normalize()` 归一
化非法字符 (`-` / `.` → `_`).

**理由**:
- name 只需承载"谁家的哪个 mode"这层可读信息, 唯一性由 address 尾部 uid 兜底
- 归一化保证符合 `CellNamePattern`
- 示例: `mosshell_default`, `g1robot_dev`

### 3. 只暴露 nodes, 不带 mesh (2026-09-02)

**决策**: fractal cell 的 providing channel 只包含 `new_nodes_channel(matrix)`,
不复用 `matrix_channel` 的完整 matrix / mesh 结构.

**理由**:
- mesh 是 host 的全景视图, fractal cell 不做多级嵌套
- 简化远端接管的心智: 一个 fractal 就是"远端 mode 的 nodes 治理面", 语义单一
- 若真需要嵌套 (fractal 再 accept 别的 fractal), 应该加一台真 host, 不该在
  fractal 层做

### 4. singleton = True, persist = True (2026-09-02)

**决策**: fractal cell 声明 singleton (同 project+mode 只允许一个) + persist
(长驻).

**理由**:
- singleton: 同机拉两个同 project+mode 的 fractal 语义混乱 —— 谁的 nodes 治
  理面覆盖谁? DuplicatedError 是正确表达
- persist: fractal 是长期在网等 accept 的角色, 不是一次性任务; event_level 默
  认 (=INFO) 让 host 能感知上线/下线

### 5. 复用 moss-shell 而非新建 entry point (2026-09-02)

**决策**: 启动命令为 `moss-shell fractalize` (子命令), 不新建
`moss-fractal` / `moss-cell` 独立 entry point.

**理由**:
- 复用 moss-shell 现有的 mode 解析 / workspace 绑定 / matrix boot 路径, 零重
  复代码
- 语义: fractalize 是"这个 shell 以 cell 身份对外交出"的动作, 天然是
  moss-shell 的一个模式
- 独立 entry point 会引入 CLI 表面 + 打包声明 + 文档三处冗余, 收益不匹配

### 6. 不复活老 fractal 抽象 (2026-09-02)

**决策**: 不引入 FractalHub / FractalNodeProvider / FractalKeyExpressions /
独立 zenoh session 中任何一个概念. 全部走 Matrix CellNetwork.

**理由**:
- 老 fractal 的复杂度 (见 `.ai_partners/features/workstreams/2026/05/zenoh-fractal/FEATURE.md`)
  完全来自"另起一套 transport + 另起一套发现 + 与 Matrix 双轨集成"
- 当前 Matrix 已经承担这三件事, 再抽象一层就是无价值的多态
- 未来若需多 transport, 是 Matrix CellNetwork 的事, 不是本 feature 的事

## Implementation Notes

### 开工前调研清单

必读 (按顺序):

1. `src/ghoshell_moss/core/blueprint/cell.py` — `Cell` / `NodeManifest` /
   `Matrix.new()` / `NODE_ROLE` / `CellNamePattern` / `normalize`
2. `src/ghoshell_moss/core/blueprint/matrix.py` — `Matrix.discover` /
   `Matrix.new` / `network()` / `provide_channel` (或等价 API, 见 4)
3. `src/ghoshell_moss/channels/matrix_channel.py` — `new_nodes_channel` 直接复
   用
4. `src/ghoshell_moss/core/blueprint/service.py` (`ServiceOperator`) — cell 对
   外提供 channel 的具体 API, 结合 `matrix-operator` feature 现状确认落地形态
5. `moss --ai features status matrix-operator` — 网络层最新状态, 确认 cell
   providing channel 的当前接口
6. `src/ghoshell_moss/cli/` — `moss-shell` 子命令注册模式, 参照现有子命令实
   现 `fractalize`

### 潜在 gotchas

- **name 冲突处理**: `normalize()` 需覆盖 `-` / `.` / 空格等. project_name /
  mode_name 各自若已合法, 直接拼接; 存在特殊字符时需归一
- **singleton 语义边界**: singleton 是"同治理域" (mode 层级) 唯一. 若同一物
  理机器上跑两个不同 project (不同 workspace), 各自有独立 project_id, 不冲
  突
- **project_id 承载 fractal 归属**: fractal cell 的 `Cell.project_id` 应从当
  前 Environment 取, host 端 `is_local()` 判定基于此
- **event_level 感知**: fractal cell 的上线 / 下线走 `CellEventLevel.INFO`,
  host 端 mesh channel 已经把这层转成 Signal (matrix_channel.py:496
  `_dispatch_event`), 无需新代码
- **旧 `moss-as-fractal` binary 遗留**: `.moss_ws/apps/sensors/voice/.venv/bin/moss-as-fractal`
  等旧安装残留, 与本 feature 无关, 不清理

### 验收标准

同机两进程验证:

```
# Terminal A (host)
moss-ghost <ghost>
  → 观察 mesh 中出现 fractal cell 待 accept 的事件

# Terminal B (fractal)
moss-shell fractalize --mode <mode>
  → cell 上线, 阻塞等 accept

# Terminal A (host, AI 侧)
<mesh:accept name="<fractal short>"/>
<fractal.<short>:list/>
  → 返回 fractal 侧 mode 的 nodes 清单

<fractal.<short>:run target="<node>"/>
  → 在 fractal 侧拉起该 node, 返回 handle
```
