# Fractal: 通讯发现模式与下一步实现

## 讨论背景

2026-05-10, 讨论 zenoh 多实例在相同 IP 上随机端口是否会冲突, 以及 Fractal 分形通讯的设计方向。

## 核心结论

### Zenoh 多实例不冲突

- `listen: ["tcp/0.0.0.0:0"]` = 随机端口, 同一台机器多个实例不会端口冲突
- Zenoh 本地发现靠 UDP 多播 scout, 不依赖固定端口
- 跨网络需要至少一个固定入口 (router 或固定端口 peer)

### 四种通讯发现模式

| 模式 | 方向 | 适用场景 |
|------|------|----------|
| 1. 约定发现 | Main 固定锚点, App 主动连接 | Matrix 内部层级拓扑 |
| 2. 正向注册 | Provider 推给 Registry | MCP 模式 (有泄漏问题) |
| 3. 反向注册 | g1 主动注册到电脑 | 具身→控制中心 (你的场景) |
| 4. 双向契约 | 共享 key space 互发现 | 对等 Matrix 协作 |

### Fractal 选型: 反向注册

- g1 (具身) 知道电脑在哪, 电脑不需要知道 g1 在哪
- g1 主动 `provide_channel` 给电脑 — 方向正确
- MCP 方向是反的: Consumer → Server, 不适合你的场景

### 最简实现路径

```
电脑 Matrix 启动
  → fractal zenoh session 随机端口
  → 打印自身地址 (如 tcp/192.168.1.x:45678)

g1 侧:
  moss fractal register --to tcp/192.168.1.x:45678 --transport=zenoh
```

## 明天 (2026-05-10) 实现计划

### Step 1: `moss --mode --session_scope`

把乱散的状态参数迁回 moss CLI 全局参数:
- `--mode`: 选择运行模式
- `--session_scope`: 隔离通讯网络

### Step 2: `moss fractal register`

```
moss fractal register --to <addr> --transport=zenoh
```

- 只实现 zenoh transport, 不抽象
- `--to` 指定父节点地址 (电脑的 fractal listen 地址)
- 底层: `ZenohSessionFractal` + `connect.endpoints` 指向 `--to`

### Step 3: 脚本验证

用两个脚本跑通 channel 通讯:
1. 电脑侧: 启动矩阵, 打印 fractal 地址
2. g1 侧: `moss fractal register --to <电脑地址>`, provide channel
3. 验证电脑侧 `channel_hub` 能看到 g1 的 channel

### Step 4: 再决定集成到哪个抽象

跑通之后再讨论:
- 是 CLI 直接操作 Fractal?
- 还是封装到 Matrix 生命周期?
- 还是独立 fractal 子进程?

## 关键约束

- **不固化 Fractal 抽象** — 先跑通一个场景, 再抽象
- **不配 scout** — 不做零配置发现, 明确方向是反向注册
- **fractal 只做 zenoh** — `--transport=zenoh` 目前只有一个实现

## 当前未提交改动

- `matrix.py`: `resources()` 方法, import 整理
- `zenoh_fractal.py`: 异步 liveness 刷新, `FractalCell.mark_dead()`, 生命周期管理
- `environment.py`: `MOSSMeta` → `MossMeta`
- `manifests.py`: `resource_storages()` 方法
- `manifests/__init__.py`: `PackageResourceStorages` 集成
- `manifests/resource_storages.py`: 新文件
- `ghosts/atom/_runtime.py`: pydantic_ai Agent 集成

明天开始编码时, 可能需要先 decide 这些改动是 staging 还是暂留。
