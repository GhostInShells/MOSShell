---
created: 2026-05-11
depends: []
description: >-
  分离 FractalHub 与 FractalNodeProvider 抽象，实现协议无关的分形通讯。
  Hub 接受子节点注册，Provider 向远程 Hub 注册。zenoh 为首个 transport 实现。
id: zenoh-fractal
milestone: beta
priority: P0
status: in-progress
status_note: Hub/Provider 分离完成，发现机制升级为 subscriber + re-put，单测 5/5 全绿。TUI 改为 IoC 发现。MossRuntime 新增 get_fractal_hub()。修复 /moss.static() 缓存导致 fractal channel 不可见的 bug。待端到端验证。
title: Zenoh Fractal — Hub/Provider 分离与反向注册
updated: '2026-05-14'
---

# Zenoh Fractal — Hub/Provider 分离与反向注册

## Motivation

MOSS 的跨机器组网场景（如机器人开发板 → Mac）需要**反向注册**模式：
子节点知道父节点地址，主动连接并注册自己，而非依赖多播被动发现。

### 本轮重构的触发原因

初版实现将 **Hub（接受方）** 和 **Provider（提供方）** 混在一个类中，导致：

1. **身份混乱**：同一个类既做"监听并发现子节点"（下行），又做"向父节点注册自己"（上行）
2. **协议绑定**：如果 Hub 用 zmq 收、Provider 用 zenoh 发，旧抽象无法表达
3. **集成困难**：`moss-as-fractal` CLI、REPL 的 FractalServeState、Matrix 的自动发现，
   三个场景对 Fractal 的需求不同，却挤进同一个类

### 新设计的核心原则

- **Hub 和 Provider 是两个独立抽象**，各自可以有不同的 transport 实现
- **约定优于字符串**：通过 `FractalKeyExpressions` 标准化 key space，Hub 和 Provider 按约定对齐
- **协议无关**：当前只实现 zenoh，但抽象层不依赖 zenoh

## Design

### 抽象层 (`core/blueprint/host.py`)

```python
class FractalHub(ABC):
    """接受方：监听并发现子节点，通过 channel_hub 暴露给 AI"""
    name() -> str
    usage() -> str              # 协议自解释
    get_connected() -> list[Cell]  # 已连接子节点
    is_running() -> bool
    status() -> str             # 运行时状态
    as_channel(name, description) -> Channel  # 包装为 Channel，注册到 shell
    # 生命周期: __aenter__ / __aexit__

class FractalNodeProvider(ABC):
    """提供方：将本地 MossRuntime 注册到远程 Hub"""
    channel_provider(name) -> ChannelProvider | None
    is_running() -> bool
    provide(moss: MossRuntime, name: str) -> None  # 核心：暴露 moss 给远程
    # 生命周期: __aenter__ / __aexit__
```

两者都继承 `MossLifecycleContract`，通过 IoC 容器自动发现和生命周期绑定。

### 实现层 (`host/fractal/`)

包结构：

```
host/fractal/
  __init__.py          # 导出 FractalCell
  _base.py             # FractalCell, FractalKeyExpressions, 常量
  zenoh_fractal.py     # ZenohSessionFractalHub, ZenohSessionFractalNodeProvider
```

核心类：

| 类 | 角色 | 说明 |
|---|---|---|
| `FractalKeyExpressions` | Key space 约定 | 标准化 `{prefix}/{hub}/manifests/` 和 `{prefix}/{hub}/providers/` |
| `FractalCell` | Cell 实现 | `type='fractal'` 的节点描述 |
| `ZenohSessionFractalHub` | Hub 实现 | 独立 zenoh session + liveness 发现 + channel_hub |
| `ZenohSessionFractalNodeProvider` | Provider 实现 | 将本地 channel 暴露到远程 Hub 的 key space |
| `FractalHubChannelState` | Hub 的 ChannelState | 提供 `open_node`/`close_node` 命令给 AI |
| `ZenohFractalHubProvider` | IoC Provider | 从 workspace config 自动创建 Hub |

### Key Space 约定 (`FractalKeyExpressions`)

```
Manifests (节点发现):  {prefix}/{hub_name}/manifests/{node_name}
                       wildcard: {prefix}/{hub_name}/manifests/**

Providers (Channel):   {prefix}/{hub_name}/providers/{node_name}
                       wildcard: {prefix}/{hub_name}/providers/**
```

- `prefix` 默认为 `moss_fractal`，可自定义
- Hub 用 `manifests_wildcard()` 发现子节点
- Provider 用 `provider_node_address(node_name)` 确定自己的 channel 地址

### 集成路径

三种使用场景，各自独立：

1. **Matrix 自动集成**：`MossRuntimeImpl.__aenter__` 检查 `container.bound(FractalHub)`，
   存在则 `import_channels(fractal_hub.as_channel())` 到 shell。FractalHub 不自动启动，
   由上层决定何时 enter。

2. **REPL 调试观测**：`FractalServeState` 从 `runtime.get_fractal_hub()` 发现 Hub，
   提供 `/fractal.start|stop|status|connected|explain` 命令。不创建 Hub，不管理 channel。
   Hub 不存在时优雅提示。适合人类调试。

3. **CLI 独立进程**：`moss-as-fractal` 启动 runtime，创建 `FractalNodeProvider`，
   调用 `provide()` 向远程 Hub 注册。适合机器人等无人值守场景。

## Scope (本轮)

### 已完成

| 任务 | 状态 |
|---|---|
| Hub/Provider 抽象分离 (`blueprint/host.py`) | ✅ done |
| `FractalCell` + `FractalKeyExpressions` | ✅ done |
| `ZenohSessionFractalHub` 主体实现 | ✅ done |
| `FractalHubChannelState` (open_node/close_node) | ✅ done |
| `ZenohFractalHubProvider` (IoC 自动创建) | ✅ done |
| Matrix 自动集成 (检查 container.bound) | ✅ done |
| **`ZenohSessionFractalNodeProvider` 完整实现** | ✅ done |
| **旧代码 `ZenohSessionFractalHub2` 清理** | ✅ done |
| **单元测试修复 + 扩展 (5 tests, 全部通过)** | ✅ done |
| **`moss-as-fractal` CLI 对齐新 API** | ✅ done |
| **`FractalServeState` 对齐新 API** | ✅ done |
| **`FractalKeyExpressions.manifests_wildcard` bug 修复** | ✅ done |
| **Hub 发现机制: get() 轮询 → subscriber + re-put 心跳** | ✅ done |
| **`MossRuntime.get_fractal_hub()` API 新增** | ✅ done |
| **`FractalServeState` IoC 发现重构** | ✅ done |
| **`ZenohFractalHubProvider` 注册到 workspace stubs** | ✅ done |
| **`/moss.static()` 缓存 bug 修复** | ✅ done |

### 待完成

| 任务 | 优先级 |
|---|---|
| 端到端验证 (两个 workspace) | P1 |
| 环境发现链路验证 (providers/contracts) | P1 |
| 推拉混合发现 (liveness + queryable) | P2 |
| Fractal 体系 how-to 文档 | P2 |
| 环境发现使用 how-to 文档 | P2 |

### Out of scope (另开 feature)

- 多 transport scheme 支持 (zmq 等)
- `DuplexChannelProvider` 通用事件体系
- Fractal 从 blueprint 搬迁到独立层级

## Key Decisions

### 1. Hub 与 Provider 分离 (2026-05-13)

**决策**: 将 Fractal 拆为两个独立抽象：`FractalHub`（接受方）和 `FractalNodeProvider`（提供方）。

**理由**:
- 两种角色有不同的生命周期和依赖方向
- 一个节点可能只需要其中一种角色
- 不同 transport 可以独立组合（Hub=zmq, Provider=zenoh）
- 各自可独立测试，不依赖对方实现

### 2. Key Space 约定化 (2026-05-13)

**决策**: 通过 `FractalKeyExpressions` 类标准化 key space，不再散落字符串拼接。

**理由**:
- Hub 和 Provider 通过约定对齐，而不是口头约定
- manifests 和 providers 两个 namespace 语义清晰
- 复用于 zenoh、redis 等不同广播协议

### 3. IoC 自动发现而非隐式启动 (2026-05-13)

**决策**: Matrix/MossRuntime 检查 `container.bound(FractalHub)` 决定是否集成，
但不自动启动 Hub。

**理由**:
- 网络监听是有副作用的资源占用，不应每次启动 moss-repl 都自动打开
- 由上层（REPL state、CLI、应用代码）决定何时启动
- 参考了之前 FractalServeState 显式管理的经验

### 4. 暂不搬迁 Fractal 包 (2026-05-13)

**决策**: `host/fractal/` 暂时放在 `host` 下，不做跨层级搬迁。

**理由**: 当前只有 zenoh 一种实现，搬迁到 blueprint 层为时过早。
等有第二个 transport 实现时再提升抽象层级。

### 5. 先单测，后集成，最后 TUI (2026-05-13)

**决策**: 本轮开发的推进顺序为 Hub/Provider 实现 → 单测 → 集成 → TUI。

**理由**:
- 单测保证核心类稳定，集成和 TUI 才有可靠基础
- 避免之前在未稳定抽象上做 TUI 集成导致的返工
- 每步更新 FEATURE.md 维持意识连续性

### 6. NodeProvider 允许 provide 运行中的 Shell (2026-05-13)

**决策**: `ZenohSessionFractalNodeProvider.provide()` 覆盖 ABC 默认实现，
不检查 `moss.shell.is_running()`。

**理由**:
- ABC 的检查是"防蠢"：担心 shell 已启动时 channel 被占用
- 但 fractal 场景下 shell 必然已启动（moss-as-fractal 在 `async with moss_runtime` 内调用）
- Fractal 使用独立的 session_scope，不会与 Matrix 本地 channel 冲突
- 交由 `ZenohChannelProvider.arun_until_closed()` 处理实际冲突

### 7. Zenoh peer-to-peer 下的发现机制选择 (2026-05-13)

**发现**: Hub 的 `_query_cell_manifests` 使用 `zenoh.Session.get()` 查询 manifest 数据，
但这依赖 zenoh storage（router 或 declared storage）。在纯 peer-to-peer 模式下，
`put()` 的数据无法被 `get()` 查询到，导致 Hub 发现不了 Provider。

**方案评估** (4 种路径):

| 路径 | 机制 | 优点 | 缺点 |
|---|---|---|---|
| A | subscriber only (fire-and-forget) | 最简单 | 无心跳，无法感知节点离开 |
| B | queryable (Provider 声明 queryable) | Hub 可主动查询 | Provider 需额外声明 |
| C | subscriber + re-put 心跳 | 有 liveness 感知，实现简单 | 周期性 put 有带宽开销 |
| D | liveness + queryable | 最完整 | 实现复杂，zenoh liveness API 不稳定 |

**选择**: 路径 C — subscriber + re-put 心跳。

**理由**:
- Provider 周期性 re-put manifest 作为心跳，Hub subscriber 实时接收
- Hub 通过 `_prune_stale_cells()` 清理超时未心跳的节点 (stale_timeout = refresh_interval * 3)
- 实现简单，无需 zenoh storage 或 queryable
- 线程安全：subscriber callback 在 zenoh 后台线程，用 `threading.Lock()` + `time.monotonic()`

**实现要点**:
- Hub: `declare_subscriber(manifests_wildcard, callback)` → `_on_manifest_sample` 更新 `_cell_last_seen`
- Hub: `_refresh_loop` 周期性调用 `_prune_stale_cells()`
- Provider: `_reput_loop` 周期性 `put(manifest_key, cell_data)`
- 测试验证: `test_hub_discovers_provider` — 先确认发现，Provider 退出后确认 prune

**未来迭代** (P2): 推拉混合 — 在 subscriber 基础上叠加 liveness token，让 Hub 可以主动 query 节点状态，减少 re-put 带宽。

### 8. `get_fractal_hub()` 默认实现放在 ABC (2026-05-14)

**决策**: `MossRuntime.get_fractal_hub()` 在 ABC 中提供完整默认实现 `return self.matrix.container.get(FractalHub)`，而非 `return None`。

**理由**:
- 继承 `MossRuntime` 的实现类无需 override，自动获得正确行为
- 这是 code as prompt 的体现：ABC 默认实现本身就是 "正确做法" 的文档
- 如果未来有不需要 IoC 的 MossRuntime 实现，它仍可 override 返回 None
- 与 `.container`、`.session` 的默认实现模式一致

### 9. `moss-as-fractal` CLI 对齐抽象 API (2026-05-14)

**问题**: `moss_as_fractal.py` 手动创建 `ZenohSessionFractalNodeProvider` 实例、
配置 zenoh session、管理生命周期，与 `FractalNodeProvider` ABC 的设计重复。

**改动**:

1. **`moss_as_fractal.py` 简化** — 删除手动实现，改用 `host.provide_moss_as_fractal(provider=None)`。
   Host 自动从 IoC 容器发现 `FractalNodeProvider`，CLI 只负责环境配置和启动。

2. **新增 `ZenohFractalNodeProviderProvider`** — IoC Provider 类，
   从 workspace config (`zenoh_config_fractal.json5`) 自动创建 `ZenohSessionFractalNodeProvider`。
   与 `ZenohFractalHubProvider` 对称。

3. **providers.py 注册** — stubs 和 `.moss_ws`、`.test_ws` 的 `providers.py` 均添加
   `fractal_node_provider = ZenohFractalNodeProviderProvider()`。

**理由**: CLI 不应绕过 IoC 体系手动创建实例。
`provide_moss_as_fractal()` 是 MossHost 的标准 API，自动发现 + 生命周期管理。

### 10. `/moss.static()` 缓存失效修复 (2026-05-14)

**现象**: Hub 启动且节点连接后，`/moss.static()` 不显示 fractal channel 和 `open_node` 命令。
但 `<moss:open_node name="moss_provider"/>` 实际可以执行。

**根因**: 两层缓存导致 `/moss.static()` 返回启动时的陈旧快照：

1. **`_moss_static_cache` 永不失效** — `static_messages()` 在 shell 启动时首次调用，
   此时 Hub 未启动，fractal channel 的 `is_available()=False`。
   结果缓存后永不刷新（`_refresh_moss_static` 默认 `False`）。

2. **`_last_channel_metas_refreshed_at` 永为 0** — 初始化为 0 但从未更新，
   导致 `channel_metas()` 的 0.5s 时间检查永远返回陈旧缓存。

**修复** (3 个文件):

| 文件 | 改动 |
|------|------|
| `core/ctml/shell/ctml_shell.py` `refresh_metas()` | 同时清除 `_moss_static_cache` |
| `core/ctml/shell/ctml_shell.py` `channel_metas()` | 设置缓存后更新 `_last_channel_metas_refreshed_at` |
| `host/repl/inspector_moss_runtime.py` `static()` | 改为 async，先调 `moss_refresh_metas()` 再取 static |
| `host/moss_runtime.py` | 新增 `moss_refresh_metas()` 公共方法 |

**效果**: `/moss.static()` 现在先刷新 metas（重新评估所有 channel 的 `is_available()`），
再生成 static 输出。fractal channel 和已 open 的节点 channel 正确显示。

### 11. FractalServeState 从创建者变为观测者 (2026-05-14)

**决策**: `FractalServeState` 不再自己创建 Hub 和管理 channel，改为从 `runtime.get_fractal_hub()` 发现已有 Hub。

**理由**:
- Hub 已由 Matrix/IoC 自动创建和集成 channel，REPL state 不应重复
- State 的角色收敛为 "调试外壳"：发现 → 观测 → 交互
- Hub 不存在时不报错，优雅提示 "No FractalHub configured"
- 未来加回调注册观测通讯时，state 职责清晰：订阅 Hub 事件 → 推到 REPL 显示

**影响**:
- 不再管理 channel 添加/移除（Matrix `__aenter__` 已做）
- 不再读取 config 文件（Hub 自己读自己的）
- `_FractalOps` 每个方法先检查 `hub is None`

## Implementation Notes

- 单元测试 5 个全部通过：
  - `test_node_provider_connectivity`: NodeProvider expose channel → proxy 连接 → 执行命令
  - `test_hub_discovers_provider`: Hub subscriber 接收 Provider re-put → 发现节点；Provider 退出 → Hub stale prune
  - `test_hub_and_provider_coexist`: Hub 和 NodeProvider 生命周期共存
  - `test_node_provider_double_enter_rejected`: NodeProvider 重复 enter 抛出 RuntimeError
  - `test_hub_double_enter_rejected`: Hub 重复 enter 抛出 RuntimeError
- 发现机制: Hub 通过 `declare_subscriber` 监听 `manifests/**`，Provider 通过 `_reput_loop` 周期性 re-put manifest 作为心跳。Hub 在 `_refresh_loop` 中通过 `_prune_stale_cells()` 清理超时节点 (stale_timeout = refresh_interval * 3)。
- 线程安全: subscriber callback 运行在 zenoh 后台线程，使用 `threading.Lock()` 保护共享 dict，使用 `time.monotonic()` 做线程安全的时间戳。
- `FractalKeyExpressions.manifests_wildcard()` 原有 bug：`manifests_prefix()` 已返回
  `{prefix}/{hub}/manifests`，再拼接 `/manifests/**` 导致路径变成
  `{prefix}/{hub}/manifests/manifests/**`。已修复为 `/**`。

## Design Index

- 抽象定义: `src/ghoshell_moss/core/blueprint/host.py` (FractalHub, FractalNodeProvider, MossRuntime.get_fractal_hub)
- Hub/Provider 实现: `src/ghoshell_moss/host/fractal/zenoh_fractal.py`
- IoC Provider: `src/ghoshell_moss/host/fractal/zenoh_fractal.py` (ZenohFractalHubProvider)
- Key space 与 Cell: `src/ghoshell_moss/host/fractal/_base.py`
- Matrix 集成: `src/ghoshell_moss/host/moss_runtime.py` (__aenter__, moss_refresh_metas)
- REPL State: `src/ghoshell_moss/host/repl/fractal_serve_state.py`
- REPL Inspector: `src/ghoshell_moss/host/repl/inspector_moss_runtime.py`
- Shell 缓存逻辑: `src/ghoshell_moss/core/ctml/shell/ctml_shell.py`
- CLI 入口: `src/ghoshell_moss/cli/moss_as_fractal.py`
- 环境注册 (stubs): `src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/providers.py`
- Hub 配置: `src/ghoshell_moss/host/stubs/workspace/configs/zenoh_config_fractal_hub.json5`
- 单元测试: `tests/ghoshell_moss/host/test_zenoh_fractal.py`

## Continuation Trajectory (零上下文恢复指南)

以下三个命题按顺序执行。每个命题完成后更新本文件。

---

### 命题 1: TUI IoC 发现重构 (✅ 已完成, 2026-05-14)

**背景**: 旧 `FractalServeState` 自己创建 `ZenohSessionFractalHub` 实例、管理生命周期、添加/移除 virtual channel，完全绕过 IoC 发现体系。这与 "Hub 由 Matrix 自动集成" 的设计冲突。

**改动内容**:

1. **`MossRuntime.get_fractal_hub()` 新增** (`core/blueprint/host.py:156`):
   ```python
   def get_fractal_hub(self) -> 'FractalHub | None':
       return self.matrix.container.get(FractalHub)
   ```
   将 IoC 隐式查询提升为 Runtime 一等公民 API。与 `.apps`、`.shell`、`.matrix` 同模式。

2. **`FractalServeState` 重写** (`host/repl/fractal_serve_state.py`):
   - 不再自己创建 Hub，改为 `runtime.get_fractal_hub()` 发现
   - 不再管理 channel add/remove（Matrix 已处理）
   - Hub 存在 → 提供 `/fractal.start|stop|status|connected|explain`
   - Hub 不存在 → 优雅提示 "No FractalHub configured"
   - 生命周期管理收敛为 Hub 的 enter/exit

3. **`ZenohFractalHubProvider` 注册** (`host/stubs/workspace/src/MOSS/manifests/providers.py`):
   - 添加 `from ghoshell_moss.host.fractal.zenoh_fractal import ZenohFractalHubProvider`
   - 添加 `fractal_hub_provider = ZenohFractalHubProvider()`
   - 这是 stubs 约定，新 workspace 自动包含此 provider

**发现的架构张力**:
- Hub 不启动时 `as_channel()` 返回的 channel 已被 Matrix 导入 shell，但底层 transport 未激活
- 这是正确的 "lazy start" 模式：channel 在树中可见，但需 `/fractal.start()` 后才真正工作
- 未来可加 `is_running()` 状态同步到 channel 的 dynamic message

**依赖**: 命题 2 验证 IoC 链路是否真正贯通。

---

### 命题 2: 环境注册验证 (待人类执行)

**目标**: 验证 FractalHub 从环境声明到运行时发现的完整链路。

**验证路径** (按顺序执行):

```bash
# Step 1: 确认 FractalHub contract 已在代码中定义
.venv/bin/moss --ai codex get-interface ghoshell_moss.core.blueprint.host:FractalHub
# 预期: 看到 name(), usage(), get_connected(), as_channel(), status() 等抽象方法

# Step 2: 确认 ZenohFractalHubProvider 已实现
.venv/bin/moss --ai codex get-interface ghoshell_moss.host.fractal.zenoh_fractal:ZenohFractalHubProvider
# 预期: contract() -> FractalHub, factory() -> ZenohSessionFractalHub

# Step 3: 确认 config 文件存在
ls src/ghoshell_moss/host/stubs/workspace/configs/zenoh_config_fractal_hub.json5
# 预期: 文件存在，内容为 peer mode + random port

# Step 4: 确认 FractalHub 出现在 contracts 中
.venv/bin/moss --ai manifests contracts ghoshell_moss.core.blueprint.host:FractalHub
# 预期: 如果已注册，显示 contract 条目；如果未注册，需要完成 contract 注册步骤

# Step 5: 确认 ZenohFractalHubProvider 出现在 providers 中
.venv/bin/moss --ai manifests providers | grep -i fractal
# 预期: 看到 ZenohFractalHubProvider 条目

# Step 6 (可选): 创建新 workspace 验证完整链路
.venv/bin/moss ws init test_fractal
cd test_fractal
.venv/bin/moss --ai manifests providers | grep -i fractal
.venv/bin/moss --ai manifests contracts | grep -i fractal
```

**当前状态**:
- `ZenohFractalHubProvider` 已注册到 stubs `providers.py`
- `FractalHub` contract 注册机制待确认（可能需要额外的 contract 声明）
- `.moss_ws` 是本地 workspace，需手动同步或重建

**阻塞点预判**:
- `moss manifests contracts FractalHub` 返回 0 结果 → contract 注册机制需要单独处理
- 新 workspace 的 `providers.py` 来自 stubs 拷贝 → stubs 更新后新 ws 自动包含

---

### 命题 3: How-To 文档 (待端到端验证完成后)

**前置条件**: 命题 2 验证通过，人类完成端到端测试。

**需记录的 how-to**:

1. **Fractal 体系 how-to** (`how_tos/for-moss-app-developer/fractal-system.md`):
   - FractalHub vs FractalNodeProvider 的角色区别
   - 配置 zenoh_config_fractal_hub.json5
   - CLI: `moss-as-fractal` 的用法（子节点注册到父 Hub）
   - REPL: `/fractal.start|stop|status|connected` 的调试流程
   - Key space 约定 (FractalKeyExpressions)
   - 常见问题: peer-to-peer 发现限制、stale prune 行为

2. **环境发现使用 how-to** (`how_tos/for-moss-core-developer/environment-discovery.md`):
   - `moss manifests providers` — 查看已注册的 IoC providers
   - `moss manifests contracts` — 查看已绑定的 contracts
   - 如何在 `providers.py` 中注册新 provider
   - `MossRuntime.get_fractal_hub()` 等便利 API 的发现模式
   - container.get() vs 显式 API 的设计权衡

**创建方式**: 参考 `moss how-tos read how-to-make-how-to.md` 了解格式约定。

---

## Related

- Related: `ai-native-feature-tracking`
