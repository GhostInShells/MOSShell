---
created: 2026-05-24
depends: []
description: 提取 CellDiscovery 独立类，增加 on_refresh_meta async hook。Hub 缝合留待未来 判断：统一被动发现
  channel vs 多个手动配置 channel。
milestone: null
priority: P0
status: completed
status_note: 'CellDiscovery extracted, on_refresh_meta added. Hub缝合留待未来判断: 统一被动发现
  vs 多个手动配置 channel.'
title: Matrix Channel Hub — Zenoh 原生 Channel 发现与管理
updated: '2026-06-05'
---

# Matrix Channel Hub

## Motivation

当前 Channel 发现链路存在两层断裂：

1. **Matrix.provide_channel() 是统一的提供入口**，但发现靠外部知识（AppStore 枚举 → 逐个 proxy），网络层没有自反射能力。
2. **AppStoreChannel 通过 manifests 注册**，但 Matrix 才是通讯拓扑的单一事实源。这导致两个"发现"概念（cell liveness + channel proxy）各自维护。

这次改造的核心判断：**Matrix 上既有 provide 接口，就不该是 apps 单独发现 channel。应该有一个 Matrix Channel，在 bootstrap 时获取 Matrix 实例，用 states_channel 的 virtual channel 模式自动反射网络中所有可用的 Channel。**

同时吸收 `cell-discovery-refactor`：用 zenoh queryable/wildcard 替代 N 个 liveness subscriber，把 cell 存活和 channel 发现统一到同一个 key 体系。

## Design Index

- `BridgeExpr` Protocol + `NodeChannelBridgeExpr` 可配置化：`src/ghoshell_moss/bridges/zenoh_bridge/_utils.py`
- `ZenohChannelHub`：新建 `src/ghoshell_moss/bridges/zenoh_bridge/zenoh_hub.py`，参考 `src/ghoshell_moss/bridges/zmq_channel/zmq_hub.py`
- `MatrixChannel` + `MatrixChannelState`：新建，基于 `states_channel.py` 模式
- `MatrixImpl.provide_channel()` / `channel_proxy()`：`src/ghoshell_moss/host/matrix.py`
- 最终移除 manifests 中的 AppStoreChannel 注册

## Key Decisions

### D1: BridgeExpr 是 Protocol，NodeChannelBridgeExpr 是基线实现（无继承）

```python
@runtime_checkable
class BridgeExpr(Protocol):
    """Channel bridge key expression 的接口契约。"""
    address: str
    session_scope: str
    bridge_prefix: str
    provider_liveness_key: str
    proxy_liveness_key: str
    provider_receiver_key: str
    proxy_receiver_key: str


class NodeChannelBridgeExpr:
    """基线实现。prefix_parts=None 时用约定前缀。"""

    NODE_BRIDGE_PREFIX_TEMPLATE: ClassVar[str] = "MOSS/{session_scope}/node/{address}/channel_bridge"

    def __init__(self, address: str, session_scope: str, *, prefix_parts: list[str] | None = None):
        # prefix_parts=None → 约定优先: "MOSS/{scope}/node"（向后完全兼容）
        # prefix_parts=["MOSS", "{scope}", "matrix"] → hub 模式
```

**Why Protocol**: 它 provide 什么比它怎么 implements 重要。Provider/Proxy/Hub 通过 Protocol 感知接口契约，`isinstance(obj, BridgeExpr)` 可用于结构类型检查。`NodeChannelBridgeExpr` 不加父类，零糖。

**约定优先于配置**: `prefix_parts=None` 时用默认 node 前缀。ZenohProviderConnection / ZenohProxyConnection 接收可选 `bridge_expr: BridgeExpr | None = None`，不传用 `NodeChannelBridgeExpr`。

### D2: ZenohChannelHub 建立在 BridgeExpr 之上，不硬编码 key

Hub 内部用 `NodeChannelBridgeExpr(address, scope, prefix_parts=["MOSS", "{scope}", "matrix"])` 生成通讯 key。

Channel 发现 key 和通讯 bridge key 共享同一个 BridgeExpr 体系，不再分成两套独立 key 空间。Hub 不编造任何 key，所有 key 通过 BridgeExpr 实例生成。

Zenoh liveliness token 原生提供 PUT/DELETE 语义——provider 断开时 key 自动 DELETE，无需手动心跳超时清理。

### D3: Matrix Channel 用 ChannelState.get_virtual_children() 做动态发现

bootstrap 时注入 Matrix 实例。`get_virtual_children()` 通过 hub 的 wildcard query 获取所有已注册 provider → 为每个动态生成 proxy 作为 virtual child。与 ZMQHubChannelState 同构。

### D4: cell-discovery-refactor 吸收合并

cell liveness key 和 channel discovery key 统一在 matrix prefix 下。一个 wildcard query 覆盖两者。不再需要 per-cell liveness subscriber。

### D5: AppStoreChannel 最终从 manifests 移除

Matrix Channel 管理 app 的 open/close，运行中的 app 自动出现在 virtual children 里。AppStoreChannel 的概念合并进 Matrix Channel。

## Implementation Plan (修订顺序)

原计划 Step 1 Hub → Step 2 BridgeExpr 是错的——Hub 依赖 BridgeExpr 提供 key 体系。修订为：

1. **BridgeExpr Protocol + NodeChannelBridgeExpr 可配置** — 向前兼容。Provider/Proxy 加可选 `bridge_expr` 参数。
2. **ZenohChannelHub** — 建立在 BridgeExpr 之上，`as_channel()` 集成 Shell。独立验证。
3. **Matrix Channel** — 新建 ChannelState，用 hub 做动态发现。
4. **替换 manifests 注册** — 移除 AppStoreChannel，链路回归。

## Blast Radius

| 改动 | 影响 |
|------|------|
| BridgeExpr Protocol + NodeChannelBridgeExpr.prefix_parts | __init__ 签名向后兼容，现有调用者无需修改 |
| ZenohProviderConnection 加 bridge_expr 参数 | 可选参数，默认行为不变 |
| ZenohProxyConnection 加 bridge_expr 参数 | 可选参数，默认行为不变 |
| 新建 ZenohChannelHub | 零影响现有代码 |
| 新建 MatrixChannel + MatrixChannelState | 零影响现有代码 |
| MatrixImpl 改用 hub key | provide_channel/channel_proxy 内部调整，接口不变 |
| 移除 AppStoreChannel manifests 注册 | 需回归验证 Shell 树中 apps channel 仍可用 |

### D6: channel_proxy 降为内部实现

如果 hub 做得好（auto-discovery + create_proxy），`Matrix.channel_proxy()` 不再需要暴露为公开 API。
当前 Matrix ABC 的 docstring 已经写了"这个函数除特殊情况外，不需要手动使用"。Hub 成熟后可以移出公开接口。

### 2026-05-26 Walkthrough 验证

通过 MCP 走通了完整流程：`moss apps create` → 写码 → `apps:start` → proxy 连接 → CTML 调用。关键发现：

- **发现链路**: 当前是 manifests 静态发现（AppStoreChannel.get_virtual_children → app_store.list_apps → matrix.channel_proxy）
  目标替换为 hub 网络发现（hub.registered_channels → hub.create_proxy）
- **生命周期管理**: AppStore 管进程（circus start/stop），Hub 管发现（zenoh wildcard），MatrixChannel 缝合两者
- **open_app 的 wait_connected**: 利用 tree 的 refresh_metas → get_virtual_children → bootstrap proxy → fetch_sub_runtime → wait_connected 链条
- **模型屏蔽**: MatrixChannel 的呈现层不暴露 zenoh/hub/proxy 底层概念，模型看到的是 `open_app`/`close_app` 和 channel 树

## 2026-05-26 探索记录 (Claude Opus 4.7 结对编程)

### 已完成的提交

**Step 1: BridgeExpr Protocol + NodeChannelBridgeExpr 可配置** — done, merged.

- `_utils.py`: `BridgeExpr` 是 `@runtime_checkable` Protocol (Python 3.10 兼容)。`NodeChannelBridgeExpr` 保持基线实现，不加父类，`prefix_parts=None` 时约定优先。
- `_provider.py` / `_proxy.py`: Connection 层接收可选 `bridge_expr: BridgeExpr | None = None`，不传用 `NodeChannelBridgeExpr`（向后完全兼容）。
- `matrix.py`: 零改动。

### 失败的尝试 (已删除)

**Step 2-3: zenoh_hub.py + matrix_channel.py** — 放弃，人类主程重写。

AI 在交付模式下写的代码质量显著低于 review 模式下应有的标准。具体问题：

1. **没有 KeyExpressions 权威类** — 参考 `FractalKeyExpressions`（`src/ghoshell_moss/host/fractal/_base.py`），它是所有 key 结构的唯一真相来源，Hub 和 Provider 共享此类保证对齐。AI 的 hub 用 `_bridge_for("*")` hack wildcard，用字符串切割提取 address，全是魔法。
2. **错误的注册模型** — 最初写了 `zenoh_register_channel`/`zenoh_unregister_channel` 命令式 helper，违背了 duplex 通讯的声明式 liveness 发现原则。BridgeExpr 的 liveness 本身就是注册信号，不需要额外桥接两套 key 空间。
3. **耦合 MatrixImpl 生命周期** — 一开始把 hub 放在 MatrixImpl 里，后来放在 MatrixChannel.bootstrap 里用 `loop.create_task` 制造副作用。正确的模式是 ChannelState 的 `on_startup`/`on_close` 管理生命周期。
4. **容器注册是模式模仿不是必要** — hub 注册到 IoC 容器时没有追问"谁需要它"。实际上 `zenoh_register_channel` 是独立函数不需要 hub，只有 MatrixChannelState 需要，而它通过构造器获取。

### 关键认知

- **duplex 通讯的 liveness 本身就是发现** — `tests/ghoshell_moss/bridges/test_bridge_suites.py` 展示：provider 和 proxy 共享 BridgeExpr key 空间，provider 声明 liveness → proxy subscribe 同一个 wildcard → `wait_connected()` 自动连通。不需要额外的注册步骤。
- **FractalKeyExpressions 是正确的参考模式** — 发现 key 空间和通讯 key 空间由同一个权威类管理。manifest 数据自带 name，不需要从 key 字符串解析。
- **hub 应该传入 BridgeExpr 就能创建 proxy 和 provider** — 关键是"传入 BridgeExpr"这个动作本身定义了 key 空间，hub 的所有操作从它派生。
- **MatrixImpl 保持零感知** — hub 的生命周期绑定到 ChannelState.on_startup/on_close，Matrix 不需要知道它的存在。
- **AppStoreChannel 不急于移除** — Matrix Channel 和 AppStoreChannel 可以共存，AppStore 管进程生命周期，Hub 管网络发现，MatrixChannel 缝合两者。

### 代码质量教训

- AI 在"交付模式"下倾向于堆功能而非提炼抽象，私有函数/魔法值/字符串切割泛滥。
- 在"review 模式"下同一 AI 能识别这些问题的严重性。两种模式差距几个档次。
- 对抽象质量要求高的内核代码，应以人类主程 + AI 结对（AI 协助实现单体函数、写测试、做 review）的方式推进，而非 AI 独立写整个模块。

### 2026-06-05: CellDiscovery 提取 (DeepSeek V4 Pro)

MatrixImpl 中 6 个内嵌的 cell liveness 方法（~70 行）提取为独立类 `CellDiscovery`，
放在 `host/cell_discovery.py`。遵循 `FractalKeyExpressions` 的 key 权威类模式：

- `liveness_prefix()` / `liveness_key(addr)` / `liveness_wildcard()` — key 空间唯一真理源
- `declare_this_cell(session, address)` — context manager, liveness token 声明
- `discover_cells(session, cells, events, this_address)` — context manager, 订阅 + 初始 wildcard query
- 9 个单测覆盖 key 表达式、declare、discover、disconnect、initial query、多 cell

MatrixImpl 集成：`_session_communication_bus_ctx_manager` 改为调用 CellDiscovery，
内部逻辑保持不变。旧 6 方法删除。

同时修复 `test_environment_set_mode.py` 的 `MOSS_CELL_ADDRESS` 跨测试泄漏。

配套基础设施：`on_refresh_meta` async hook (commit 0b6d01d)，为 hub 在 refresh 周期
做 I/O（alive_cells 查询、proxy 缓存更新）提供 async 入口，与 sync 的
get_virtual_children() 分离。

### 下一步

1. **Hub + MatrixChannel** — 在 CellDiscovery 基础上构建 wildcard 动态发现,
   MatrixChannel 缝合 AppStore(进程管理) + Hub(网络发现)
2. **AI 协助单体函数、写测试、验证**
3. **AI 最终 review 架构**

### 验收点 (AI review checklist)

- [ ] 是否存在 `MatrixKeyExpressions` 或等价类，承载所有 key 结构的唯一真相？
- [ ] Hub 和 provider 注册侧是否共享同一个 key 结构权威类（参考 FractalKeyExpressions）？
- [ ] Hub 的 liveness wildcard、proxy 创建、address 解析是否全部从 key 权威类的同一个入口派生（不是每个操作各自推导）？
- [ ] 是否存在字符串切割提取 address 的逻辑（应为反模式，除非在 key 权威类内部且充分测试）？
- [ ] Hub 生命周期是否绑定到 ChannelState.on_startup/on_close，而非 MatrixImpl 或 fire-and-forget task？
- [ ] MatrixImpl 是否有零改动（hub 不应耦合 Matrix 生命周期）？
- [ ] 是否保留了 AppStoreChannel 作为进程生命周期管理（不急于移除）？
- [ ] MatrixChannel 的模型侧呈现是否屏蔽了 zenoh/hub/proxy 底层概念？
- [ ] `zenoh_register_channel` 式命令式注册是否消失（liveness 声明即注册）？
- [ ] `channel_proxy` 是否降为内部实现或移除出公开 API？
- [ ] 过一遍完整的 apps 发现 + script 验证流程

## Absorbed Features

- `cell-discovery-refactor` (completed 2026-05-24, absorbed here): liveness subscriber N→1, is_alive→reported_at, queryable 替代 pub/sub