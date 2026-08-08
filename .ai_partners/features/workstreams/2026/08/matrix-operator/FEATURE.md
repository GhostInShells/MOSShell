---
created: 2026-08-07
depends: []
description: 'Matrix 层 cell 级服务化通讯接线层. ServiceOperator 封装 zenoh queryable/pub/sub
  + 生命周期治理, 暴露 ServiceDeclaration/ServiceProvider/ServiceClient 三分体.
  二级抽象 (webview/resource/channel) 各自定义自己的 service kind 与元数据,
  operator 永不装线. 关键词: "统一无聊层, 特别化有趣层".'
milestone: v0.1.0
priority: P0
status: in-progress
status_note: 'blueprint converged (service.py), directory topology settled, ready for zenoh implementation'
title: Matrix Service — cell 级服务化通讯接线层
updated: '2026-08-09'
---

# Matrix Service

> 命名: operator 概念保留 (接线员 — Switchboard Operator), 但上层抽象命名为
> Service — 强调业务语义. ServiceOperator 是接线员, ServiceDeclaration 是
> 身份声明, ServiceProvider 是运行时接线端子.

## Motivation

端到端弧线: 从 desktop-gui / screen-node / webview 三条线汇聚, 讨论了
提供/通知/生命周期/发现/code-as-prompt, 最终收敛到一个命题:

> 在一个基于广播的全双工 mesh 中, cell 级能力通讯需要两原语: query (有址请求-应答)
> + pub/sub (topic 流式). 每一个能力 (channel / resource / webview / ...) 都是
> 这两原语的组合, 辅以 meta 声明 + 生命周期治理. **统一无聊层, 特别化有趣层.**

当前痛点:
- channel 有自己的 duplex hub + provide_channel, resource 有自己的 provide_resource,
  webview 什么都没有 — 三种能力三套装线, 每加一种能力就要到 matrix 里再装一次线.
- 没有统一的"模型读一个 howto → 独立实现 disposable 能力"的路径.
- webview 需要 on_created/on_stopped 通知 + badge 红点 + discover 发现, 这些机制
  已经在 presence/liveness 里隐式存在, 但没有被显式化为一个通用算子.

## Design Index

- 上游碰撞轨迹 (不含在本目录, 但本 FEATURE.md 吸收了所有结论):
  - `desktop-gui` FEATURE.md + discuss — 审批流、Matrix 集成
  - `screen-node` FEATURE.md + discuss — 合成器、cell providing 协议升级钩子、badge 拦截
  - `matrix-resources` FEATURE.md — `provide_resource` / `servers://` URL 发现
  - `.discuss/2026-08-08_network_os_operator_bus_necessity.md` — Network OS 命题碰撞,
    Operating/System 二分, bash `|` 的 network 等价物
- 当前代码:
  - `src/ghoshell_moss/core/blueprint/service.py` — ABC 定义 (零 zenoh 依赖)
  - `src/ghoshell_moss/core/blueprint/matrix.py` — Matrix 上的装线方法
  - `src/ghoshell_moss/matrix/operator/` — zenoh 实现 (ZenohOperator, ZenohServiceProvider)
  - `src/ghoshell_moss/matrix/services/` — 具体业务 service (webview, resource...)

## Key Decisions

### K1. 两个通讯原语, 服务端三 slot, 不发明 send/listen

zenoh `Session.get` 已支持 payload 双向 body. **query/queryable 承载所有有址一次性通讯**
(请求-应答, 包括 fire-and-forget), **pub/sub 承载所有流式通讯** (事件/状态流).

Server 端 (ServiceProvider) 暴露三个 slot:

| slot | 方向 | 机制 |
|---|---|---|
| `queryable(key, handler)` → Handle | client → server 请求-应答 | zenoh queryable |
| `pub(key, payload)` | server → 订阅者流式 | zenoh pub |
| `listen(key, handler)` → Handle | client → server 事件 | zenoh sub |

Client 端 (ServiceOperator) 暴露对应的三个操作:

| 操作 | 方向 | 机制 |
|---|---|---|
| `get(kind, key, params, *services)` → list[Reply] | client → service(s) 请求-应答 | zenoh get |
| `sub(kind, key, handler, *services)` → Handle | client 订阅 service(s) 流 | zenoh sub |
| `emit(kind, key, payload, *services)` | client → service(s) 单向事件 | zenoh pub |

删掉的 send/listen: query body + 不 await 就能做 fire-and-forget; 流式 inbound 就是
client 往 server key pub, server sub. send/listen 不是新原语, 是 query 或 pub/sub 的用法模式.

`*services: ServiceMeta` vararg 允许定向或广播: 空 = 所有 kind 的 service, 传具体
ServiceMeta = 定向到特定 service.

### K2. Operator 拥有异步桥, API 面统一收 async handler

zenoh callback 跑在 zenoh 自有线程 (sync), MOSS 是 asyncio. **operator 内部用
janus.Queue 做桥** (g1 listener 两次验证的样板): zenoh 线程 callback enqueue →
app loop 单点消费 → 跑 async handler → 结果经 to_thread 桥回 zenoh reply.

implementer 永远不碰 zenoh 线程, 写干净的 `async def handler(...)` 即可 "读一个 howto
→ 独立实现 disposable 能力". 没有这条, 每个能力都要自己学 janus.Queue + drain 模式.

### K3. provide 单例, kind 级地址, 二级抽象自己定义接口

`operator.provide(declaration)` 返回 `ServiceProvider` **单例** — 一个 kind 一个 cell.
handler 注册跟 provider 实例生命周期治理.

"不由 Matrix 自己创建实现, 它的二级抽象自己定义接口" — capabilities 在
`matrix/services/` 下有自己的目录 (如 `matrix/services/webview/`), 含
ServiceDeclaration / ServiceServer / ServiceClient 的具体实现. operator 是薄基底.

### K4. ServiceDeclaration + from_meta 构造不变量

一个 service kind 的元数据:

```python
class ServiceDeclaration(BaseModel, ABC):
    @classmethod
    def kind(cls) -> str: ...        # 约定, 如 "webview"

    def to_meta(self, address: CellAddress) -> ServiceMeta: ...
    # → {"address": ..., "kind": "webview", "data": {...}}

    @classmethod
    def from_meta(cls, meta: ServiceMeta) -> Self | None: ...
    # kind 不匹配返回 None, 匹配则 model_validate(meta['data'])
```

**不容违反的不变量: `meta.data` 必须能 model_validate 回对应的 ServiceDeclaration**

发现→使用是闭环: 拿到 ServiceMeta 就能 `from_meta` 回 ServiceDeclaration,
declaration 携带 kind-specific schema. 失败 = kind bug, 不该静默.

与旧设计的差异: 去掉了独立的 ProtocolRegistry — `from_meta` classmethod 本身就是
registry 的分形实现, 每个 kind 自描述, 不需要中心注册表.

### K5. address 格式: cell 地址为锚

ServiceMeta.address 是 cell 在 mesh 上的地址. discovery 按 kind 或 address 两个维度
枚举 — `get_services_by_kind()` / `get_services_by_address()`.

zenoh key 推导在 operator 内部: `{cell_addr}/{kind}/{slot}/{business_key}`.
blueprint 层不暴露 zenoh key 格式 — implementer 只接触 business key.

### K6. on_service_start/stop 回调 + get_connected 拉取

两种发现模式互补:
- `on_service_start(kind, callback)` → Handle — push: service 上线立即通知
- `on_service_stop(kind, callback)` → Handle — push: service 下线通知
- `get_services_by_kind(kind)` — pull: 当前在线列表
- `ServiceClient.get_connected()` — pull: client 视角的已连接 service

Handle 统一 — 所有注册持久回调的方法 (`queryable`, `listen`, `sub`,
`on_service_start`, `on_service_stop`) 都返回 Handle, `close()` 取消.

### K7. channel 不现在回填

channel 的装线特殊 (duplex hub + CTML Shell 内建拦截层 + FutureRouter).
webview/resource 先按 operator 长, pattern 被两三个实例验证后再决定 channel 是否回填.

### K8. liveness 暂不放入 V1

liveness 在分形下一层才需要 — 一个 service 的子组件挂了, 需要通知. 当前没有这个场景.
ServiceProvider 接口上留余地, 以后疼了再加 `declare_liveness` 不破坏兼容性.

---

## 接口定义 (已落地代码)

见 `src/ghoshell_moss/core/blueprint/service.py` — 所有 ABC 已定义, 零 zenoh 依赖.

核心类型:

| 类型 | 角色 |
|---|---|
| `ServiceMeta(TypedDict)` | 发现身份: address, kind, data |
| `ServiceDeclaration(BaseModel, ABC)` | kind 自描述 schema: `kind()`, `to_meta()`, `from_meta()` |
| `Query(TypedDict)` | 请求: address, key, payload, timestamp |
| `Sample(TypedDict)` | 发布/订阅数据: address, key, payload, timestamp |
| `Reply(TypedDict)` | 应答: address, key, payload, timestamp |
| `Handle(ABC)` | 可关闭句柄: `key`, `close()` |

核心 ABC:

| ABC | 角色 |
|---|---|
| `ServiceProvider` | serve 侧运行时句柄: `meta`, `queryable()`, `pub()`, `listen()`, enter/exit |
| `ServiceOperator` | 接线员: `provide()`, `get_services_by_kind/address()`, `on_service_start/stop()`, `get()`, `sub()`, `emit()`, enter/exit |
| `ServiceServer` | serve 侧生命周期包装: `declaration`, `provider`, `new(matrix)`, enter/exit |
| `ServiceClient` | client 侧生命周期包装: `get_connected()`, `new(matrix)`, enter/exit |

Matrix 上的装线 (见 `matrix.py`):

```python
# 懒加载 accessor, 遵循 network() 的 lazy-gate 模式
async def service_operator(self) -> ServiceOperator: ...

# 便捷装线 — 创建并注册到 Matrix 生命周期
async def serve_service(self, service_cls: Type[ServiceServer]) -> ServiceServer: ...
async def connect_service(self, client_cls: Type[ServiceClient]) -> ServiceClient: ...
```

---

## Validation Plan

两段验证, 第一段是保险丝.

### V1: Disposable Counter/Echo (半天)

模型读本 FEATURE.md + 一个 howto → **独立**写一个运行即丢弃的 counter/echo service:
- ServiceDeclaration 子类 + ServiceProvider (queryable handler: counter 增量, echo 原样返回)
- ServiceOperator client get + sub
- lifecycle (on_service_start/stop 观察到)

判据: 模型是否感觉"最舒服" — 是否只靠本文档 + howto, 不查 zenoh/matrix 内部源码,
独立走通. 如果这一步卡了, operator 形状死了, 停下来改.

### V2: Webview + Screen-Node Red-Dot (半天到一天)

- `webview` service kind: WebViewDeclaration (id, label, url, icon) + ServiceProvider +
  pub(badge_changed, ...)
- screen-node: consumer 侧 — `operator.get_services_by_kind("webview")` → 发现所有 webview →
  meta 进游离层 + `operator.sub("webview", "badge", handler)` → QML 红点
- 端到端判据: screen-node 运行, 一个 webview cell 上线 → screen 游离层自动浮出
  meta item, 页面调 badge → 红点亮起. 零手工 open, 全自动.

---

## Implementation Notes

- **目录拓扑**:
  ```
  blueprint/service.py          ← ABC (零 zenoh 依赖)
  matrix/operator/               ← zenoh 实现层
    _utils.py                    ← ServiceKeyspace + ServiceKeyExpr (key 表达式单一来源)
    zenoh_operator.py            ← ZenohOperator (implements ServiceOperator)
    zenoh_service_terminal.py    ← ZenohServiceTerminal (implements ServiceProvider)
  services/                      ← 一级 package, 具体业务 service
    counter.py                   ← V1 disposable validation case
    webview/                     ← webview service kind (后续)
    resource/                    ← resource service kind (后续)
  ```
- operator 的 zenoh 实现复用现有 `ZenohCellPresence` 的 liveness/queryable, 但封装为
  `ServiceProvider` 面 — operator 内部调 presence, implementer 不碰.
- async bridge = janus.Queue + `asyncio.run_coroutine_threadsafe` + reply
  `to_thread`. 参照 `ghoshell_moss_contrib/unitree/g1/channels/listener.py` 的
  janus 样板和 `ghoshell_moss/matrix/networks/zenoh_presence.py` 的 queryable 声明.
- `provide()` 第二次相同 kind 时 raise — 单例保证.
- `service_operator()` 遵循 `network()` 的 lazy-gate 模式: async def, cached singleton,
  `enter_async_context` on first call.
- webview 的 badge 流: 页面调 `navigator.setAppBadge(n)` → `badge_intercept.js`
  (screen-node S2, 已实现) → QWebChannel → pub("badge", payload) → screen-node
  consumer sub 收到 → QML 红点更新. 页面只调标准 API, 不知道 screen 的存在.
- **命名纪律**: 全链路无 `server`/`protocol` 泄漏. `kind` 替代 `protocol`, `service`
  替代 `server`. operator 概念保留在 `ServiceOperator` 类名中 (接线员隐喻).

## V1 Validation (2026-08-09)

**Attempt**: counter_service + counter_caller system test nodes on system_test mode.

**Result**: counter_caller discovered the counter service (liveness + meta query OK)
but `get("counter", "inc", ...)` returned empty — per-key queryable not reached.

**Root cause hypothesis**: cell address format inconsistency between the two paths
operator depends on:

1. **Discovery path** (liveness listener): `address_from_cell_key()` strips
   `{cells_ns}/` prefix → returns address suffix. This address flows into
   `ServiceMeta.address` via the meta queryable response.

2. **Query path** (zenoh get): `ServiceKeyExpr.query_key()` constructs
   `{services_ns}/{normalize(meta['address'])}/{kind}/query/{key}`.

If the cell layer exposes two different address formats — a truncated one in
one path (e.g. `counter_service`) and a full one in another (e.g.
`node/counter_service/01KZ...`) — the query key won't match the terminal's
declared queryable key, and zenoh routes the query into empty space.

**Fix required in cell layer**: `CellAddress` must be consistent across all
zenoh key paths (liveness token, queryable declaration, event publishing).
The operator itself is correct — it uses the same `ServiceKeyExpr` to both
declare queryables and build query keys from discovered meta.

**Per-key queryable fix**: initial implementation used a wildcard queryable
(`{prefix}/**`), which may not be supported across zenoh versions. Changed to
per-key `declare_queryable()` calls in `queryable()`. The wildcard approach
can be revisited once the address consistency issue is resolved and zenoh
wildcard queryable support is verified.

**Files**: `.moss/system_test_nodes/counter_service/` and `counter_caller/`
are ready for re-validation once the cell address issue is fixed.
