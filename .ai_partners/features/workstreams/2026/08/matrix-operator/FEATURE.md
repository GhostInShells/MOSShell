---
created: 2026-08-07
depends: []
description: 'Matrix 层 cell 级服务化通讯接线层. ServiceOperator 封装 zenoh queryable/pub/sub
  + 生命周期治理, 暴露 ServiceDeclaration/ServiceProvider/ServiceClient 三分体. 二级抽象 (webview/resource/channel)
  各自定义自己的 service kind 与元数据, operator 永不装线. 关键词: "统一无聊层, 特别化有趣层".'
milestone: v0.1.0
priority: P0
status: testing
status_note: '2026-09-01 客户端+服务端桥接层重写完成, 金丝雀单测待 Opus 补齐. 核心修复:
  (1) 服务端: create_task-per-query + 出站 worker, query 不阻塞 loop;
  (2) 客户端: get 全回调化 (零线程), sub 共享管线, emit 出站 worker;
  (3) meta cache 修 K4, liveness 事件 create_task 化;
  (4) sync/async handler 双支持, 并发契约写入 ABC docstring.'
title: Matrix Service — cell 级服务化通讯接线层
updated: '2026-09-01'
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

**system_test/node attempt** (first): counter_service + counter_caller
on system_test mode.  Discovery (liveness + meta) OK, but `get("counter",
"inc")` returned empty.  Root cause: cell address format inconsistency
(short/long name coexistence in cell layer — another session handles this).

**operator-level unit test** (this round): `tests/ghoshell_moss/services/
test_counter.py` — two ZenohOperators on a single zenoh session, no
Matrix harness.  `CounterServer.from_operator(op)` construction seam
exposed for testability.

**Result**: discovery, query (inc stateful + echo params round-trip +
aggregate), pub/sub transport, on_service_start/stop lifecycle — all pass.
Operator is confirmed correct; V1 system-test `get` failure is isolated to
cell/matrix layer.

**Real bugs discovered and fixed during this round:**

| Bug | Location | Symptom |
|---|---|---|
| `session.get` returns blocking iterator, `asyncio.to_thread` only wrapped the call | `zenoh_operator._fetch_meta`, `_query_one` | event-loop blocked on iteration; hang |
| `_undeclare()` was `async def` passed to `asyncio.to_thread` (expects sync) | `zenoh_service_terminal.__aexit__` | liveness token never undeclared; stop callback never fired |
| `query.payload` attribute access on TypedDict | `counter._on_echo` | AttributeError → no reply → caller timeout |
| subscriber callback (`declare_subscriber(key, cb)`) not fired on single-session pub | `zenoh_operator.sub`, `zenoh_service_terminal.listen` | pub/sub samples never delivered |

**Root cause fix — subscriber pattern**: zenoh `declare_subscriber` with
callback is unreliable on single sessions.  Switched to `declare_subscriber(key)`
(no callback) + daemon-thread iterator `for sample in sub:` + `janus.Queue`
bridge — matching the proven `ZenohTopicSubscriber` pattern in
`matrix/topics/zenoh_topics.py`.

**Performance guard**: added `_QUERY_TIMEOUT = 5.0` to bound thread hold
time on `session.get` misses.  `_fetch_meta` log promoted from DEBUG to
WARNING.  Error-reply on handler failure (no more timeout-only failure mode).

### V2 Webview Protocol — design decisions

Two-usage badge split (discussed 2026-08-09):

- **用法 1 状态机** (红绿灰呼吸灯): cross-kind 通用协议，mesh 承载。Provider pub
  `state`，消费方 sub 渲染呼吸灯。不是 webview 专属。
- **用法 2 事件信号** (badge 数字): 面向人类的提醒。渲染侧本地化（page → screen,
  screen-node S2 已实现），不写 mesh。真正需要 mesh 事件的是第三方推送。

**WebViewDeclaration**: url / title / description / icon (可缺省)，不设 `id`
（身份 = address，Sample envelope 自带）。

**分层顺序**: operator 声明式 → screen 迭代 → 页面。协议设计以"对 screen 产生
什么自动效果"为驱动。

**理论最小实现**: 本期不做 webview 服务实现，交给 screen-node 完善。本期交付：
operator 级 counter 单测（证明 operator 正确）+ 上述 bug 修复。

## Kernel Review 2026-08-13 — REOPENED

operator 从 completed 重开为 in-progress。counter 单测全绿不构成内核层质量关的证据——
单测只证明 happy path 传输正确，没压任何故障路径。以下是逐条致命问题（解决顺序即列表顺序）。

### 致命问题清单

| # | 问题 | 位置 | 症状 |
|---|---|---|---|
| 1 | **单消费串行 + hang 即死** | `_consume_queries` 单 consumer 串行 `await handler(q)` | head-of-line blocking：一个慢 handler 卡住该 cell 全部后续 query；一个挂起 handler 永久冻结 consumer，**无 per-task 超时可救** |
| 2 | **队满静默丢弃** | `_on_query` `put_nowait` 失败即丢 | 只 log error，query 被吞。内核层不允许静默丢请求 |
| 3 | **shutdown 无防阻塞** | `__aexit__` 只 cancel consumer task | 挂起的 handler 在停机时不能被掐掉；没有 in-flight 治理 |
| 4 | **无错误隔离** | 整条桥 | 没有 per-query 故障边界；一个 handler 的异常路径可以污染 consumer |
| 5 | **无异常回复 / 异常日志纪律** | while 循环内 | 调用方可能挂到超时；异常不 log 则不可诊断 |
| 6 | **无 sync/async 双支持** | ABC 强制 `Awaitable[bytes]` | 若不做足背压/线程卸载/防阻塞 shutdown，就必须同时支持 sync + async 两条路，否则 bridge 是半吊子 |
| 7 | **handler 锁约定未定义** | — | shared-state handler 的并发正确性归谁？内核不串行后，锁是 handler 自己的责任，必须写明 |

### 实证：zenoh 回调模型（2026-08-13 实测）

zenoh-python 的回调跑在 `pyo3-closure` 线程上（不是主线程，不是 asyncio loop）：

- 同一 closure（同一 subscriber/queryable）串行在一条专属线程；
- 不同 closure → 不同线程 → **跨 closure 并行**；
- 回调内联同步执行，阻塞回调即阻塞该线程；zenoh 不暴露队列，无背压控制。

结论：operator 的 janus 队列是 **pyo3 线程 → asyncio loop 的适配器**，不是模拟 zenoh 的
内部队列。handler 可能被并发调用 → 可重入性归 implementer。FastAPI 参照：`async def`
endpoint = loop 上 create_task，per-request 故障隔离——operator 的 create_task-per-query
设计是主流正确形状。

### 已对齐的修复形状

- `_consume_queries`：只 dequeue + `asyncio.create_task(self._dispatch_query(...))`，
  追踪 in-flight set（剪枝 done）。
- `_dispatch_query` 全容错：无 handler → error reply；handler 异常 → log + error reply；
  reply 失败 → 包裹；CancelledError → 尽力 error reply 后 re-raise（防 "Task exception was
  never retrieved"）。
- shutdown：cancel consumer → cancel 全部 in-flight → `await asyncio.gather(*inflight,
  return_exceptions=True)`。
- 背压保留：janus 有界队列；create_task 数量上界 = 队列 maxsize（1000）。
- **契约变化**：handler 会被并发调用。有 await 点 + 共享状态的 handler 必须自己加锁；
  counter 的 `_on_inc` 无 await 点仍原子（不炸）。

### 金丝雀（修复后必跑）

双 ZenohOperator 单 session（仿 `test_counter.py`）：两个并发慢 query 验证并行完成 +
一个 handler 抛异常不影响另一个；handler 内 `await asyncio.sleep` + 真实 I/O 验证
deferred reply + event loop 不被 zenoh 线程卡住。

## Client Review 2026-09-01

服务端 kernel review 7 条致命问题已对齐修复方向。启动服务端改造时，对客户端
ZenohOperator 做完整 review，发现另一组独立致命问题（与服务端无交集）：

### 客户端致命清单

| # | 问题 | 位置 | 症状 |
|---|---|---|---|
| 1 | **get() 钉线程广播即耗尽** | `get()` 用 `to_thread(list(session.get(...)))` | 每目标钉一条 executor 线程 5s；广播 N 个目标 = N 条阻塞线程；线程池打满后 `get()` 串行等待可用线程 |
| 2 | **sub() thread-per-subscription** | `declare_subscriber` 返回迭代器，daemon 线程 `for sample in sub` | 每个 `sub()` 调用创建一条 daemon 线程，无生命周期治理；N 个订阅 = N 条线程 |
| 3 | **emit() 内联阻塞 loop** | `session.put` 直接在 `emit()` 内调用 | zenoh put 在 congestion control 时可能阻塞，`emit()` 是 async 方法但内部无卸载 |
| 4 | **发现路径 O(N) 串行 meta query** | `get_services_by_kind` 对每个 live service 做一次 `_fetch_meta` | N 个 service = N 次串行 zenoh get round-trip；活跃度高时发现延迟线性增长 |
| 5 | **on_service_stop 违反 K4** | liveness offline 时合成空 meta `data={}` | `ServiceDeclaration.from_meta` round-trip 不变量失效（空 data 通不过 pydantic schema）；stop callback 收到不可用 meta |
| 6 | **_sub_handles 只增不减** | `sub()` 创建 handle 加入列表，close 时不摘除 | handle 泄漏；exit 时遍历已 close 的 handle 重复 undeclare |

### Probe 实证 2026-09-01 (zenoh 1.9.0, 单 session)

四条关键结论（命令行 probe 脚本，见 commit history）：

1. **queryable 回调投递串行且队列阻塞**: 同一 queryable 收到两个并发 query，第二个在
   `t=1.01` 才被投递（第一个回调 sleep 1s）。**同一 closure 的投递严格串行，阻塞回调
   即阻塞该 queryable 的全部后续 query**。FEATURE.md 里 subscriber 的实测结论对
   queryable 同样成立。
2. **跨 queryable 并行**: 两个不同 queryable 各 sleep 1s，总耗时 1.01s——不同
   closure 各自专属线程，互不影响。
3. **get 的 reply 回调互相独立**: 一个 get 的 reply 回调阻塞 1s，另一个 get 的回调
   `t=0.0` 即触发。每次 `session.get` 是独立 closure，**阻塞一个 get 的回调不影响
   其他 get**。
4. **deferred reply 成立（最关键）**: queryable 回调里只把 `query` 对象存起来直接
   返回，0.5s 后从另一条线程 `reply`，caller 正常收到 `b'deferred-ok'`。**回调可以
   做到"只入队、立即返回"，reply 完全异步化**。

结合上一轮 probe（get 的 `Callback(cb, drop)` 形式可用、drop 终结信号可靠；
subscriber callback 在 1.9 单 session 下正常触发——V1 时的"不可靠"结论在当前版本不复现）。

### 架构定型原则（人类架构师对齐）

- 所有 zenoh 回调一律**入队即返回，零阻塞**（纪律，非优化）。queryable 投递串行、
  回调阻塞 = 队头阻塞整个 queryable。
- **get() 全回调化**: `session.get(key, Callback(on_reply, on_drop))`，reply 收集在
  回调线程，`drop` 触发时经 `loop.call_soon_threadsafe` 完成 asyncio Future。彻底
  消灭 to_thread 钉线程，广播 N 个 get 零线程成本，且互不干扰（结论 3）。
- **服务端 reply 异步化**: 回调只 enqueue `(query, key)`，loop 侧
  `create_task`-per-query 跑 handler，reply 从 loop/worker 发出（结论 4 证明合法）。
- **sub 也可以去 daemon 线程**: callback → 共享 janus queue → loop 单点分发。不过
  V1 的旧教训提示要在金丝雀单测里保留单 session pub/sub 用例防回归。
- **全链路没有任何一点需要 zenoh 线程等待 loop 计算结果**: 服务端靠 deferred reply
  解耦（query 对象即回复信道，不需要 Future）；客户端 get 靠 asyncio.Future 单向
  回流（`call_soon_threadsafe(fut.set_result, ...)`，guard cancelled/loop-closed）。
- **验收底线**: query 不阻塞 loop 上其他 task。

### 修复形状 2026-09-01

服务端（`zenoh_service_terminal.py`）：

- `_on_query`（zenoh 回调）：只 enqueue `(query, business_key)`。**队满 → log error +
  尽力内联 error reply**（短同步操作，防 caller 挂到超时），绝不静默。
- `_consume_queries`：只 dequeue + `create_task(self._dispatch_query(...))`，
  in-flight set（done 剪枝）。并发上界 = 队列 maxsize（1000）。
- `_dispatch_query` 全容错：无 handler → error reply；decode 失败 → log + error reply；
  handler 异常 → log + error reply；reply 失败 → log 包裹；`CancelledError` → 尽力
  error reply 后 re-raise。
- **reply/pub 出站单点**: per-terminal 出站 worker（1 条 daemon 线程 + janus queue），
  执行 `query.reply` / lazy `declare_publisher` + `pub.put` 等同步 zenoh 操作，失败
  逐条 log。`pub()` 改为 enqueue。shutdown 时 sentinel + join(timeout)。
- `listen()`：daemon 迭代线程 → **callback subscriber**（probe 已证可用），回调
  enqueue 到 listen queue；`_consume_listen` 同样 `create_task`-per-sample + in-flight。
- handler **sync/async 双支持**: `iscoroutinefunction` → await；sync → `asyncio.to_thread`
  （防 loop 阻塞）。
- `queryable()/listen()/pub()` 在 `__aenter__` 前调用 → 显式 `RuntimeError`（现在是
  zenoh 线程内 AttributeError 静默杀线程）。
- shutdown 顺序：cancel consumers → cancel 全部 in-flight →
  `gather(return_exceptions=True)` → 出站 worker sentinel+join → to_thread undeclare →
  queue shutdown。
- 契约写入 docstring（fatal #7）：handler 会被并发调用，共享状态 + await 点的 handler
  自己加锁。

客户端（`zenoh_operator.py`）：

- **`get()` 全回调化**: per-target 在 loop 侧建 `asyncio.Future`；
  `session.get(key, Callback(on_reply, on_drop), payload=..., timeout=_QUERY_TIMEOUT)`；
  on_reply 在回调线程收集；on_drop → `loop.call_soon_threadsafe` 完成 future（guard
  `fut.cancelled()` / loop closed RuntimeError）。广播 = gather N 个 future，**零线程
  占用**。外层 `wait_for(timeout=_QUERY_TIMEOUT + margin)` 兜底。
- **目标解析去 meta 化**: get/emit 不传 `*services` 时从 liveness cache
  （`parse_live_identity`）直接解析地址，**不再做 N 次串行 meta query**。
  `get_services_by_kind/address` 保留 meta 语义但改为并发 gather + 回调式 `_fetch_meta`。
- **`sub()` 去线程化**: callback subscriber → operator 共享分发 janus queue（带 handler
  标签）→ 单点 consumer → `create_task`-per-sample + in-flight。队满 → log warning
  （流语义容忍丢，不容忍静默）。`sub()` 在未 started 时 → RuntimeError。Handle close
  时从 `_sub_handles` 摘除；聚合 handle 不重复注册。
- **`emit()`**: 目标从 liveness cache 解析；`session.put` 经 operator 出站 worker
  （1 条 daemon 线程 + janus queue）执行，loop 零阻塞。
- **liveness 分发**: `_consume_liveness` `create_task`-per-event；`_fetch_meta` 不再
  队头阻塞管线。**meta cache 修 K4**: online 时缓存 `(dotted_addr, kind) → meta`，
  offline 弹出缓存交给 stop callback；缓存缺失才合成并 log warning。start/stop
  callback 支持 sync + async。
- shutdown：close subs → cancel consumers + in-flight gather → 出站 worker sentinel+join
  → liveness listener exit → queue shutdown；`call_soon_threadsafe` 全部 guard。

ABC (`blueprint/service.py`)：

- handler 类型放宽：`Callable[[Query], Awaitable[bytes] | bytes]`、
  `Callable[[Sample], Awaitable[None] | None]`、start/stop callback 同理。
- docstring 增补：并发契约（handler 可能并发调用、锁归 implementer）、队满策略、stop
  meta 来自缓存的语义。
- 签名结构（sync factory / async method 的分布）不动——counter 等业务 service 零改动。

### 端到端冒烟结果

双 operator 单 session 临时脚本（2026-09-01）：

- 两个并发慢 query（各 sleep 1s）: **并行完成 1.0s**（旧实现 5.0s 串行）
- handler 异常隔离: boom handler 抛 ValueError，caller 收到 error reply 0.0s（不超时）
- sync handler 支持: 同步 `lambda q: q['payload'].upper()` 正常 reply
- pub/sub 单 session: 两次 pub 均正确投递
- emit/listen 单 session: sync listen handler 正常收到
- discovery: `get_services_by_kind` 返回可 `from_meta` 的 meta
- **loop 心跳测量**: 全部测试期间 max gap 0.005s（< 200ms 阈值），**loop 从未被阻塞**

单测清单（委托 Opus，见计划文件 `wobbly-tumbling-lagoon.md`）10 条金丝雀用例待补齐。