---
created: 2026-08-07
depends: []
description: 'Matrix 层能力通讯的通用算子: 封装 zenoh liveness/queryable/pub/sub + 生命周期治理, 暴露
  ProtocolMetaModel + client proxy 体系. 二级抽象 (channel/resource/webview) 各自定义自己的协议与元数据,
  operator 永不装线. 关键词:"统一无聊层, 特别化有趣层".'
milestone: null
priority: P0
status: draft
status_note: 'design converged: two primitives (query+pub/sub), three slots, ProtocolMetaModel
  invariant, V1 disposable validation'
title: Matrix Operator — 有址/无址通讯统一算子, cell 级能力声明的通用骨架
updated: '2026-08-07'
---

# Matrix Operator

> 命名借用 The Matrix 电影的接线员 (Switchboard Operator) — 打电话时, 你是让
> 接线员帮你接线, 不是自己爬进交换机机房接. 这个 operator 就是那个接线员.

## Motivation

端到端弧线: 从 desktop-gui / screen-node / webview 三条线汇聚, serial 讨论了
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
  - 根目录 CLAUDE.md 中 `moss --ai codex blueprint matrix` / `moss --ai codex concepts` — channel/matrix 底层

## Key Decisions

### K1. 两个通讯原语, 三个 slot 契约, 不发明确 send/listen

zenoh `Session.get` 已支持 payload 双向 body (环境验证: 安装版本签名
`get(selector, payload=None, ...)`, `Query.reply(key, payload, ...)`).
**query/queryable 承载所有有址一次性通讯** (请求-应答, 包括 fire-and-forget),
**pub/sub 承载所有流式通讯** (事件/状态流).

删掉的 send/listen: 它在草稿里是"有址单向"——但 query body + 不 await 就能做 fire-and-forget;
流式 inbound 就是 client 往 server key pub, server sub. 所以 send/listen 不是新原语,
是 query 或 pub/sub 的用法模式.

Server 声明的 key namespace 至少含两个 slot:

| slot | key pattern | 方向 | 机制 |
|---|---|---|---|
| `pub/` | `{addr}/P/pub/{foo}` | server → 订阅者流式 | pub/sub |
| `queryable/` | `{addr}/P/queryable/{foo}` | client → server 请求-应答 | query (zenoh get) |

address 在前, protocol 在中, slot 在后. 跨 server 聚合不走通配符, 走 `on_server_start('P')`
→ 逐个订阅该 server 的 pub slot, 复用同一通知协议.

### K2. Operator 拥有异步桥, API 面统一收 async handler

zenoh callback 跑在 zenoh 自有线程 (sync), MOSS 是 asyncio. **operator 内部用
janus.Queue 做桥** (g1 listener 两次验证的样板): zenoh 线程 callback enqueue →
app loop 单点消费 → 跑 async handler → 结果经 to_thread 桥回 zenoh reply.

implementer 永远不碰 zenoh 线程, 写干净的 `async def handler(...)` 即可 "读一个 howto
→ 独立实现 disposable 能力". 没有这条, 每个能力都要自己学 janus.Queue + drain 模式,
howto 长一倍, 不是"最舒服".

### K3. serve 单例, protocol 级地址, 二级抽象自己定义接口

`operator.serve(declaration)` 返回 `ServedCapability` **单例** — 一个 protocol 一个 address,
handler 注册跟 protocol 实例生命周期治理. 不每次重 serve.

"不由 Matrix 自己创建实现, 它的二级抽象自己定义接口" — capabilities 有自己的
protocol 目录 (如 `matrix/protocols/webview/`), 含 ProtocolMetaModel / Provider /
Client. operator 是薄基底.

### K4. ProtocolMetaModel + meta→client 构造不变量

一个 protocol 的元数据:

```python
class ProtocolMetaModel(BaseModel):
    @classmethod
    def protocol(cls) -> str: ...  # 约定, 如 "webview"

class ServerDeclaration(BaseModel):
    address: str                   # global/{cell}/{capability}/{instance}
    protocol: str                  # = meta.protocol()
    data: dict[str, Any]           # ProtocolMetaModel.model_dump()
```

**不容违反的不变量: `declaration.data` 必须能实例化成对应的 client**

```python
# registry[protocol] = (MetaModel, ClientFactory)
# discovery → meta = MetaModel(**declaration.data) → client = factory(operator, meta)
# 失败 = protocol bug, 不该静默
```

这个不变量是"声明即后果 / code-as-prompt"的结构化形态. 发现→使用是闭环:
拿到 declaration 就能构建 client, client 的构造校验 meta 合法性.

### K5. address 格式: cell 锚 + capability 子路径

```
global/{cell_address}/{protocol}/{instance?}
```

cell 锚在前 — liveness 按 cell 挂, 前缀 = address, 发现按 cell 枚举.
与 matrix-resources `scheme://{host}` 同构 (host = cell 身份).

### K6. session 做全局 query, protocol 的枚举由 queryable 承载

一个 server 的 queryable slot 是 `{addr}/P/queryable/{foo}` — 也是单个 server 自己公开的
能力文档. session 级全局 query (`get('**/P/queryable/meta')`) 收拢所有 P 协议的 server meta.

"session 把 zenoh 的协议暴露完" — 已落在 operator: liveness / queryable / pub / sub, 四个裸原语,
外加 on_server_start/stop 生命周期回调. 无再装线.

### K7. channel 不现在回填

channel 的装线特殊 (duplex hub + CTML Shell 内建拦截层 + FutureRouter).
webview/resource 先按 operator 长, pattern 被两三个实例验证后再决定 channel 是否回填.
不要现在改 channel — 会拖慢 webview.

---

## 接口草图 (pseudo-code, 代替已删除的草稿)

```python
"""Matrix Operator — 有址/无址通讯统一算子.

Server (有址, 声明自己存在, client 可以采到)
  - ServerDeclaration = {address, protocol, data}
  - server 声明 key namespace: {addr}/P/pub/{foo}, {addr}/P/queryable/{foo}
  - liveness = server 进程存在性 (zenoh liveliness token), 免费
  - pub slot = emit(foo, payload) → 订阅者收到流式事件
  - queryable slot = query(key, params, payload) → reply

Client (无状态, subscribe + discover)
  - query(key_expr, payload) → replies
  - sub(key_expr, callback) → 接收 pub slot 事件
  - servers(protocol) → 枚举所有 P server
  - on_server_start/stop(protocol, cb) → lifecycle 通知
"""

from typing import Any, Callable, Awaitable
from abc import ABC, abstractmethod
from pydantic import BaseModel

# ---- protocol meta ----

class ProtocolMetaModel(BaseModel):
    """每个 protocol 定义自己的 meta model, 继承此类."""

    @classmethod
    def protocol(cls) -> str:
        """类方法约定: 返回本协议名, 如 "webview"."""
        raise NotImplementedError(f"{cls.__name__} must define protocol()")


class ServerDeclaration(BaseModel):
    """一个 server 的完整声明 (announce payload)."""
    address: str
    protocol: str
    data: dict[str, Any]                   # ProtocolMetaModel.model_dump()

    def meta(self, registry: 'ProtocolRegistry') -> ProtocolMetaModel:
        """从注册表恢复 ProtocolMetaModel, 并校验 protocol 字段一致."""
        ...


# ---- served capability ----

class ServedCapability(ABC):
    """serve() 返回的单例, 一个 protocol 一个 address.

    __aenter__ → announce (liveness + queryable), __aexit__ → revoke.
    operator 内部拥有 async bridge (janus.Queue), handler 直接写 async def.
    """

    @property
    @abstractmethod
    def declaration(self) -> ServerDeclaration:
        """本 capability 的声明 (含 meta.data)."""

    @abstractmethod
    def emit(self, foo: str, payload: bytes) -> None:
        """pub slot: {addr}/P/pub/{foo}. 流式 fan-out 到所有订阅者."""

    @abstractmethod
    def queryable(
        self,
        foo: str,
        handler: Callable[[Query], Awaitable[list[bytes]]],
    ) -> None:
        """query slot: {addr}/P/queryable/{foo}. handler 是 async, 由 operator 内部桥接."""

    @abstractmethod
    async def __aenter__(self) -> 'ServedCapability': ...

    @abstractmethod
    async def __aexit__(self, *args) -> None: ...


# ---- operator ----

class MatrixOperator(ABC):
    """per-cell 的 matrix 接线面. 每个 cell 通过 operator 与 mesh 交互.

    serve — 声明本 cell 提供的能力
    client 侧 — 发现 + 通讯
    """

    @abstractmethod
    def serve(self, declaration: ServerDeclaration) -> ServedCapability:
        """创建 server 单例, 一个 protocol 一个 address."""

    @abstractmethod
    async def servers(self, protocol: str) -> list[ServerDeclaration]:
        """枚举当前在线的所有 {protocol} server."""

    @abstractmethod
    async def protocols(self, address: str) -> list[ServerDeclaration]:
        """一个 address 上声明的所有协议."""

    @abstractmethod
    def on_server_start(
        self, protocol: str, callback: Callable[[ServerDeclaration], None],
    ) -> None:
        """{protocol} server 上线."""

    @abstractmethod
    def on_server_stop(
        self, protocol: str, callback: Callable[[ServerDeclaration], None],
    ) -> None:
        """{protocol} server 下线."""

    def key(self, declaration: ServerDeclaration, slot: str, foo: str) -> str:
        """展开 key: {address}/{protocol}/{slot}/{foo}"""
        return f"{declaration.address}/{declaration.protocol}/{slot}/{foo}"

    # -- 通讯 (角色无关, 吃 key_expr) --

    @abstractmethod
    async def query(
        self, key_expr: str, *, params: dict[str, str] | None = None,
        payload: bytes | None = None,
    ) -> list[bytes]:
        """有址请求-应答. 返回所有 queryable 回复."""

    @abstractmethod
    def sub(
        self, key_expr: str, callback: Callable[[bytes], None],
    ) -> None:
        """订阅 pub slot."""


# ---- protocol registry (phase-2, 现在伪代码) ----

class ProtocolRegistry:
    """协议注册表: 维持 meta→client 构造不变量.

    每个 protocol 注册两项:
      meta_model: type[ProtocolMetaModel]
      client_factory: Callable[[MatrixOperator, ProtocolMetaModel], Client]

    发现流程:
      decls = operator.servers("webview")
      for decl in decls:
          meta, factory = registry.get(decl)
          client = factory(operator, meta)    # 必定成功, 失败 = protocol bug
    """
```

## Validation Plan

两段验证, 第一段是保险丝.

### V1: Disposable Counter/Echo (半天)

模型读本 FEATURE.md + 一个 howto → **独立**写一个运行即丢弃的 counter/echo protocol:
- serve 声明 + queryable handler (counter 增量, echo 原样返回)
- operator client query + sub
- lifecycle (on_server_start/stop 观察到)

判据: 模型是否感觉"最舒服" — 是否只靠本文档 + howto, 不查 zenoh/matrix 内部源码,
独立走通. 如果这一步卡了, operator 形状死了, 停下来改.

### V2: Webview + Screen-Node Red-Dot (半天到一天)

- `webview` protocol: WebViewMeta (id, label, url, icon) + ServedCapability +
  emit(badge_changed, ...)
- screen-node: consumer 侧 — `operator.servers("webview")` → 发现所有 webview →
  meta 进游离层 + `operator.sub(key(server, "pub", "badge"))` → QML 红点
- 端到端判据: screen-node 运行, 一个 webview cell 上线 → screen 游离层自动浮出
  meta item, 页面调 badge → 红点亮起. 零手工 open, 全自动.

---

## Implementation Notes

- operator 的 zenoh 实现 (`src/ghoshell_moss/matrix/operator/`) 复用现有
  `ZenohCellPresence` 的 liveness/queryable, 但封装为 `ServedCapability` 面 —
  operator 内部调 presence, implementer 不碰.
- async bridge = janus.Queue + `asyncio.run_coroutine_threadsafe` + reply
  `to_thread`. 参照 `ghoshell_moss_contrib/unitree/g1/channels/listener.py` 的
  janus 样板和 `ghoshell_moss/matrix/networks/zenoh_presence.py` 的 queryable 声明.
- `CellProtocol` 值域可以扩展 (`Literal['channel', 'resource', 'webview']`),
  但 operator 本身不知道 protocol 语义 — 它只做接线.
- key namespace 冲突: `serve()` 第二次相同 protocol 时 raise, 保证单例.
- webview 的 badge 流: 页面调 `navigator.setAppBadge(n)` → `badge_intercept.js`
  (screen-node S2, 已实现) → QWebChannel → emit("badge", payload) → screen-node
  consumer 收到 → QML 红点更新. 页面只调标准 API, 不知道 screen 的存在.