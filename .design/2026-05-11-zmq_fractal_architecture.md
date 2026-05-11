# ZMQ Fractal Architecture

## Design Intent

实现一个 ZMQ 版本的 `Fractal`，用 ROUTER/DEALER socket 做动态节点注册与发现，
替代 alpha `ZMQChannelHub` 的静态声明模式。

## Topology

```
Hub Node (ZMQFractal)
  ├── ROUTER @ ipc:///tmp/moss-zmq-fractal-hub.sock  ← registry
  │
  ├─ Leaf A: DEALER → REGISTER {name, channel_address}
  ├─ Leaf B: DEALER → REGISTER {name, channel_address}
  │
  └── connected() → [FractalCell("A"), FractalCell("B")]
```

## Registry Protocol

JSON over ZMQ multipart (identity + empty + json):

```
REQUEST  (DEALER → ROUTER):
  {"action": "register",   "name": "...", "description": "...", "channel_address": "tcp://..."}
  {"action": "unregister", "name": "..."}
  {"action": "heartbeat",  "name": "..."}

RESPONSE (ROUTER → DEALER):
  {"status": "ok"}
  {"status": "error", "reason": "..."}
```

## Lifecycle

1. Hub 创建 ZMQFractal，`__aenter__` 启动 `_registry_loop`
2. Leaf 创建 ZMQFractal，`provide_channel(parent_addr)` 时：
   - 启动 ZMQChannelProvider 暴露自己的 channel
   - 用 DEALER 连到 parent_addr 发送 REGISTER
3. Hub 的 `_registry_loop` 收到 REGISTER → 更新 `_nodes` 缓存
4. Hub 的 `connected()` 即时返回缓存
5. Leaf 退出时发送 UNREGISTER

## Class Design

```
ZMQFractal(Fractal):
    _name: str
    _registry_address: str          # this node's ROUTER bind address
    _ctx: zmq.asyncio.Context
    _router: zmq.asyncio.Socket     # ROUTER for receiving registrations
    _nodes: dict[str, _NodeInfo]    # name → (cell, channel_address, last_seen)
    _nodes_lock: threading.Lock
    _registry_task: asyncio.Task
    _provided_future: asyncio.Task | None

    __aenter__ → start _registry_loop
    __aexit__  → cancel _registry_loop, close sockets
    connected() → list[Cell] (cached)
    provide_channel(channel, transport, ...) → Future
    channel_hub(name, description) → Channel
    explain() → str
```

## Key Design Decisions

1. **IPC vs TCP**: 默认 IPC，单机场景零网络开销。通过 transport 参数支持 TCP。
2. **No heartbeat in v1**: 首版不做超时检测，节点靠 UNREGISTER 正常退出清理。
3. **Channel proxy 按需创建**: `channel_hub()` 返回的 Channel 中，AI 用 `open_node` 时才创建 ZMQChannelProxy。
4. **与 Matrix 的关系**: ZMQFractal 是 Fractal 的另一种 transport 实现，与 zenoh 平级，由 Provider 选择注入。
