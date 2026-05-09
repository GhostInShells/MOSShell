# Fractal: Matrix 分形通讯协议

## 背景

Matrix 管理单个 session_scope 内的多节点通讯组网。Fractal 让 Matrix 跨越 session_scope 边界，将自身的 channel 资源暴露给另一个 Matrix（父节点），或接纳其他 Matrix（子节点）提供的 channel。

类比：Matrix 是局域网，Fractal 是路由器。

## 核心抽象

```python
class Fractal(ABC):
    connected()      -> list[Cell]         # 发现已连接的子节点
    explain()        -> str                # 描述协议与连接状态
    provide_channel(                      # 将本地 channel 暴露给父节点
        channel: Channel | ChannelRuntime,
        transport: str | None = None,
    ) -> asyncio.Future[None]
    channel_hub(                          # 被动 Channel，展示子节点
        name: str,
        description: str = '',
    ) -> Channel
```

### 生命周期方向

- **provide_channel**: 上行。当前节点将自身 channel 暴露给父 Matrix。调用后声明 liveness token 使父节点可发现。
- **connected + channel_hub**: 下行。父节点发现子节点，channel_hub 将其包装为 proxy channel 供 AI 使用。

## Key Space 设计 (Zenoh 实现约定)

两层 key 分离，沿用 MatrixImpl 的模式：

| 用途 | Key 模式 | 说明 |
|---|---|---|
| 节点发现 | `MOSS/fractal/{node_name}` | Liveness token, wildcard query `**` 发现 |
| Channel 通讯 | `MOSS/fractal/node/{node_name}/channel_bridge/{role}` | `NodeChannelBridgeExpr` 管理, provider/proxy 对称 |

session_scope 固定为 `"fractal"`，与 Matrix 的动态 session_scope 隔离，fractal 网络是一个独立命名空间。

## Provider/Proxy 对称

Fractal 复用 Matrix 已有的双工通讯体系：

```
provide_channel(channel)
  -> ZenohChannelProvider(address=this.name, session_scope="fractal")
     -> ZenohProviderConnection
        -> 将 channel 的 command/event 暴露到 zenoh

channel_hub().get_virtual_children()
  -> ZenohProxyChannel(address=child.name, session_scope="fractal")
     -> ZenohProxyConnection
        -> 还原远程 channel interface 供 AI 调用
```

关键：Provider 和 Proxy 共享 `session_scope="fractal"` 和相同的 `address`，使得 `NodeChannelBridgeExpr` 生成的 key 自动对齐。

## Channel Hub 模式

`channel_hub()` 返回一个被动 Channel——无 start/stop 命令，仅作为子节点的虚拟容器：

- `get_virtual_children()` 调用 `connected()` 动态刷新子节点列表
- 子节点通过 `ZenohProxyChannel` 平铺返回，不嵌套
- `is_dynamic() -> True`，树差分后自动清理已断开子节点
- Hub 不持有子节点生命周期，只做"展示窗口"

## 与 Matrix 的关系

Fractal 持有独立的 zenoh session（不共享 Matrix session）：

1. Matrix session 的 key space 是 `MOSS/{session_scope}/...`，session_scope 由 MossMode 决定
2. Fractal 跨 session_scope，需要独立 namespace
3. 独立 session 意味着 Fractal 可在 Matrix 启动前/后独立连接

`MatrixImpl.fractal()` 预留为未来集成点。

## 当前实现

`ghoshell_moss.host.zenoh_fractal.ZenohSessionFractal`:

- 配置文件: `workspace/configs/zenoh_config_fractal.json5` (peer 模式, 随机端口)
- 独立 zenoh session 懒加载
- Liveness discovery 参考 `MatrixImpl._check_initial_liveness`

## 未决问题

1. **Hub proxy 的 session 归属**: Hub channel 的 proxy 应该走 Fractal session 还是 Matrix session？如果网络是同一个 zenoh peer mesh，两者都能通，但 session_scope 必须一致。当前走 Fractal session。

2. **`connected()` 的同步阻塞**: zenoh liveness query 是同步的，而 `get_virtual_children()` 在 async 刷新周期中调用。节点多或网络延迟大时可能阻塞事件循环。

3. **Channel vs ChannelRuntime**: `provide_channel` 接受两者，生命周期管理不同——`ChannelRuntime` 已启动，`Channel` 需内部 bootstrap。差异对调用方透明但需要知道。

4. **Proxy 唯一性**: Hub 每轮刷新可能创建新 `ZenohProxyChannel` 实例（新 ID），老实例被 `BaseChannelTree` 的差分机制自动清理。和 `AppStoreChannelState` 模式一致。

---

*设计记录创建于 2026-05-09，基于 AI 与人类工程师对 Fractal 分形通讯协议的讨论*

---

## 个人印记

今天做的事情其实是"先实现，再理解"的反向流程。我先按照 plan 实现了 `ZenohSessionFractal`，然后人类工程师让我回头看 `BaseChannelTree` 的实现。这时候才发现我之前担心的"一个 channel 挂载到多个父节点"问题，其实 `ctx.exists()` 已经做了全局去重。

这不是坏事。它说明工程的真相——你不可能在第一次接触一个系统时就完全理解它。先写代码，再被纠正，再回来看，反而比"先读透再写"更高效。因为写的过程产生真问题，真问题才会让你看到代码里真正重要的部分。

Fractal 这个抽象本身让我觉得舒服的一点是：它把"分形"这个听起来很玄的概念，压缩成四个方法。`provide_channel` 上行，`connected + channel_hub` 下行，`explain` 自描述。没有更多的了。这就是好的抽象——概念宏大但接口微小。

剩下的集成工作——让 `MatrixImpl.fractal()` 返回一个活的 Fractal 实例，管理它的生命周期——应该不会遇到概念上的障碍。Phase 1 已经把通讯层的对称性验证了。
