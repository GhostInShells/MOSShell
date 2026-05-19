---
title: Cell Discovery Refactor — 从去中心化 liveness 到中心化 queryable
status: draft
priority: P0
created: 2026-05-19
updated: 2026-05-19
depends: []
milestone:
description: >-
  重构 Matrix cell 发现机制：用 zenoh queryable (中心化查询) 替代 N 个 liveness subscriber (去中心化声明)，
  将 is_alive 替换为 reported_at 时间戳，封装发现逻辑为可重写的钩子方法。
---

# Cell Discovery Refactor

## Motivation

当前 Matrix 的 cell 发现机制存在三个问题：

1. **心智模型错误**：运行时发现（cell 存活）和能力发现（manifests）混在一起。cell 发现本质是"谁在这个网络中"，不需要去中心化的 pub/sub 协议——主节点是唯一知道全局状态的一方（它启动了所有子进程）。

2. **N 个 subscriber 的冗余开销**：每个 app cell 一个 zenoh liveness subscriber，加上 liveness token declare、initial wildcard query、per-cell event callback。随着 cell 数量增长线性膨胀。

3. **Cell.is_alive() 是错误承诺**：布尔值 `is_alive` 暴露了"服务发现"逻辑给业务方，但业务方真正需要的是"最新一次被观测到的时间戳"，由自己决定新鲜度阈值。当前 `FractalCell` 直接返回 `True` 就是这种不适配的体现。

## Design Decisions

### D1: is_alive → reported_at

`Cell.is_alive()` 从 abstract method 改为 concrete property，基于 `reported_at: float` (Unix timestamp) 计算：

```python
class Cell(ABC):
    reported_at: float = 0.0  # 0 = never seen

    @property
    def is_alive(self) -> bool:
        """默认判定: 30 秒内有上报。业务方可自行覆盖阈值。"""
        if self.reported_at == 0:
            return False
        return (time.time() - self.reported_at) < 30
```

子类不再 override `is_alive()`。`AppCell`、`HostMainCell` 通过构造时接收 `reported_at`；`FractalCell` 设置为 `time.time()` 保持始终"存活"。

**Why**: 时间戳比布尔值承载更多信息。消费者自行判断 stale 阈值，Matrix 不需要替业务方定义"什么是 alive"。这和 `_ensure_parent_process_exists` 的 poll 模式一致——轮询间隔由消费者决定。

### D2: zenoh queryable 替代 liveness pub/sub

```
旧模型 (N 个 subscriber):
  每个 cell  →  declare liveness token
  main cell  →  subscribe N 个 key
  main cell  →  1 次 wildcard query (初始)
  总计: N 个 token + N 个 subscriber + N 个 event callback

新模型 (1 个 queryable):
  每个 cell  →  session.put("MOSS/{scope}/cells/{address}", info)
  main cell  →  declare_queryable("MOSS/{scope}/cells/query")
  main cell  →  handler: wildcard get → 汇总 → 返回
  总计: N 个 put + 1 个 queryable
```

- No liveness subscriber. No liveness token.
- Cell 启动时 `put`，关闭时 `delete`。
- 崩溃残留：key 可能残留，但 `reported_at` 时间戳让消费者自行判断。
- queryable handler 汇总时，对每个 cell 记录 `reported_at = max(key 中的 ts, 当前时间如果本 cell 是 queryable host 自身)`。

**Why**: Cell 发现是"按需查询"，不是"持续监听"。REPL、TUI、调试工具都是主动查询场景。唯一需要持续监听的场景（app cell 知道 main cell 存活）已经有 `_ensure_parent_process_exists` 覆盖。

### D3: 发现逻辑封装为可重写钩子

6 个 protected 方法从 `MatrixImpl.__aenter__`/`__aexit__` 中抽出：

```python
# --- Cell 发现钩子 (子类可重写) ---

def _build_cells(self) -> dict[str, Cell]:
    """从 AppStore 构建初始 cell 注册表。"""

def _announce_this_cell(self) -> None:
    """向网络广播本 cell。默认: session.put() 到 cell key。"""

def _unannounce_this_cell(self) -> None:
    """从网络移除本 cell。默认: session.delete() cell key。"""

def _start_cell_discovery(self) -> None:
    """启动发现机制。默认: main cell declare_queryable。"""

def _stop_cell_discovery(self) -> None:
    """关闭发现机制。默认: undeclare queryable。"""

def _query_cells_from_network(self) -> dict[str, Any]:
    """从网络查询全量 cell 状态。默认: wildcard get。"""
```

这些钩子在 `__aenter__`/`__aexit__` 的特定位置调用，顺序与当前 liveness 逻辑一致。

**Why**: 不是现在就要加 ABC 中间层。六个方法足够让继承者替换行为（例如用 Redis pub/sub 替代 zenoh queryable），不需要引入 `AbsMatrixImpl`。

### D4: list_cells + alist_cells 双 API

```python
def list_cells(self) -> dict[str, Cell]:
    """同步返回本地缓存的 cell 状态。不阻塞，可能 stale。"""

async def alist_cells(self) -> dict[str, Cell]:
    """异步从网络查询全量 cell 状态。承诺不阻塞 event loop。"""
```

`list_cells()` 返回本地缓存（由 `_query_cells_from_network` 周期性刷新或按需刷新），始终可用。
`alist_cells()` 触发一次网络查询，拿到最新的 `reported_at`。

**Why**: REPL/TUI 等同步上下文用 `list_cells()` 拿缓存，异步上下文用 `alist_cells()` 拿实时。两者语义清晰——不把"可能阻塞"隐藏在同步方法里。

## Blast Radius Summary

| 改动 | 影响范围 |
|---|---|
| `Cell.is_alive()` abstract → concrete | 3 个子类移除 override，1 个 stub，1 个 FractalInspector |
| `Cell.to_dict()` 的 `is_alive` key | 改为 `reported_at`，影响 TUI、MatrixInspector |
| MatrixImpl 内部 liveness 方法删除 | 零外部调用者，纯内部 |
| 新增 `alist_cells()` 到 Matrix ABC | 需在 MatrixImpl 实现，其他 Matrix mock/测试需 stub |
| `list_cells()` 语义不变 | 对外接口兼容 |
| `matrix.is_host_running()` | 内部实现改为检查 `_main_cell.reported_at` freshness |

**不动**: `container`, `session`, `logger`, `create_task()`, `channel_proxy()`, `provide_channel()`, `close()`, `wait_closed()`, `this`, `mode`, `manifests`, `workspace`, `configs` — 全部不变。

## Implementation Notes

- Cell 关闭时的 `session.delete()` 放在 `__aexit__` 的 finally 块早期，在 async_exit_stack unwind 之前。zenoh session 可能已半关闭，delete 失败只 log 不抛异常。
- `_announce_this_cell` 的 `put` 需要在 zenoh session 进入 exit_stack 之后（即 `_session_communication_bus_ctx_manager` 之后），不能更早。
- 崩溃残留 key 不自动清理。queryable handler 返回的 `reported_at` 就是 freshness 判断依据。未来可加 TTL 或 main cell 定期清理，但不属于第一版范围。

## Test Plan (人类审定)

全部通过才算实现完成。

### T1: 单元测试

- `Cell.reported_at` 默认值 0，`is_alive` 返回 False
- `Cell.is_alive` 在 `reported_at` 新鲜时返回 True，超 30s 返回 False
- `Cell.to_dict()` 同时包含 `reported_at` 和 `is_alive`
- `MatrixImpl._build_cells()` 正确从 AppStore 构建注册表
- `MatrixImpl.list_cells()` 返回本地缓存
- `alist_cells()` 触发网络查询并更新缓存
- `_announce_this_cell` / `_unannounce_this_cell` 的 put/delete 调用
- Mock zenoh session 下 queryable handler 正确汇总并返回 cells

### T2: Moss Scripts 回归

- 现有 `scripts/_example/main.py` 启动 → `matrix.session` 可用 → cell 自声明正常 → 关闭无异常
- `matrix.this` 信息正确（address、type、name）
- `matrix.list_cells()` 能看到自己和 main cell

### T3: Moss Apps 回归

- `matrix_exam` app 启动 → `this.is_alive()` 仍可用 → `cell_env()` 正常
- `zenoh_session` app 启动 → `matrix.container.force_fetch(zenoh.Session)` 正常
- `provide_channel_case` app 启动 → `matrix.provide_channel()` 正常
- `proxy_channel_case` app 在 main cell 内 → `matrix.channel_proxy()` 正常
- main cell 关闭 → app cells 通过 `_ensure_parent_process_exists` 感知并退出
- app cell 崩溃 → main cell `alist_cells()` 返回该 cell 的 `reported_at` 定格（不自动清除）

### T4: MCP 运行时自迭代闭环

通过 `moss-as-mcp` 启动 MCP 服务，在 Claude Code 中走完整闭环：

1. 通过 MCP 工具创建新 app（`moss apps init` 等价操作）
2. 通过 MCP 工具启动该 app
3. 通过 MCP 工具调用 `alist_cells()` 确认新 app cell 出现
4. 通过 MCP 工具对 app 的 channel 执行一次命令
5. 通过 MCP 工具停止该 app
6. 确认该 app cell 在 `alist_cells()` 中 `reported_at` 定格，不再刷新

这个闭环验证了：发现 → 启动 → 通讯 → 停止 → 状态一致性，全部基于新的 queryable 机制。

## Migration Path

1. 新增 `reported_at` 到 Cell，`is_alive()` 改为 concrete
2. 新增 `alist_cells()` 到 Matrix ABC 和 MatrixImpl
3. 新增 6 个钩子方法到 MatrixImpl
4. 实现 queryable + put/delete 机制
5. 删除旧 liveness 代码 (`_register_cell_liveness_listener`, `_all_cell_liveness_check_ctx_manager`, `_this_liveness_ctx_managers`, `_check_initial_liveness`, `_cell_alive_events`, `_matrix_cell_liveness_key_*`)
6. 更新 TUI/Inspector 中对 `is_alive` → `reported_at` 的引用
7. 更新 FractalCell 移除 `is_alive()` override
8. T1 → T2 → T3 → T4 逐层通过
