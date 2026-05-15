# 03 — 基建依赖准备

日期：2026-05-14

## 背景

Ghost 运行时需要跨进程通讯能力——logos 流的广播、多 UI 端监听等。Session 是三循环的通讯总线，底层基于 zenoh，需要抽象出最小原语来支撑这些场景。

## 讨论要点

### 1. Session 现状

Session ABC 已有两套核心基建：

| 语义 | 方法 | 底层 zenoh key |
|------|------|---------------|
| signal | `add_signal()` / `on_signal()` | `MOSS/{scope}/signals` |
| topic | `topics` property (TopicService) | topic key_expr |
| output | `output()` / `on_output()` | `MOSS/{scope}/outputs` |

Signal 是感知链路的核心基建（输入信号 → Nucleus → Impulse），Topic 是通用消息服务。两者语义明确，继续保留，不冲突。

### 2. 需要新增：通用 key-value pub/sub

当前每加一个新通讯需求就要在 Session 上开一个新的 abstractmethod + 一个新的 zenoh key_expr。这会导致接口持续膨胀。解决方案：加两个最简原语。

```python
# Session ABC 新增
def put(self, key: str, payload: bytes) -> None: ...
def on(self, key: str, callback: Callable[[bytes], None]) -> None: ...
```

### 3. 职责边界

**Session 负责**：屏蔽传输层（zenoh / zmq / ...），暴露 key-based routing。

**Session 不负责**：
- 流式拆包/粘包（first/intermediate/last packet）
- 消息序列化语义
- key 命名约定

这些由上层抽象自行定义。Session = 最简 pub/sub bus。

### 4. 与已有语义的关系

```
Session Layer:
  ┌──────────────────────────────────┐
  │  add_signal() / on_signal()      │  ← 感知链路专用（保留）
  │  topics (TopicService)           │  ← 通用消息服务（保留）
  │  output() / on_output()          │  ← 对外输出（保留）
  │  ┌────────────────────────────┐  │
  │  │  put(key, payload)         │  │  ← 通用 pub（新增）
  │  │  on(key, callback)         │  │  ← 通用 sub（新增）
  │  └────────────────────────────┘  │
  └──────────────────────────────────┘
             ↓ 底层实现
  ┌──────────────────────────────────┐
  │  zenoh / zmq / ...               │
  └──────────────────────────────────┘
```

Signal 和 Topic 不做在 put/on 之上——它们是独立的上层语义，直接对接底层实现。put/on 是平行的通用原语。

### 5. 使用场景

Ghost 运行时通过 `put("stream/logos", chunk)` 广播 logos 流，其他进程的 app（如 TUI）通过 `on("stream/logos", callback)` 监听。key 命名空间是约定，不是 Session 的约束。

## 决策结论

1. Session ABC 新增 `put(key, payload)` + `on(key, callback)` 通用 pub/sub 原语
2. 屏蔽传输层，不暴露 zenoh
3. 不做流式语义——拆包/粘包由上层自行处理
4. 已有 signal / topic / output 语义保留，与 put/on 平行存在
5. Session 接口不再扩张

## 后续影响

- 需修改 `Session` ABC，新增两个 abstractmethod
- `MossSessionWithZenoh` 需实现 put/on，底层用 zenoh key_expr
- Ghost 运行时通过 put 广播 logos，外部 app 通过 on 监听
