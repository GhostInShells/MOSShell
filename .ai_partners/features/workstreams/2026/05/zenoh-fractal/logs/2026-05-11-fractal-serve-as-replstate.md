# Fractal 从隐式监听迁移到显式 REPLState

**日期**: 2026-05-11
**模型**: DeepSeek V4 (via Claude Code)

## 问题

原来的设计将 `FractalZenohProvider` 注册在 `MatrixImpl._default_providers()` 中，并在 `MossRuntimeImpl.__aenter__` 内自动启动 fractal 生命周期 + hub channel import。这意味着：

1. 每次启动 moss-repl 都会自动打开 zenoh 监听端口
2. 大部分情况下用户并不需要 fractal 组网——这是多余的资源消耗
3. 增加了 Matrix/Runtime 的耦合度，fractal 不是 Matrix 的职责

## 决策

将 fractal 作为**显式 REPLState** 管理：

- **移除** `FractalZenohProvider` 从 `MatrixImpl._default_providers()`
- **移除** fractal 自动启动从 `MossRuntimeImpl` 生命周期
- **新建** `FractalServeState(REPLState)`，在用户主动进入该标签页时才启动 fractal session
- 进入标签页 → `ZenohSessionFractal` 启动 + hub channel import
- 离开标签页 → fractal 关闭 + hub channel 清理
- 保留 hub channel 的 `open_node`/`close_node` 按需管理模式

`MatrixImpl.fractal()` 访问器保留，但无 provider 时返回 None。

## 影响

- moss-repl 默认不再监听 zenoh 端口——需要 fractal 时手动切换到 Fractal Serve 标签页
- Matrix 和 Runtime 回归到纯粹的环境发现 + 本地 app 组网职责
- Fractal 成为 `MossRuntimeTUI` 的可选标签页，和 `MOSSRuntimeREPLState` 并列
- `moss-as-fractal` CLI 不受影响（它自己创建 session，不依赖 Matrix provider）

## 变更文件

| 文件 | 变更 |
|------|------|
| `host/matrix.py` | 移除 `FractalZenohProvider` import + 注册 |
| `host/runtime.py` | 移除 fractal hub import + 生命周期管理 |
| `host/repl/fractal_serve_state.py` | 新建 — REPLState 封装 fractal 生命周期 |
| `host/tui_entries/moss_runtime_ui.py` | `create_states()` 新增 `yield FractalServeState` |

## 架构原则

**显式优于隐式**: 网络监听端口是有副作用的资源占用，不应自动启动。用户应明确进入 fractal serve 状态才开始监听。

**单一职责**: Matrix 负责本地环境通讯总线的构建与管理，不应承担 fractal（跨 workspace 分形组网）的职责。

**关注点分离**: Fractal 是 MossHost 层的能力，不是 Matrix 层的能力。未来应搬迁到 `blueprint.host`（见 out of scope）。
