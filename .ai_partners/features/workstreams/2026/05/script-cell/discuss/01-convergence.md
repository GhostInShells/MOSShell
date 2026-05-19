# Script Cell — 设计收敛讨论

## 上下文

2026-05-19，人类工程师提出开发 script cell——一种 dev-time 一次性脚本，用于在 ghost 运行时发送 signal、查询状态、调试。讨论从理解 features 机制开始，逐步收敛到具体设计。

## 共享词汇

- **Script Cell**：`moss script run <name>` 启动的一次性脚本。复用 moss 运行时依赖，通过 Zenoh 连接运行中的 matrix 网络，完成任务后退出。
- **Connector 模式**：非 host cell 的统一 Zenoh 配置——connect 到主节点而非 listen。

## 锚点

> "它更面向开发时而不是运行时。如果要和一个运行时 ghost 通讯，走 signal，调试时除了 tui/gui 外，最简单的就是调用脚本发送一些讯号。"

> "第一版不做发现做约束，就是考虑一件事，发现是被动的，好像未来会有安全隐患，但又不想过早做防御。"

> "default providers 可能只有 main 和非 main，配置文件也要想想怎么搞，一次解决这个坑。或者两个文件合并都比现在好。"

## 核心决策

### 1. 第一版轻量级 signal injection，不做 liveness 动态发现

Zenoh signal 路径已完全打通：`MOSS/{session_scope}/signals` → Session → Mindflow。Script 只需一个 Zenoh session 和正确的 key expression 就能注入 signal，不需要主节点 "感知" 它。

动态 liveness 发现暂缓：基础设施已存在（`_register_cell_liveness_listener` 等），但目前只监控预注册 cell。扩展到 wildcard 在技术上简单，但引入了被动发现面——未来的安全隐患。等威胁模型更清晰时再做。

### 2. `moss script` CLI，统一生命周期管理

参照 `moss apps test` 的模式：Host 发现环境 → 构建 MOSS env vars → subprocess 前台运行。保持所有生命周期管理在 moss CLI 下。

### 3. SCRIPT.md 清单，极简

`[ws]/scripts/<name>/SCRIPT.md`，比 APP.md 简单得多——没有 `executable`、`workers`、`respawn`。脚本严格复用 moss Python 运行时，不做 pyproject.toml 隔离。

### 4. Zenoh 配置合并：main vs connector

当前 `zenoh_config_main.json5`（listen）和 `zenoh_config_app.json5`（connect）的本质区别不是 "main vs app"，而是 "listener vs connector"。重命名 `zenoh_config_app.json5` → `zenoh_config_cell.json5`，所有非 host cell 统一使用。

`MatrixImpl._default_providers()` 从 if-elif-else 链改为 host vs 非 host 二分。`HostEnvZenohProvider` 同理。

### 5. CellType.script 已存在于设计

`CellType.script = "script"` 在 matrix.py 枚举中已定义，本 workstream 是将其落地实现。

## 影

- 最初考虑是否需要完整 Matrix 集成（ChannelProxy、TopicService），讨论后收敛为第一版只做 signal injection。轻量优先。
- 对 "是否需要主节点感知" 的纠结，最终以安全考虑（被动发现面）为理由，决定暂不做动态发现。
