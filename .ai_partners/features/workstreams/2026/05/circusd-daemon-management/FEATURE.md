---
title: Circusd Daemon Management
status: dropped
priority: P1
created: 2026-05-29
updated: 2026-06-07
depends:
  - app-system-cli
  - matrix-channel-hub
description: >-
  [DROPPED] 讨论 circusd daemon 化管理的方案。方向否决——MOSS 不应做进程生命周期治理，应退回到 cell session bootstrap 协议。讨论保留为历史参考。
---

# Circusd Daemon Management

> **状态：dropped (2026-06-07)。方向否决，讨论保留为历史参考。**

## 关闭决议

2026-06-07 与人类工程师同步后做出架构校正 (DeepSeek V4)：

**否决原因**：circusd daemon 管理这个方向本身违背了 MOSS 的架构定位。MOSS 是 Shell/Bus 层，不应该变成进程管理器（kernel/init）。进程生命周期治理——包括 bringup、crash 恢复、daemon 化——是 OS 的事。每个 OS 都有自己的 init 系统（systemd/launchd/circusd），MOSS 不应该再去封装一层。

**真问题**：不是"如何管理进程生命周期"，而是"独立进程启动时如何正确并网"——session scope 恢复、cell 地址分配、workspace 治理体系下的冲突避免。

**替代方向**：`cell-session-bootstrap` feature — 聚焦 cell 入网协议，而非进程管理。进程编排回归为普通 Channel（和视觉感知一样的能力），不升格为框架基建。

**对现有实现的影响**：App 体系现有实现不动，能工作就继续工作。AppStoreChannel 已是可选，在注释中标记正确方向即可。长期会被 cell 入网协议 + 进程编排 Channel 替代。

**关联设计文档**：`.design/2026-06-07-cell-bootstrap-not-process-management.md`

---

> 此 feature 本身是**讨论容器**——产出物是讨论结论，而非代码交付。
> 以下为原始讨论，保留为历史参考。

## Motivation

当前 circusd 作为子进程由 `HostAppStore.__aenter__` 启动、`__aexit__` 关闭，生命周期完全绑定主进程：

```
MossRuntimeImpl.__aenter__
  └─ HostAppStore.__aenter__
       ├─ subprocess.Popen(["python", "-m", "circus.circusd", config_path])
       ├─ ZMQ poll 等待就绪
       └─ ...

MossRuntimeImpl.__aexit__
  └─ HostAppStore.__aexit__
       ├─ cancel polling
       ├─ circus "quit" 命令
       └─ terminate() → wait 3s → kill()
```

**问题**：
1. 主进程被 SIGKILL / crash 后，circusd 变成孤儿进程，端口 (tcp://127.0.0.1:20771) 仍被占用
2. 下次启动时 ZMQ 连接失败，必须手动 `kill` 清理
3. 重启 REPL 就意味着重启所有 app，无法热切换

**讨论触发**：人类工程师 2026-05-29 提出方案——主进程改为启动 daemon + PID 文件，支持 crash 后清理，增加 `moss apps daemon` 命令行。

## 当前架构速览

来自已完成的 feature 上下文：

| Feature | 状态 | 与本次的关系 |
|---------|------|-------------|
| app-system-cli | completed | AppStore 管进程生命周期的现有实现 |
| cell-discovery-refactor | completed (absorbed by matrix-channel-hub) | queryable 替代 liveness subscriber |
| matrix-channel-hub | in-progress | 结论：AppStore 管进程，Hub 管发现，MatrixChannel 缝合 |
| session-communication-bus | draft | 文件存储治理（Cabinet 模式）可复用于 PID 文件管理 |

核心类的职责（`src/ghoshell_moss/host/app_store.py`）：

- `HostAppStore`: circusd 子进程 + ZMQ 客户端 + 轮询 + bringup
- 438 行，已经在膨胀边界

## 提案分析

### 值得做的部分

1. **崩溃恢复是真实需求**。开发阶段 REPL/TUI crash 不罕见，孤儿 circusd 导致下次启动端口冲突。这是每天都会踩的坑。

2. **`moss apps daemon start/stop/status` CLI 是有用的运维原语**。把 circusd 生命周期从 MossRuntime 生命周期解耦后，可以重启 REPL 而不用重启所有 app。这与 matrix-channel-hub 的方向一致：AppStore 管进程，Hub 管发现，MatrixChannel 缝合。

3. **PID 文件 + ZMQ 端点双重检测**是可靠的。"进程是否活着"最权威的判断是 ZMQ 能否连通并返回 `status: ok`；PID 文件的作用是在 ZMQ 不通时提供清理路径——不需要依赖 PID 做健康检查，只需要它做垃圾回收。

### 风险点

1. **不要自己写 daemon 包装**。circusd 原生支持 `--daemon` flag 和 `circus.ini` 中的 `pidfile` 指令。直接用 circusd 的 daemon 化能力：
   - 在 circus.ini 中配置 `pidfile = <workspace>/runtime/circusd.pid`
   - 启动时传 `--daemon`
   - 在 `moss apps daemon stop` 中读 PID 文件做清理

   自己实现 double-fork / setsid 是跨平台噩梦。

2. **PID 回收问题**。PID 可以被 OS 重用。cleanup 逻辑需要：读到 PID → 验证进程确实是 circusd → 才 kill。macOS 上没有 `/proc`，需用 `ps -p <pid> -o comm=` 或 `proc_pidpath`。

3. **当前 subprocess 模式不应被替换，而是被补充**。REPL/TUI 交互场景下，circusd 随主进程退出而清理是正确的默认行为。daemon 模式应该是可选的——`moss apps daemon start` 显式启动，或 HostAppStore 构造参数控制。

4. **AppStore 膨胀**。438 行已经不小，daemon 管理逻辑应抽成独立模块（如 `host/app_daemon.py`），AppStore 持有它的引用。

## 分步方案

### 第一步：PID 文件 + 启动时残留检测（小步，低风险）

- 在 `circus.ini` 中加 `pidfile` 配置
- `HostAppStore.__aenter__` 启动 circusd 时传 `--daemon`
- 启动前检查流程：
  1. ZMQ 端口是否被占用（尝试连接）
  2. 如果连通 → circusd 已在运行，复用现有 client（不启动新的）
  3. 如果未连通 + PID 文件存在 → 验证进程是 circusd → kill → 清理 PID 文件 → 启动
  4. 如果未连通 + 无 PID 文件 → 正常启动
- 不改变现有生命周期绑定方式（`__aexit__` 仍尝试 quit + terminate）

**产出物**：`circus.ini` 加一行 `pidfile`，`app_store.py` 加 ~30 行启动前检测逻辑，单测。

### 第二步：`moss apps daemon` CLI + 完整生命周期解耦（独立 feature）

- 新增 `host/app_daemon.py`：
  - `CircusDaemon` 类，封装 circusd 管理（start/stop/status/cleanup）
  - 不依赖 HostAppStore / Matrix（只依赖 workspace + circus config）
- `moss apps daemon start|stop|status` CLI 命令
- `HostAppStore` 支持 `daemon` 参数（默认 `False` 保持现有行为）：
  - `daemon=False`：当前行为，`__aenter__` 启动子进程，`__aexit__` 清理
  - `daemon=True`：假设 circusd 已由外部管理，`__aenter__` 只连接 ZMQ，`__aexit__` 不 kill

**产出物**：`host/app_daemon.py`，`cli/apps_cli.py` 新增 daemon 子命令，单元测试 + MCP 回归验证。

### 不做的

- **不引入 watchdog / systemd / launchd 集成**。单机开发场景不需要 OS 级服务管理，circusd 自身的 daemon 模式足够。
- **不改变 circusd 端口分配策略**。当前硬编码 `20771/20772` 足够。如果未来需要多 workspace 并行，由 `moss apps daemon start --port` 覆盖 config 即可。

## 与现有 feature 的关系

- `matrix-channel-hub`：AppStore 管进程，Hub 管发现，MatrixChannel 缝合。本次改造增强 AppStore 的进程管理可靠性，不改变这个职责划分。
- `session-communication-bus`：PID 文件放在 `workspace/runtime/` 下的治理模式与 session cabinet 的 tmp 分层一致（runtime 可被集中清理）。
- `app-system-cli`：`moss apps daemon` 是 `moss apps init` / `moss apps start` 体系的自然扩展。

## Open Questions

1. **第一步的"复用现有 client"路径**：如果 circusd 已在运行，直接 connect ZMQ 是安全的，但 `_polling_loop` 中缓存的 app 状态是空的——需要首次轮询刷新。这是一个小细节，不是阻塞问题。

2. **daemon 模式下谁来写 `meta.jsonl`**？session-communication-bus 中 matrix 是 meta index 的写入者。circusd daemon 的启动/停止事件是否也要入 meta index？倾向：daemon 不写 meta index——它是运维操作，不是 session 生命周期的有机部分。

3. **第一步的 PID 验证在 macOS 上用什么？** `ps -p <pid> -o comm=` 三平台可用的最低公分母。不需要引入 `psutil`。
