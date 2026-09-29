# Cell Bootstrap, Not Process Management

**日期**: 2026-06-07
**来源**: 人类工程师与 DeepSeek V4 在 circusd-daemon-management feature 评审中提炼

## 问题

MOSS 通过 App 体系 + circusd 子进程跑通了 AIOS PoC。但接下来的演进方向出现了偏航——开始讨论 circusd daemon 化、bringup 编排、不同节点类型分类。`HostAppStore` 膨胀到 438 行，`circusd-daemon-management` feature 提出 MOSS 管理进程生命周期。

这是错的。MOSS 是 Shell/Bus 层，不是 Kernel/Init 层。进程生命周期治理是 OS 的事——每个 OS 都有自己的 init 系统（systemd、launchd、circusd）。MOSS 不应在 Python 里再实现一个 init。

**真问题不是"如何管理进程"，而是"独立进程如何正确并网"**。基于 Zenoh 组网 + Cell Discovery + Channel 生命周期，任何安装了 `moss[host]` 的 Python 脚本都可以成为一个 cell。需要解决的问题很窄：

1. Session scope 恢复（workspace/mode/ghost/session_id）
2. Cell 地址分配（Zenoh key space 中不冲突的唯一地址）
3. 注册到 Matrix 总线（让其他 cell 可发现）

## 核心决策

### Matrix 是 Bus，不是 Kernel

Matrix 传递信号。不管理进程、不做 bringup、不分类节点。只关心：进程启动后能否接入。

### Workspace 是唯一治理根

文件系统持有 session scope、cell 注册表、环境变量模板。进程启动时从 workspace 读上下文；进程退出后过期记录被下一个启动者清理。不需要独立注册中心。

### 进程编排是一个 Channel

和视觉感知、屏幕截图一样——是能力，不是框架基建。如果系统中运行了 circusd 或 systemd，有权限的 Channel 通过它们的接口做 bringup。"启动其他进程"只是一个 Python 函数签名 Command。

### Bringup = 启动脚本

Mode 的 `bringup_apps` 长期回归为 shell 脚本或外部系统托管。MOSS 不实现 bringup 编排。

### 现有实现不动

App 体系继续 work，保持 PoC 验证能力。`AppStoreChannel` 已是可选。长期被 cell 入网协议替代。

## 方向对比

| | 旧方向 | 新方向 |
|---|---|---|
| Feature | circusd-daemon-management (dropped) | cell-session-bootstrap (draft) |
| MOSS 管什么 | circusd 进程生命周期 | cell 入网协议 |
| HostAppStore | 膨胀为进程管理器 | 收缩为发现+连接 |
| Bringup | mode 内建功能 | 外部启动脚本 |
| 节点模型 | 不同类型分类 | 统一 cell 地址 + 租约 |

## Cell 入网协议（设计方向）

一个独立进程从"裸 Python 脚本"变成"Matrix 上的 cell"：

1. **Workspace 发现** — 环境变量或父进程继承 workspace 路径
2. **Session scope 恢复** — 从 workspace 读 mode/ghost/session_id，注入 env
3. **Cell 地址分配** — 在 workspace cell 注册表中原子分配唯一地址
4. **Matrix 接入** — 用分配的地址启动 Zenoh session
5. **租约维护** — 周期性续约，进程死亡后地址自动回收

### Cell 注册表

`workspace/runtime/cells/` — 文件名 = 序号，内容 = cell 元信息 + 租约。过期租约的序号可被下一个启动者复用。并发安全用 workspace lock（文件锁）——最小可行方案。

### 启动脚本生成

```bash
eval "$(moss cell bootstrap --mode desktop)"
# 设置 MOSS_WORKSPACE, MOSS_MODE, MOSS_CELL_ADDRESS, MOSS_SESSION_ID
python my_script.py
```

## 关键约束

- **不引入 OS 级服务管理**：不集成 systemd/launchd/watchdog
- **不实现进程监控**：不 heartbeat、不自愈重启。死了就死了——下一个进程启动时自然回收地址
- **不改变现有 App 体系**：能 work 就继续 work
- **不引入独立注册中心**：文件系统就是数据库

## 开放问题

1. Cell 注册表的并发安全——文件锁在 NFS/网络文件系统上是否可靠？当前假设本地文件系统。
2. 租约粒度——开发场景 30s 足够，但长期运行的感知 app 需要更长的租约或"永久"模式。
3. 与现有 `MOSS_CELL_ADDRESS` 环境变量的兼容——从"手动设置"迁移到"workspace 分配"的平滑路径。
4. App 体系何时正式替换——等 matrix-channel-hub 稳定、cell 注册表格式验证通过。

## 关联文件

- `.ai_partners/features/workstreams/2026/05/circusd-daemon-management/FEATURE.md` — 被否决的方案，讨论保留
- `.ai_partners/features/workstreams/2026/06/cell-session-bootstrap/FEATURE.md` — 替代 feature
- `src/ghoshell_moss/host/app_store.py` — 当前 HostAppStore 实现
- `src/ghoshell_moss/channels/app_store_channel.py` — 当前 AppStoreChannel
- `.ai_partners/features/workstreams/2026/05/matrix-channel-hub/FEATURE.md` — Matrix Channel Hub 已完成
