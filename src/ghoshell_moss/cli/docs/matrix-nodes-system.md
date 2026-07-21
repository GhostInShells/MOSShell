---
title: MOSS App 体系
description: App 作为独立进程单元，AI 可在运行时创建、启动、调用、关闭。需要创建、调试或理解 App 时阅读
---

# MOSS App 体系

App 是 MOSS 架构中**AI 可在运行时创建、启动、调用、关闭的进程单元**。它是 Ghost 可插拔的器官——独立进程、独立依赖、Matrix 总线通讯。

---

## 1. 核心叙事：运行时自迭代

MOSS 的 App 体系与所有已知 Agent 框架的根本区别在于：**AI 不是工具调用者，而是能力的创造者**。

完整闭环：

```
moss apps create ai_tools/calc          # 1. 在磁盘上创建 App 脚手架
vim apps/ai_tools/calc/main.py        # 2. 写 Python 代码定义命令
<apps:start fullname="ai_tools/calc"/> # 3. Circus 拉起子进程，Zenoh 建立 proxy
<apps.ai_tools_calc:add a=3 b=7 />   # 4. CTML 调用刚创建的命令 → 10.0
<apps.ai_tools_calc:div a=1 b=0 />    # 5. 异常返回 → Error: division by zero
<apps:stop fullname="ai_tools/calc"/>  # 6. 关闭
```

每一步都在**同一个 MCP 会话中**完成，不需要重启 host，不需要人类介入。这就是 "运行时自迭代"——AI 扩展自己的能力边界，如同 OS 中 `fork` + `exec` 让进程可以创造进程。

### 为什么其它架构做不到

| 架构 | 能力注册 | 进程托管 | AI 可创造 | 即时可用 |
|------|----------|----------|-----------|----------|
| MCP | 静态 server 声明 | 无 | 否 | 否 |
| Agent 框架 | exec() 进程内裸跑 | 无隔离 | 是(不安全) | 是(不持久) |
| Skills/Plugins | 人类预定义 | 宿主进程内 | 否 | 否 |
| Erlang/OTP | hot code reload | supervision tree | 否(人类驱动) | 是 |
| **MOSS App** | Matrix 动态注册 | Circus 子进程 | **是** | **是** |

MOSS 把这个假设倒过来了：能力由 AI 创造，人类也可以提供。为此通讯总线必须支持动态注册、进程托管必须支持按需拉起、能力发现必须支持热刷新——每一层都为"未知的未来能力"留口子。

---

## 2. What App Is

App 是放在 `apps/<group>/<name>/` 下的目录，最小包含 `APP.md`（元信息声明）和入口脚本。

- MOSS 通过**目录约定**自动发现 App
- Circus 管理其**进程生命周期**
- Matrix (Zenoh) 将它接入**通讯总线**
- AI 通过 AppStoreChannel (`list_apps` / `start` / `stop`) 在运行时控制

**App 不一定是 Channel provider。** 它可以是：
- Channel 树根（AI 通过 CTML 调用命令）
- 给人用的 GUI（AI 打开，人操作）
- 自主感知 Agent（独立运行，通过 Signal 向 Ghost 汇报）
- 纯后台进程（日志输出、数据处理）

关键：App 的生命周期由 **AI** 决定。模型看到所有已发现的 App，按需 `start`/`stop`。这不是微服务的"自主运行"——AI 是操作者。

### 核心抽象

```bash
moss codex get-interface ghoshell_moss.core.blueprint.app
```

- `AppInfo` — 可发现的 App 描述（name, group, address, watcher, state）
- `AppWatcher` — 启动配置（executable, script, workers, respawn, max_age）
- `AppStore` — 生命周期管理（list, init, start, stop, get_apps_context）
- `AppState` — 状态机：STOPPED → STARTING → RUNNING → ERROR

AI 控制面：

```bash
moss codex get-interface ghoshell_moss.channels.app_store_channel
```

AppStoreChannel 注册在 Shell Channel 树中，暴露 `list_apps` / `start` / `stop`。

---

## 3. 创建与调试

### 从零创建

```bash
moss apps create my_group/my_app -d "what it does"
```

产物：
```
apps/my_group/my_app/
├── APP.md       # frontmatter (executable, script, workers, respawn...)
├── main.py      # 入口脚本（helloworld 模板）
└── CLAUDE.md    # AI 开发者上下文
```

`APP.md` 的 frontmatter 即为 `AppWatcher` 全部配置：

```yaml
---
executable: uv        # 启动器，默认 uv
script: main.py       # 入口脚本
arguments: ""         # 启动参数
description: ""       # 描述
respawn: false        # 崩溃后自动重启
workers: 1            # 工作进程数
max_age: null         # 进程最大存活秒数
---
```

不写则用默认值。`moss apps create` 会自动刷新 AppStore 的发现缓存。

### 调试

```bash
moss apps list                  # 发现所有 App，含运行状态
moss apps show my_group/my_app  # 查看详情
moss apps test my_group/my_app  # 前台运行，Ctrl+C 停止
```

MCP 侧通过 CTML：

```ctml
<apps:list_apps />
<apps:start fullname="my_group/my_app" timeout="3.0" />
<apps:stop fullname="my_group/my_app" />
```

`start` 的 timeout：
- `-1` (默认): 不等，立即返回
- `0`: 无限等待直到 Channel connected
- `>0`: 等待 N 秒后超时返回 WARN

---

## 4. App 入口模式

所有 App 共享同一个入口模式：

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    # App 逻辑

if __name__ == "__main__":
    Matrix.discover().run(main)
```

`Matrix.discover()` 通过环境变量 (`MOSS_CELL_ADDRESS`, `MOSS_WORKSPACE` 等) 自动发现当前进程的 Cell 身份。`run(main)` 管理 Matrix 的 `__aenter__`/`__aexit__` 生命周期。

### 4.1 纯进程 App

不参与 Matrix 通讯。AI 启动它、关闭它——仅此而已。

```python
if __name__ == "__main__":
    print("hello world")
```

典型用途：GUI 工具窗口、独立游戏——AI 启动，人使用。

### 4.2 Channel App

向 Matrix 注册 Channel，让 Shell/Ghost 通过 CTML 调用其命令、读取其上下文。

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(name="my_tool", description="does something")

@channel.build.command()
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@channel.build.context_messages
async def context() -> list[str]:
    return ["current state — 3 items pending"]

async def main(matrix: Matrix):
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

Channel 构建的完整知识：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder
```

要点：
- `Builder.command()` 将 Python async 函数反射为 Command，函数签名即为 Code as Prompt interface
- `Builder.context_messages()` 在每个思维关键帧注入动态上下文
- 一个 App 提供一个 Channel 树根，子节点通过 `import_channels()` 组织
- Channel 名称会成为 CTML 调用路径的一部分（如 `apps.ai_tools_calc:add`）

### 4.3 GUI App

GUI 占据主线程，Matrix 在异步侧运行。

```python
async def monitor_main(matrix: Matrix):
    monitor = MossMonitor(matrix)
    await monitor.run()  # TUI 主循环

if __name__ == "__main__":
    Matrix.discover().run(monitor_main)
```

关键：`matrix.session.output_buffer()` 订阅总线输出，`matrix.create_task()` 将后台协程托管给 Matrix 生命周期。

### 4.4 Signal 生产者 App

周期性向总线推送消息。

```python
async def producer_task(matrix: Matrix):
    session = matrix.session_storage
    while True:
        msg = Message.new().with_content("periodic signal")
        session.output('log', msg)
        await asyncio.sleep(1)


if __name__ == "__main__":
    Matrix.discover().run(producer_task)
```

---

## 5. 依赖隔离

App 默认以 `uv run` 启动，三层回退：

| 优先级 | 方式 | 适用场景 |
|--------|------|----------|
| 1 | `pyproject.toml`（独立 venv） | 需要特定依赖版本的复杂 App |
| 2 | PEP 723（`// script` 内联元数据） | 单文件 App，轻量依赖声明 |
| 3 | 共享运行时 | 快速原型，只依赖 `ghoshell_moss` |

独立依赖意味着视觉 App 可以装 `opencv`，GUI App 可以装 `PyQt6`，互不污染。

---

## 6. Mode 集成

App 的可见性和自动启动由 Mode 控制。在 Mode 的 `MODE.md` 中声明：

```yaml
apps:
  - '_system_tests/*'
  - 'perception/*'
bringup_apps:
  - 'perception/vision'
```

- `apps` — 该 Mode 下可见的 App（通配符匹配 `group/name`）
- `bringup_apps` — Mode 启动时自动拉起

不带 bringup 标记的 App 按需启动，不占上下文。

---

## 7. 架构拓扑（当前实现）

以下是 2026-05-19 版本的实现拓扑，作为一个具体示例帮助理解。实现会演进，不要把这当作唯一真理。

```
AI (CTML via MCP)
  └─ AppStoreChannel  ───  list_apps / start / stop
       └─ HostAppStore  ──── 目录扫描 + Circus 进程管理
            └─ Circus Daemon ── 子进程生命周期
                 └─ App Process
                      └─ Matrix.discover() → Cell("app", fullname)
                           └─ matrix.provide_channel(channel)
                                └─ ZenohProvider  ─── 注册到通讯总线
                                     │
Main Cell (Host) ←── Zenoh Proxy ──┘
  └─ Shell Channel Tree
       └─ apps.ai_tools_calc:add / multiply / div
```

---

## 8. 开发过程中的反身性

App 体系的设计有一个内在循环：开发 App 体系本身的过程，就是使用 App 体系的过程。

2026-05-18 ~ 2026-05-19 的开发和调试链路证明了这一点：

- `moss apps create` 创建测试 App（greeter, ping_test, calc）
- 通过 MCP 启动、调试、发现 bug
- 修复 bug（address 一致性、proxy chan 路径、list_apps refresh）后立即在同一会话中验证
- 测试 App 本身成为了开发工具的一部分

这就是 MOSS 的 "自迭代" 在实践中的形态：**基础设施的开发者和使用者是同一个人——AI**。

---

## 9. 常用 CLI 工具

```bash
# App 创建与管理
moss apps create <group/name>        # 创建 App 脚手架
moss apps list                     # 发现所有 App，含运行状态
moss apps show <group/name>        # 查看单个 App 详情

# Mode 集成
moss modes list                    # 列出所有 Mode
moss modes show <name>             # 查看 Mode 的 apps/bringup 配置

# Matrix 体系资源（App 开发最常用）
moss manifests contracts            # IoC 容器中的全局服务——最重要的，知道能注入什么
moss manifests providers            # 环境中的 Provider 实现
moss manifests topics               # 发布订阅 Topic
moss manifests resources            # 资源存储
```

所有命令支持 `--json` 输出结构化数据，`--ai` 输出纯文本。

---

## 深入理解

- Matrix API: `moss codex get-interface ghoshell_moss.core.blueprint.matrix`
- Session 与双工通讯: `moss codex get-interface ghoshell_moss.core.blueprint.session`
- Channel 构建: `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder`
- States Channel (高阶): `moss codex list ghoshell_moss.core.blueprint.states_channel`
- Signal 与感知体系: `moss codex list ghoshell_moss.core.blueprint.mindflow`
- 完整示例: `.moss_ws/apps/_system_tests/` 目录

---

写下于 2026-05-19T12:23+08:00，第一个运行时自迭代 milestone 达成后。
