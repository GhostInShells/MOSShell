---
title: Minecraft Bot App — 从 examples 到 apps 体系迁移
status: completed
priority: P2
created: 2026-06-11
updated: 2026-06-12
depends: []
milestone:
description: >-
  将 examples/minecraft_bot 从自包含 Agent 示例重构为 MOSS App 体系下的
  Channel-based App，支持多实例、ConfigStore 配置、正确的双向 IO 语义。
---

# Minecraft Bot App

## Motivation

`examples/minecraft_bot` 是一个基于 mineflayer (Node.js) + `javascript` (Python-JS bridge) 的 Minecraft 机器人示例。它当前自己组装了完整的 `SimpleAgent` + `CTMLShell` + `QueueChat`，是一个**自包含 Agent 运行时**。

目标：将其迁入 MOSS App 体系（`.moss_ws/apps/minecraft/bot/`），使其成为进程隔离、Host 托管、Ghost 调用的标准 Channel App。同时解决以下问题：
- 消除对 `SimpleAgent`/`ghoshell_moss_contrib` 的依赖
- 配置外置（服务器地址、bot 名称等），支持多实例
- 双向 IO 语义正确（输入 → Ghost，Ghost 回复 → Minecraft 聊天栏）

## Key Decisions

### 1. 架构模式：纯 Channel App，不自带 Agent 运行时

**选择**：App 只提供 `new_channel()`，由 Host 的 `GhostRuntime` 提供 Agent 循环。

**拒绝的替代方案**：保留 `SimpleAgent` 在 App 内。拒绝原因：App 体系的设计哲学是"能力包"，Agent 运行时由 Host 统一提供。自包含 Agent 会绕过 Mindflow、Signal、Topic 等 Host 级能力，也无法被其它 Ghost 复用。

**影响**：
- 删除 `QueueChat`、`CTMLShell`、`SimpleAgent`
- App 入口改为 `async def main(matrix: Matrix)`
- 通过 `matrix.provide_channel(chan)` 注册 channel

### 2. 输出通道：A2 方案 — Ghost 显式调用 `reply` command

**核心约束**：当前 MOSS 架构下，App/channel **无法被动抓取** Ghost 的 articulate 输出（LLM 回复）。Ghost 的输出直接流向 Host 默认输出通道（TUI/REPL），不经过 Topic 或 Signal 回流。

**选择**：Channel 提供 `reply(text: str)` command，Ghost 的 instruction 中强制声明"所有回复必须通过 `<minecraft:reply text="..." />` 发送到游戏内聊天栏"。

**拒绝的替代方案**：
- A1（command 内部自发自报）：会导致 Ghost 收到重复信息（chat 消息作为 input signal 回环 + function result）。
- B（监听 Ghost 输出 Topic）：需要修改 Host/Ghost 层发布 articulate 输出为 Topic，超出本次 App 迁移范畴。

**Prompt 设计要点**（必须在 channel description / context_messages 中体现）：
1. 强制路由声明："你的所有回复必须通过 `minecraft:reply` 发送到游戏内聊天栏"
2. 双向上下文："你收到的 input 来自 Minecraft 聊天，你的 reply 会出现在同一个聊天栏"
3. 区分 return 值与 reply：command return 值只给 Ghost 自己看，人机对话必须通过 `reply`

### 3. 输入通道：Minecraft chat → `Signal(name="input")`

**选择**：JS `@On(bot, "chat")` 回调将聊天事件转为 `Signal(name="input", priority=Priority.NOTICE)`，通过 `matrix.session.add_signal(sig)` 上报。

**注意**：JS 回调是异线程的，需要保留 `asyncio.Queue` 做线程桥接，再由 App 主循环消费后转 Signal。不能直接在回调里调用 `CommandUtil.send_signal()`。

### 4. 配置体系：ConfigType + YamlConfigStore

**选择**：使用 MOSS 的 `ConfigType` + `YamlConfigStore`，配置放在 App 目录 `runtime/configs/minecraft_bot.yaml`。

**配置项**：
```yaml
host: "127.0.0.1"
port: 25565
bot_name: "Jarvis"
```

**读取方式**：
```python
conf = matrix.configs().get(MinecraftBotConfig)
```

**注意**：需要验证 App 进程是否能正确访问 Host 的 `matrix.configs()`。如果隔离级别导致不可用，fallback 为 App 本地 `load_dotenv()` 或读取 `runtime/configs/` 下的 yaml。

### 5. 多 bot 控制：多 App 实例，非单 App 多 channel

**选择**：每个 bot 是一个独立的 App 进程，通过不同的 config 区分。

**拒绝的替代方案**：单 App 进程内管理多个 mineflayer bot，提供 `minecraft/jarvis`、`minecraft/steve` 等 sub-channel。拒绝原因：mineflayer 的 JS 事件循环与 Python asyncio 的桥接已经复杂，多 bot 会引入额外的状态隔离和事件路由复杂度。先保证单 bot 稳定，多实例通过 Host 的 AppStore 管理。

**多实例方式**：
- 不同 config 文件（如 `jarvis.yaml`、`steve.yaml`）
- 或者通过环境变量 `MINECRAFT_BOT_NAME` 区分
- Host 侧启动多个 App 实例

### 6. 状态管理：模块级变量，单实例够用

`to_follow_player` 等状态继续用模块级变量。原因：
- 单 App 实例 = 单 bot
- 不需要 `StatefulChannel` 的复杂状态机
- 如果未来需要多 bot 单进程，再升级为 `StatefulChannel`

## Implementation Notes

### 连接失败显式报错

Mineflayer bot 连接 Minecraft 服务器是异步的，`createBot` 不会立即失败。需要监听 JS 事件并在 App 层面显式处理：

- **监听事件**：`error`（连接错误）、`kicked`（被踢出）、`end`（连接断开）
- **启动时阻塞等待**：`main(matrix)` 中调用 `createBot` 后，应等待 `login` 事件确认连接成功，或等待 `error`/`end` 事件确认失败
- **失败处理**：
  - 抛异常中断 App 启动（`moss apps test` 前台启动时用户立刻看到报错）
  - 记录清晰错误日志：`f"无法连接到 Minecraft 服务器 {host}:{port}: {reason}"`
- **运行时断开**：如果运行中连接断开，通过 `context_messages` 报告 `"状态：与服务器断开连接"`，并通过 Signal 上报故障（`Priority.ERROR`）
- **命令保护**：连接断开期间，所有 command 应返回错误提示（如 `"未连接到服务器，无法执行移动命令"`），而不是静默失败或触发 mineflayer 内部异常

**注意**：JS 事件回调是异线程的，连接状态变更（`login`/`error`/`end`）需要通过 `asyncio.Queue` 或 `asyncio.Event` 同步到 Python asyncio 主循环。

## Implementation Tasks

### T1: App 骨架创建
- [x] `moss apps create games/minecraft_bot`
- [x] 调整目录结构，保留 `server/docker-compose.yml`
- [x] 编写 `APP.md` 元数据
- [x] 编写 `pyproject.toml`，声明依赖：`javascript`, `ghoshell-moss`

### T2: Mineflayer 桥接层重构
- [x] 提取 mineflayer bot 初始化为函数（接受 host/port/bot_name）
- [x] 保留 JS 事件桥接：`@On(bot, "spawn")`、`@On(bot, "chat")`
- [x] 实现线程安全的 Signal 上报队列（JS 回调 → asyncio.Queue → App 主循环 → `matrix.session.add_signal`）
- [x] 删除 `QueueChat`、`chat_task()`

### T3: Channel 定义
- [x] `new_channel(name="minecraft", description="...")`
- [x] 迁移所有 command：`move`, `come`, `where_i_am`, `where_player_is`, `find_blocks`, `dig_under`, `dig_target`, `set_follow_player`, `stop_follow_player`
- [x] 新增 `reply(text: str)` command（A2 输出通道）
- [x] 迁移 `@chan.build.idle` 持续跟随逻辑
- [x] 迁移 `@chan.build.context_messages`（位置、周围方块）
- [x] 在 description / context_messages 中写入强制 reply prompt

### T4: ConfigStore 集成
- [x] 定义 `MinecraftBotConfig(ConfigType)`
- [x] 编写默认 `runtime/configs/minecraft_bot.yaml`
- [x] 在 `main(matrix)` 中读取配置并初始化 bot
- [x] 验证 `matrix.configs()` 在 App 进程中的可用性（**未通过** — App 隔离 venv 中无法访问 Host ConfigStore，fallback 到本地 yaml 已生效）

### T5: 入口与生命周期
- [x] `async def main(matrix: Matrix)` 注册 channel
- [x] `Matrix.discover().run(main)` 标准启动
- [x] 删除旧的 `if __name__ == "__main__": asyncio.run(main())`

### T6: 测试与验证
- [x] `moss apps test games/minecraft_bot` 前台启动验证 — App 成功启动，bot 登录服务器
- [x] 验证 config 读取（bot_name、host/port）— 本地 yaml fallback 生效
- [~] 端到端链路验证（chat → Signal → Ghost → reply → bot.chat）— deferred，等 Ghost 运行时配合验证

### T7: 清理与归档
- [ ] 删除 `examples/minecraft_bot/` 或标记为 deprecated
- [ ] `moss features set-status minecraft-bot-app completed`
- [ ] Commit FEATURE.md 与代码一同提交

## Rejected / Deferred

- **语音输出**：当前示例可选的 `VolcengineTTS` + `PyAudioStreamPlayer` 剥离。TTS 由 Host 统一提供，App 只负责上报 input 和提供 reply command。
- **单 App 多 bot**：见 Key Decision 5， deferred。
- **Ghost articulate 输出自动回流**：需要 Host 层增强，deferred。

## Post-Completion Refinement: Signal Handling & Cleanup

2026-06-12 对 `apps_cli.py` 和 `minecraft_bot/main.py` 的信号处理进行了重构，解决子进程孤儿化问题并简化 App 内部代码。

### 问题背景

`moss apps test` 原来的实现使用 `subprocess.run` + `proc.send_signal(signum)`，存在两个问题：
1. **SIGKILL 截断**：`subprocess.run` 在 `KeyboardInterrupt` 时会强制 `proc.kill()`（SIGKILL），跳过 `main.py` 的 `finally` 清理逻辑
2. **单点转发**：`proc.send_signal` 只发给直接子进程，Node.js bridge（孙进程）收不到信号，导致 orphaning

### 修复：进程组广播

**`src/ghoshell_moss/cli/apps_cli.py`**：
- `subprocess.Popen(..., start_new_session=True)` — 子进程成为新进程组 leader（pgid == pid）
- `os.killpg(proc.pid, signum)` — 收到什么信号就广播到什么进程组，行为与 terminal Ctrl+C 一致
- `except KeyboardInterrupt` 中手动 `os.killpg(proc.pid, signal.SIGINT)` 并 `proc.wait(timeout=10)`，超时再 `SIGKILL`

### App 内部简化

**`.moss_ws/apps/games/minecraft_bot/main.py`**：
- 删除 `atexit.register(_force_kill_node_bridge)` — 冗余，信号 handler + `finally` 已覆盖所有优雅退出路径
- 删除 `finally` 块里的 `os.kill(SIGKILL)` 兜底 — `javascript.terminate()` 发送的 SIGTERM 已足够（测试验证通过）
- 保留 signal handler — asyncio 程序优雅关闭的必要入口，把同步信号转成 `CancelledError`
- 保留 `finally` 块里的 `_bot.end()` 和 `javascript.terminate()` — 核心清理逻辑

### 验证

- SIGTERM 路径：`kill <moss_apps_test_pid>` → 广播到进程组 → main.py `CancelledError` → `finally` → bridge 被清理，无残留
- SIGINT 路径：前台 Ctrl+C → 广播到进程组 → 同上，无残留

### 关键认知

asyncio 程序优雅关闭的三要素缺一不可：
1. **Signal handler** — 把 SIGTERM/SIGINT 转成 `CancelledError`
2. **`finally` 块** — 执行实际清理（`_bot.end()`、`javascript.terminate()`）
3. **父进程进程组广播** — 确保信号到达整个子进程树，而非只到一层

## Related Code

- 当前示例：`examples/minecraft_bot/main.py`
- App 参考：`.moss_ws/apps/genkits/image/main.py`（channel-based App）
- App 参考：`.moss_ws/apps/sensors/listener/main.py`（matrix-based，无 channel）
- Signal/Input：`src/ghoshell_moss/core/blueprint/mindflow.py`（Signal, InputSignal）
- ConfigStore：`src/ghoshell_moss/contracts/configs.py`
