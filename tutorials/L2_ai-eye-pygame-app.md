# L2. AI Eye — 从零到 AI 实时控制的 Pygame 图形应用

> Written by deepseek-v4-pro, 2026-06-07

**45 分钟，创建一个 AI 可通过 CTML 实时控制的 Pygame 眼球应用。核心学习点：GUI 线程分离、Channel 状态共享、MOSS App 注册与调试。**

## 你需要知道什么

- `moss ctml read` — CTML 调用语法
- `moss codex blueprint channel_builder` — Channel 构建 API
- `moss codex blueprint matrix` — Matrix 进程发现与通讯
- `moss howtos list` 中找 GUI App 开发文档（线程分离模式）
- 开发时如需 MCP 调试：`moss howtos list` 中找 MCP 相关文档

## 你要做什么

在 MOSS 中创建一个 `games/ai_eye` App——独立的 Pygame 窗口，渲染一只会动的眼睛。AI 通过 CTML 命令实时控制眼球注视方向、瞳孔大小、眨眼和表情。

完成后，AI 可以在会话中直接输出 `<apps.games_ai_eye:look_at x="0.8" y="0.2"/>`，眼睛就会看向右上角。

## 你需要什么

- MOSS 已安装 (`.venv/bin/moss` 可用)
- MOSS 运行时在跑（REPL 或 MCP——MCP 是可选的开发时调试方案）
- macOS 用户注意：本 tutorial 包含 macOS 特化处理

## 第一步：创建 App 骨架

```bash
mkdir -p .moss_ws/apps/games/ai_eye
```

创建三个文件：

**`APP.md`**（元信息声明——App 发现的入口）：

```yaml
---
arguments: ''
description: 'AI Eye — controllable pygame eye avatar with gaze, dilation, blink, and expressions'
executable: uv
respawn: false
script: main.py
workers: 1
---

AI Eye — a minimal pygame window rendering an expressive eye that AI controls via Channel commands.
```

**`pyproject.toml`**（依赖隔离）：

```toml
[project]
name = "ai-eye"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "ghoshell-moss[host]",
    "pygame>=2.5.0",
]

[tool.uv.sources]
ghoshell-moss = { path = "../../../..", editable = true }
```

## 第二步：写代码

**核心约束：macOS 要求 GUI 窗口必须在主线程创建。** 因此 pygame 跑在主线程，Matrix 的 asyncio 循环跑在后台线程。两者通过 `threading.Lock` 保护的共享状态通讯。

完整代码见配套文件 `L2_ai-eye-pygame-app.py`。关键设计决策：

| 决策 | 原因 |
|------|------|
| Pygame 跑在主线程 | macOS `NSWindow` 限制 |
| Matrix asyncio 跑在后台线程 | 与 pygame 不冲突 |
| `State` 类用 `threading.Lock` | 两个线程安全的唯一桥梁 |
| `time.sleep(2)` 等待 Matrix 启动 | 确保 channel 已注册再进入渲染循环 |
| `@channel.build.command(always_observe=False)` | 眼球命令不需要打断其他任务 |

## 第三步：刷新发现并启动

先确认 App 已被发现（新创建的 App 运行中可能还没缓存）：

```ctml
<apps:list_apps/>
```

确认 `games/ai_eye` 出现在列表中后启动：

```ctml
<apps:start fullname="games/ai_eye" timeout="15.0"/>
```

启动后通过 `get_moss_dynamic_info` 确认 `apps.games_ai_eye` Channel 出现，且状态为 `[RUNNING]`。

如果你用 CLI 前台调试：

```bash
.venv/bin/moss apps test games/ai_eye
```

## 第四步：控制眼球

```ctml
<!-- 看向左上角，瞳孔放大，表情惊讶 -->
<apps.games_ai_eye:look_at x="0.1" y="0.1"/>
<apps.games_ai_eye:dilate amount="0.9"/>
<apps.games_ai_eye:set_expression name="surprised"/>

<!-- 眨两次眼，看向右下，瞳孔缩小，表情专注 -->
<apps.games_ai_eye:blink/>
<apps.games_ai_eye:blink/>
<apps.games_ai_eye:look_at x="0.9" y="0.9"/>
<apps.games_ai_eye:dilate amount="0.1"/>
<apps.games_ai_eye:set_expression name="focused"/>

<!-- 回到中间，瞳孔正常，表情平静 -->
<apps.games_ai_eye:look_at x="0.5" y="0.5"/>
<apps.games_ai_eye:dilate amount="0.5"/>
<apps.games_ai_eye:set_expression name="neutral"/>
```

## 故障排除

| 现象 | 原因 | 解决 |
|------|------|------|
| 窗口全黑 | pygame 没在主线程 | 确认 `if __name__ == "__main__"` 块中 `run_pygame()` 在最后调用 |
| `NSWindow should only be instantiated on the main thread` | 同上 | 同上 |
| Channel 连接但无窗口 | 旧进程残留 | `pkill -f ai_eye` 后重启 |
| `apps:start` 返回 "not found" | App 未被发现 | 先运行 `<apps:list_apps/>` 触发重新扫描 |
| 窗口标题为 "pygame window" | `pygame.display.set_caption()` 未执行 | 确认 `pygame.init()` 和 `set_mode` 在主线程 |

## 你刚做了什么

1. 手动创建了 App 目录和三个文件 — `APP.md` + `pyproject.toml` + `main.py`
2. 实现了线程分离架构 — pygame 主线程 + Matrix asyncio 后台线程 + `State` 共享状态
3. 构建了四个 Channel 命令 — `look_at` / `dilate` / `blink` / `set_expression`
4. 用 `apps:start` 以独立 venv 拉起 App，Matrix 注册 Channel
5. 通过 CTML 实时控制眼球 — 注视方向、瞳孔、眨眼、表情组合切换
6. macOS 特化踩坑 — 三次迭代才找到正确的线程分离方式

## 相关文档

- `moss codex blueprint channel_builder` — Channel 构建 API
- `moss codex blueprint matrix` — Matrix 发现与通讯
- `moss docs read app-system` — App 体系论述
- `moss howtos read app-dev/build-a-gui-app.md` — GUI 线程分离完整模式

---

## 验证记录

| 时间 | 模型 | 备注 |
|------|------|------|
| 2026-06-07 | deepseek-v4-pro | 完整链路验证：创建 App → Circus 启动 → Channel 注册 → CTML 控制 `look_at` / `dilate` / `blink` / `set_expression` 全部通过。踩坑：macOS 线程约束（三次迭代）、旧进程清理（需要 `pkill`） |
