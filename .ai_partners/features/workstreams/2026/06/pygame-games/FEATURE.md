---
created: 2026-06-04
depends: []
description: 在 apps/games 分组下创建最小依赖、开箱可用的 pygame 游戏 App，让 AI 通过 Channel 参与回合制游戏或控制图形化身（如
  AI 眼球）。
milestone: null
priority: P2
status: completed
title: Pygame Games — AI 可参与的回合制游戏与可控图形化身
updated: '2026-07-19'
---

# Pygame Games

> 让 AI 有一张脸、一双手——在 pygame 窗口中与人类对弈，或用眼球注视人类。

## Motivation

MOSS 的 App 体系已经证明了"AI 可在运行时创造和操作能力"。但验证闭环偏向开发者视角（calc、greeter、ping_test）——缺少让**非技术用户**直观感受"AI 在和我互动"的场景。

pygame 游戏填补这个空位：
- **证明 MOSS 的交互性不是纸上谈兵**——人类在 pygame 窗口走一步棋，AI 在 CTML 通道里回一步，真实可感
- **为 Ghost 提供"脸"**——AI 眼球是最小的化身锚点。说话时眨眼、思考时瞳孔放大、跟随焦点移动——让人类直观感受到"它在看着我"
- **开箱可用**——不需要摄像头、不需要机器人、不需要额外硬件。一个 `uv run` 就能启动

## Design Index

- 每个游戏 App 的设计细节记录在：`feature/design/`
- 讨论轨迹记录在：`feature/discuss/`

## Key Decisions

### KD1: 分组为 apps/games，每个游戏独立 App

所有游戏放在 `apps/games/` 分组下，每个游戏是独立 App 目录：
```
apps/games/
  gomoku/          # 五子棋
  ai_eye/          # AI 眼球
  reversi/         # 黑白棋（后续）
```

每个 App 有独立的 `pyproject.toml`（pygame 依赖隔离）、`APP.md`、`main.py`。

**Why**: App 体系的目录约定天然支持。独立进程 = 崩溃不拖垮 MOSS。独立 pyproject.toml = pygame 依赖不污染主环境。

### KD2: 依赖极简——pygame 唯一重型依赖

每个游戏 App 的依赖锁定在 `pygame`（以及 MOSS 自身的 `ghoshell-moss`）。游戏 AI 策略内嵌在 App 代码中（不依赖外部模型推理），用经典的 minimax/alpha-beta 或简单规则。

**Why**: "开箱可用"是硬约束。用户 clone 项目后 `uv sync` 就能跑，不需要下载模型权重、不需要 CUDA、不需要额外服务。pygame 是 PyPI 最广泛安装的包之一，跨平台零配置。

### KD3: 回合制游戏 — AI 通过 Channel 参与

回合制游戏的交互模式：

```
人类在 pygame 窗口操作 → 游戏状态变更
→ context_messages 将棋局状态注入 AI 上下文
→ AI 推理并输出 CTML: <games.gomoku:move row="7" col="7"/>
→ pygame 接收并渲染 AI 的落子 → 等待人类下一步
```

Channel 暴露的命令（以五子棋为例）：
- `move(row, col)` — AI 落子
- `reset()` — 重置棋盘
- `undo()` — 悔棋

`context_messages` 返回当前棋盘的文字表示（15x15 网格），供 AI 推理。

**Why 回合制而非实时**: 回合制天然匹配当前 Ghost 的关键帧思考模式。实时游戏需要流式推理（尚未成熟），回合制每个关键帧有完整的决策上下文。且回合制是人类与 AI 对弈的最自然形式。

### KD4: AI 眼球 — 最小的 AI 化身锚点

AI 眼球是一个**非游戏的图形 App**——它提供的是"AI 存在感的视觉载体"。

眼球通过 Channel 命令实时控制：
- `look_at(x, y)` — 注视屏幕坐标，眼球平滑跟随
- `dilate(radius)` — 瞳孔缩放（0.0=针尖, 1.0=最大），映射"兴趣/惊讶"
- `blink()` — 眨眼
- `set_expression(name)` — 预设表情：neutral/curious/surprised/focused/sleepy

眼球可以独立运行（纯装饰），也可以配合 Ghost 对话使用——AI 说话时自动眨眼、听到人类输入时瞳孔放大。

**Why**: 这是 MOSS 哲学最直接的视觉化——"AI 降临到屏幕上"。比终端输出更直观，比 TUI 更有"身体感"。不需要 Live2D、不需要 Blender、不需要 3D 引擎——两个椭圆 + 一个圆就能做出有灵魂的眼睛。

### KD5: App 类型选择 — GUI App + Channel

每个游戏 App 同时是 GUI App 和 Channel App：
- pygame 主循环在主线程运行（GUI App 模式）
- Matrix 在 asyncio 侧运行，注册 Channel
- 两者通过 `asyncio.to_thread` / 共享队列协调

**Why**: 不是二选一。游戏必须渲染窗口（GUI），AI 必须能调用命令（Channel）。这是 MOSS App 体系中 GUI App + Channel App 的典型组合模式。

### KD6: 先做五子棋 + AI 眼球，预留扩展

首批实现：
1. **五子棋 (Gomoku)** — 验证回合制游戏全链路
2. **AI 眼球 (AI Eye)** — 验证 AI 实时控制图形

后续可扩展：黑白棋、象棋（需 python-chess）、21 点、表情化脸。

**Why**: 两个 App 覆盖两条交互线（回合制决策 vs 实时控制），验证模式后其他游戏是重复劳动。三个以上初始 App 会稀释开发质量。

## Implementation Notes

### pygame 与 asyncio 的共存

pygame 的事件循环是同步的、阻塞的。Matrix 的通讯是异步的。共存方案：

```python
async def game_loop():
    while running:
        for event in pygame.event.get():
            # 处理人类输入
        # 处理来自 AI 的命令队列
        # 渲染
        pygame.display.flip()
        await asyncio.sleep(0.016)  # ~60fps, 让出控制权给 asyncio
```

主入口：
```python
async def main(matrix: Matrix):
    await matrix.provide_channel(channel)
    await game_loop()

if __name__ == "__main__":
    Matrix.discover().run(main)
```

### 五子棋 AI 策略

内嵌 alpha-beta 剪枝搜索。搜索深度 2-4 层（pygame 进程中运行，不阻塞 Matrix）。AI 落子通过 Channel 命令触发（不是自动——AI 必须显式调用 move），保留"AI 决策"的透明性。

### AI 眼球的数学

2D 眼球渲染用椭圆 + 圆。注视方向映射到瞳孔在眼白内的偏移。平滑跟随用 lerp（线性插值），避免瞬移。眨眼用正弦波控制上眼睑高度。

瞳孔缩放公式：`pupil_radius = 0.15 + dilation * 0.35`（相对眼白半径的比例）。

### 依赖清单

每个 App 的 `pyproject.toml`:
```toml
[project]
dependencies = [
    "pygame>=2.5.0",
    "ghoshell-moss>=0.1.0",
]
```

主进程的 `pyproject.toml` 不添加 pygame——依赖隔离在 App 内。