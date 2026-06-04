---
title: Build a MOSS App
description: 创建 MOSS app 前的决策路径：app 有哪几种形态、依赖怎么治理、从哪里开始。最小化知识，引导探索。
---

# How to Build a MOSS App

## 背景

App 是 MOSS 中独立的进程单元。Ghost 可在运行时创建、启动、调用、关闭它。

每个 app 有自己的目的（GUI、传感器、机器人驱动、web 工具……）。**Channel 不是一种 app 类型——它是模型控制 app 的手段。任何 app 都可以选择是否暴露 channel。**

## 第一步：决策

在 `moss apps create` 之前，确认两件事：

### App 的形态

- 是否需要让 Ghost 通过 CTML 命令控制？→ 暴露 channel（见 `moss codex blueprint channel_builder`）
- 是否独立运行、不参与命令体系？→ 纯进程（不接 Matrix）
- GUI 占主线程？→ Matrix 在异步侧运行
- 持续向总线推送信号？→ 使用 `Matrix.session` 生产信号

**要理解上述模式的具体写法，读生成的 CLAUDE.md**——它在每次 `moss apps create` 时产出，是当前版本的最新事实源。

### 依赖隔离级别

| 级别 | 方式 | 适用 |
|------|------|------|
| 独立 venv | `pyproject.toml` + `uv run` | 需要第三方库，长期维护 |
| 单文件 | PEP 723 内联元数据 | 少于 100 行，依赖极少 |
| 共享运行时 | 无 pyproject.toml | 快速原型，只依赖 `ghoshell_moss` |

`moss apps start` 和 `moss apps test` 都通过 `uv run` 启动——app 自动获得自己的 venv。**在 app 目录内不要执行 `uv sync --active`**，它会用 app 的依赖替换主项目 venv。

需要 Matrix 通讯时，`pyproject.toml` 中依赖 `ghoshell_moss[host]`。纯进程 app（不接 Matrix）可以用任何语言、任何可执行文件——`moss apps create` 生成的 Python 模板只是默认约定，不是硬约束。MOSS 只关心 `APP.md` 中的 `executable` 和 `script` 字段能否启动你的进程。

## 第二步：创建

```bash
moss apps create <group>/<name> -d "what it does"
```

产物：`APP.md` + `main.py` + `CLAUDE.md`。**先读 CLAUDE.md**——它包含入口模式、依赖管理细节、测试约定，是当前版本的最新事实源。

然后编辑 `main.py` 实现你的逻辑。

## 第三步：测试

本地 → MCP → 运行时三层递进。详见 `moss howtos read app-dev/test-an-app`。

## 第四步：集成

让你的 app 在特定 Mode 下可见或自动启动——读 `moss docs read workspace-and-mode`。

简要原则：`apps:` 白名单控制可见性，`bringup_apps:` 控制自动拉起。不带 bringup 的 app 由 Ghost 按需启动。

## 深入路径

- Channel 构建：`moss codex blueprint channel_builder` — 最小知识入口。更多知识路由到 docs/howtos
- CTML 语法：`moss ctml read` — 模型如何通过 CTML 控制你的 channel
- Matrix 通讯：`moss codex blueprint matrix`
- App 体系论述：`moss docs read app-system`
- 现有 app 参考：`.moss_ws/apps/` 下的正式 app
