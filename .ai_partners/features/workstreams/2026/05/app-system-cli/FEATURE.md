---
title: App System CLI
status: in-progress
priority: P1
created: 2026-05-18
updated: 2026-05-18
depends: []
milestone:
description: >-
  moss apps init 命令 + app_stub 脚手架 + 文档，完成 AI 自迭代闭环的基础设施。
---

# App System CLI

> Use `moss features set-status app-system-cli <status> -m "note"` to update state.

## Motivation

`HostAppStore.init_app` 已实现但引用的 `app_stub` 路径错误（指向不存在的 `ghoshell_moss.host.app_stub`），且 CLI 没有暴露 `init` 命令。补全这条链路：`moss apps init <group/name>` 从 stub 模板创建 App 脚手架。

start/stop 不走 CLI — 运行时由 AI 通过 AppStoreChannel 的 `start`/`stop` 命令控制。

## Key Decisions

1. **init 走 CLI，start/stop 走 Channel** — App 创建是开发期操作，启停是运行期操作。后者通过 CTML 由 AI 在 Shell 中实时调度
2. **stub 路径**: `ghoshell_moss.host.stubs.app` (不是 `app_stub`)。`stubs/app/` 是随包分发的 Python 模块，`init_app` 通过 `importlib.util.find_spec` 定位后复制
3. **CLAUDE.md 在 stub 中** — 每个新 App 自带 AI 开发者上下文
4. **init_app 始终写入 APP.md** — 不再只写 description 非空的情况，保证 frontmatter 完整
5. **返回值包含路径** — 方便人类和 AI 知道 cd 去哪里

## Implementation Notes

- `host/app_store.py:121`: `app_stub` → `stubs.app`
- `host/app_store.py:131-138`: 移除 `if description` 条件，始终生成 APP.md；返回值附加 `target_dir`
- `cli/apps_cli.py`: 新增 `init` 命令（函数名 `create_app`），支持 `--json` 和 `--description`
- `stubs/app/CLAUDE.md`: 暂时简版，后续随 app 体系文档完善

## MCP Self-Iteration Test (2026-05-18)

通过 moss-as-mcp 验证了 App 体系的运行时自迭代体验：

**通过的链路**：
- `moss apps init ai_tools/greeter` → 脚手架创建成功
- 编写 Channel App 逻辑 (`main.py` with `greet` command + `context_messages`)
- `<apps:list_apps />` → MCP 可见 STOPPED 状态
- `<apps:start fullname="ai_tools/greeter" />` → 启动成功，状态变 RUNNING
- `<apps:stop fullname="ai_tools/greeter" />` → 停止成功（修复后）

**发现的 bug 与修复**：
1. `start_app`/`stop_app` key 不一致 (fullname vs address) → stop 永远 "not under management"。已修复统一使用 `app.fullname`
2. MCP 重启后 `_managed_apps_with_fullname` 清空但 Circus 仍持有 watcher → add 重复失败。已修复：add 前先查 Circus 已有列表

**未通的路径**：
- `<apps.ai_tools_greeter:greet />` → "command not found"。App 进程启动成功，`get_virtual_children()` 返回 ChannelProxy（设计如此——外侧 bootstrap），但 Shell 未将 proxy 解析为可用命令。用户判断可能是 app store 通讯地址错误或 Shell 刷新问题，下个会话继续 debug

**下个会话入口**：
- 调试 ChannelProxy 在 Shell 树中的 bootstrap 链路
- 怀疑点：app store 通讯地址、Shell channel tree 刷新触发、`wait_connected` 未调用

## 文档 (2026-05-18)

- `docs/ai/model-oriented-application-system.md`: 完成初稿 — What App Is, Minimal Path, 5 种 App 类型, 依赖隔离, Mode 集成
- `stubs/app/CLAUDE.md`: 每 App 脚手架自带，简洁索引指向完整文档
- 待讨论：`apps/CLAUDE.md` (目录级约定) vs `apps/<group>/<name>/CLAUDE.md` (App 级上下文) 是否需要两份
