---
created: 2026-05-18
depends: []
description: moss apps init 命令 + app_stub 脚手架 + 文档，完成 AI 自迭代闭环的基础设施。
milestone: null
priority: P1
status: completed
status_note: init/start/stop/list 全链路 MCP 验证通过，核心功能交付
title: App System CLI
updated: '2026-05-19'
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

## Proxy 连通性调试与 timeout 机制 (2026-05-19)

**通过的修复**：

1. **`get_virtual_children()` key 对齐**: `_app_channels` dict 内部 key 从 `address` 改为 `fullname`，与 `get_virtual_children()` 对外返回的 `name()` 推导路径一致。消除三套 key (address/fullname/proxy.name) 的不一致
2. **`MatrixImpl.channel_proxy()` 权限检查修复**: `HostMainCell.type` 是 `'host'` 不是 `'main'`，原来 `self.this.type != 'main'` 永远为 True，阻止所有主 cell 创建 proxy。改为 `not self._is_main`
3. **`start` 命令 timeout 参数**:
   - `-1` (默认): 不等待，立即返回
   - `0`: 无限等待直到 connected
   - `>0`: 等待 N 秒后超时返回 WARN
   - 采用 tree 路径: `ChannelCtx.runtime()` → `refresh_metas()` → `fetch_sub_runtime()` → `wait_connected()` → `refresh_metas()`，不走直接 bootstrap proxy
4. **单元测试**: 8 个测试覆盖 proxy 连通性、bootstrap 时机、timeout 全分支

**发现的 bug — Zenoh 通讯协议未通**：
- App 进程启动成功 (`[RUNNING]`)，但 proxy 永远连不上
- 怀疑：`zenoh_config_main.json5` 与 `zenoh_config_app.json5` 配置不一致，导致 App Cell 的 Zenoh 网络与 Main Cell 不互通
- 已知但未验证：App Cell 启动时使用的 Zenoh config 路径不同
- **解决方案方向**: 统一 Zenoh 配置来源，或在 App 启动时传递正确的 Zenoh 配置

**ping_test app**:
- `.moss_ws/apps/ai_tools/ping_test/` — 最简连通性测试 App (ping/echo 命令)
- 单元测试中用 `ZenohChannelProvider` + `ZenohProxyChannel` 原语验证通过
- MCP 实测时 App 进程正常启动但 channel 未连通（上述 config 问题）

## Zenoh 连通性根因修复 (2026-05-19)

人类工程师定位并修复了 proxy 永远连不上的两个根因：

1. **`AppInfo.make_address` 魔法值**: 硬编码 `f"apps/{fullname}"` → 改为 `Cell.make_address("app", fullname)`，统一地址生成入口。`Cell.make_address` 同时增加了 `str(cell_type).lower()` 归一化，`CellTypes` Literal 改为 `CellType(StrEnum)` 并新增 `script` 类型。
2. **`send_command_task` chan fallback**: `chan: str = ''` + `chan or task.chan` 导致当 chan 为空字符串时 fallback 到 consumer 侧路径 `task.chan`，provider 侧寻址失败。改为移除默认值，重命名为 `provider_side_chan_path`，所有调用方显式传入正确的 provider 侧路径。

## list_apps refresh 缺失修复 (2026-05-19)

`AppStoreChannelState` 的 `list_apps` 命令调用 `get_apps_context()` 未传 `refresh=True`，导致新创建的 App 目录在当前会话中不可见。

- `AppStore.get_apps_context` 新增 `refresh: bool = False` 参数
- `HostAppStore.get_apps_context` 透传至 `list_apps(refresh=refresh)`
- `AppStoreChannelState.list_apps` 命令传 `refresh=True`

## MCP 全链路验证通过 (2026-05-19)

从零开始创建 `ai_tools/calc`，完整走通 AI 自迭代闭环：

```
moss apps init ai_tools/calc → 脚手架创建
写 main.py (add/multiply/div + context_messages)
<apps:list_apps /> → [STOPPED] 可见 (refresh 生效)
<apps:start fullname="ai_tools/calc" timeout="3.0"/> → [OK] connected and ready
<apps.ai_tools_calc:add a=3 b=7 /> → 10.0
<apps.ai_tools_calc:multiply a=6 b=8 /> → 48.0
<apps.ai_tools_calc:div a=10 b=3 /> → 3.333...
<apps.ai_tools_calc:div a=1 b=0 /> → Error: division by zero
```

这是 MOSS 架构下第一个端到端走通的运行时自迭代链路。AI 不再是固定的工具调用者，而能在运行时通过 init → 写码 → start → call 扩展自己的能力边界。

## 当前状态

核心链路已通。feature 保持 in-progress，主任务回到文档体系完善。下阶段：
- 文档: `model-oriented-application-system.md` 补充自迭代验证记录
- 讨论: App 级 CLAUDE.md 约定
- 清理: 删除测试用 ping_test/greeter/calc ws 目录

## 文档 (2026-05-18)

- `docs/ai/model-oriented-application-system.md`: 完成初稿 — What App Is, Minimal Path, 5 种 App 类型, 依赖隔离, Mode 集成
- `stubs/app/CLAUDE.md`: 每 App 脚手架自带，简洁索引指向完整文档
- 待讨论：`apps/CLAUDE.md` (目录级约定) vs `apps/<group>/<name>/CLAUDE.md` (App 级上下文) 是否需要两份