---
title: App System CLI
status: in-progress
priority: P1
created: 2026-05-18
updated: 2026-05-18
depends: []
milestone:
description: >-
  moss apps init 命令 + app_stub 脚手架，让 AI 能通过 CLI 一键创建可运行的 App 骨架。
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
