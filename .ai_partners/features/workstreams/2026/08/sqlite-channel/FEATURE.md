---
created: 2026-08-05
depends: []
description: sqlite3 数据库作为"文件资源协议"——单 channel 持有多个 sqlite 连接，query(name, sql) 为唯一
  SQL 入口，context 常驻连接映射，给模型 file editor 式的数据库理解手段。d3 的前置。
milestone: null
priority: P1
status: completed
title: Sqlite Channel
updated: '2026-08-05'
---

# Sqlite Channel

> 关键命题：把 sqlite3 数据库当成一种"文件资源协议"。
> 给模型类似 file editor + read + list 的手段去理解一个数据库；
> 对同一数据库的生产/消费逻辑由别的技术独立去做（d3 node），跨进程共享同一 `.db`。

## Motivation

- MOSS channel 树本身就是资源协议：instruction/interface/context 是统一认知面，
  数据库只是一类新资源节点，无需发明协议。
- 杠杆点：sqlite 本身是跨进程共享文件。ghost 进程的 channel 与 d3 node 进程
  直接开同一个 `.db`，文件就是总线，零协议。这是 MCP 给不了的。
- 数据库是**给 ghost 用的，不是给人类**——schema 是面向模型的面。
- d3 需要它：webview node（reflex/js）+ 复合 command（文件 + sql 脚本 + 数据转换）
  跨进程完成对同一数据库的利用。

## Design Index

- 讨论上下文：2026-08-05 与人类协作者的机制调研 + 设计碰撞（virtual channel 决策被推翻）。
- 实现：`src/ghoshell_moss/channels/sqlite_channel.py` + `tests/ghoshell_moss/channels/test_sqlite_channel.py`。
- 关联框架改动（人类 + 模型协作）：`core/concepts/channel.py`（PATH_SEPARATOR）、
  `core/concepts/interpreter.py`（`Interpreter.run()` 语法糖）、
  `core/ctml/token_parser.py` + `test_token_parser.py`（子 channel 路径解析）。

## Key Decisions

1. **以 channel 承载，而非 MCP**。MCP 是"进程间走协议"；sqlite 是"文件就是协议"。
   channel 是 MCP 的降级形态——CTML 接管调度（mcp_hub docstring 同思路）。
   `mcp-fusion-point` workstream 与此同线。
2. **单 channel 持有 N 个连接，以 name 寻址**。命令面：`open/close/list`（生命周期）+
   `tables/schema/sample`（探索）+ `query(name, sql)`（唯一 SQL 入口）。
   单 channel 内命令 FIFO，`open→query→close` 严格有序。
3. **【推翻】不用 virtual channel**。最初设计是 hub + virtual child（`add_virtual_channel`
   动态开子节点），踩到两个坑：
   - virtual channel 下一轮才生效，同流 open→query 时序无法保证；
   - 父命令（open/close）与子命令独立调度，close 可抢在子命令前移除子节点，
     pending 子命令卡死且无报错。
   结论：单 channel + name 寻址，无父子时序竞争。
4. **【推翻】不用 `__content__`**。自由文本路由对多库定位太绕（content 无法带 name 参数）。
   SQL 只走 `query(name, sql)`（text__ 流式 body）。
5. **context 常驻"open connections"映射**（name -> db_path）。`context_messages` 每次
   refresh 生成，几十 token。列/数据按需取——模型永远知道"手上有哪些库"，深度信息进历史。
6. **大结果集封顶 + 可选落盘**。内联结果按 `max_rows`（默认 100）/`max_chars`（默认 4000）
   截断并标 `truncated: N rows total`；溢出时若 `results_dir` 存在则全量落盘返文件路径。
   `results_dir` 未传入时 channel 自动分配临时目录，关闭时删除（复用 session tmp_storage 约定）。
7. **WAL + busy_timeout 是跨进程成败关键**。`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=...`
   保证多进程读 + 单写互不阻塞。
8. **read_only 开关**。for-ghost 的风险是 ghost 损坏自己的记忆，故 read_only + WAL 备份纪律
   > 权限模型。
9. **Ground 负责发现**。verbs 有 `ls`，锚定数据库目录即可列出 `.db`；无需新 verb。
   （dolores 场景：ground 锚定 sqlite 目录 + skill.md 配合发现。）

## Implementation Notes

- **命令 observe 约定**（channels/CLAUDE.md）：`open/query/tables/schema/sample/list`
  `always_observe=True`（信息）；`close` `always_observe=False`（确认）。显式标注不依赖默认。
- **docstring 首行**：`一句话 | 类型 | 状态`（`认知模块 | beta`），对接 `moss codex channeltypes`。
- **现成先例**：
  - `sqlite_cache.py` / `session_parameter.py` — 项目内已验证 sqlite3 用法
    （check_same_thread=False、WAL、跨进程）。session `tmp_storage` 约定
    `[ws]/runtime/tmp/session-[scope]`，启停清理。
  - `nodes/webview_apps/text_blocks/` — d3 的骨架已存在："Reflex webview 从 store 轮询"。
- **测试范式**：`new_ctml_shell` + `interpreter.run()`（多轮）+ `shell.channel_metas()`。
  meta 是快照，取前先 `await shell.refresh_metas()`。context 提取用
  `Message.content_as_string(c) for m in meta.context for c in m.contents`。
- **遗留**：session 的 `tmp_storage` 实现疑似有 bug（人类 2026-08-05 提及，未深挖）；
  连接关闭时未对未关闭的查询做守卫（query 先于 close 时 close 会等待 FIFO）。
- 预算：~200 行 + 7 个测试，全绿（channels + ctml 套件 348 passed 无回归）。