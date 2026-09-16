---
created: 2026-09-16
depends: []
description: 'kill 一个 cell 应当是优雅关闭: SIGTERM → close() → Matrix 收尾. 顺带修日志轮换与退出控制流被记
  ERROR.'
milestone: null
priority: P1
status: completed
status_note: SIGTERM → Matrix close() 优雅退出; 日志轮换 midnight; 退出控制流不记 ERROR
title: Cell Graceful Exit
updated: '2026-09-16'
---

# Cell Graceful Exit

> Use `moss features set-status cell-graceful-exit <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss nodes kill <address>` 之后, cell 是**硬死**的:

- `Matrix.__aexit__` 完全没跑 — 没有 cell 下线 publish (mesh 消费者不知道它下线了),
  没有 channel / topic / session / Subprocesses 收尾, 设备 (音频采集 / TTS) 没停。
- `enter_cell_lifecycle` 的 finally 没跑 — 账本文件留在 `.moss/runtime/cells/`
  (`moss nodes status` 里显示为 stale), singleton 锁靠 OS 兜。

对照物证 (2026-09-16 21:51:03 起的 `llm_judge_probe`, pid 41708): 启动有完整日志,
21:52:53 被 SIGTERM 之后**一条收尾都没有**, 账本至今残留。同一日志里另一些 cell
退出时有完整收尾 — 差别在收到的是 SIGINT (矩阵侧 Subprocesses 停子进程走 SIGINT)
还是 SIGTERM (CLI kill 路径)。**两条停止路径语义不对称**: 一条优雅, 一条硬杀。

## Design Index

- 承接概念: `moss codex blueprint matrix` (Matrix 生命周期), `moss codex blueprint host`
  (`Host.run_until_closed` 的 SIGINT → close() 范式)
- 相关历史: `.ai_partners/features/workstreams/2026/08/node-lifecycle/`,
  `.../2026/08/matrix-operator/`

## Key Decisions

**1. 修在接收侧, 不修在发送侧。** `moss nodes kill` (NodeManager._terminate) 与
前台 `moss nodes run` 的信号转发都发 SIGTERM; 改发 SIGINT 也能"修好", 但只修了
MOSS 自己发的信号 — `kill <pid>` / `docker stop` / systemd 这些外部停止仍然是硬死。
SIGTERM 本来就是"请优雅结束"的标准信号, 该由进程自己处理。

**2. SIGTERM 走 `close()`, 不走 KeyboardInterrupt 注入。** MOSS 已有范式
(`Host.run_until_closed`): 注册 handler → `close()` → closing event → 运行循环自然唤醒
→ `__aexit__` 收尾, 而不是让 asyncio.run 暴力取消。照此对齐 `Matrix.run`:
`signal.SIGTERM → self.close()`(主线程才装, `Matrix.run` 允许在子线程跑)。

**3. 补上 Matrix 缺失的 `wait_close()`。** `MossRuntime` 有一对语义:
`close()` 置位关闭请求, `wait_close()` 等请求, `wait_closed()` 等**完全关闭**。
Matrix 只有 `wait_closed()`, 而 `arun` 里的 `exit_signal` 恰恰等了它 —
`_closed_event` 只在 `__aexit__` 里置位, 在 `async with` 体内永远等不到。
后果: ABC 承诺的 "close() = graceful exit" 从未生效, `close()` 变成只翻
`is_running()` 标志的空操作 (matrix 还在跑却报告 not running)。
现 `arun` 等 `wait_close()`, close() 成为真正的退出请求。

**4. 日志轮换 `when='d'` → `when='midnight'`。**
`TimedRotatingFileHandler` 的 rolloverAt = **文件 mtime + interval**, 不是自然日边界。
每个 moss 命令都是新建 handler 的短命进程, mtime 就是上次写入 → deadline 永远在
24h 之后 → **天天用反而永远不轮换**; 只有中断几天后再跑才轮换, 且备份名取 mtime 日期
(所以 `.moss/runtime/logs/` 里是 08-22 / 08-28 / 08-30 / 09-02 / 09-05 这种不规则名字)。
`midnight` 按自然日边界算 deadline, 新的一天第一次运行就轮换, 备份名 = 前一天。

**5. 退出控制流不记 ERROR。** `Project.__exit__` 把跨过 `with Project.discover()` 边界的
退出当故障记 `logger.exception` → `ERROR - -15 [project.py:840]` / 空 message 的
`ERROR -  [project.py:840]`。两半:
- 内置 `SystemExit` / `KeyboardInterrupt` 直接跳过 (core 不依赖任何框架);
- CLI 框架的 `typer.Exit` 由 CLI 层登记: `cli/main.py` 调
  `register_control_flow_exit(typer.Exit)` — "谁持有'这是正常退出'的知识, 谁登记",
  依赖方向仍是 CLI → core。

**6. cell 日志走 per-cell 唯一文件, 名字不带 uid。**
多进程竞争的真源是"node 子进程往 owner 的 moss.log 写"。拆法: 有 `MOSS_CELL_ADDRESS`
(spawner 注入子进程) → 写 `moss.{role}__{name}.log`, 否则 (host / CLI 一次性命令) →
`moss.log`。**稳定身份不带 uid** —— 同 `locker_name()` 的取舍 ("uid 不进锁名"), uid
只区分同名并发实例, 带进文件名会每跑一次留一个文件、无界膨胀。`role`/`name` 用
`CellAddressCodec` 从 address 取 (唯一解析入口, 不手写 split)。同名非 singleton 并发
共享一个文件, 可接受。**装配分层**:

- `Project.log_file` 是"默认机制" —— 有 `this_cell_address` 就返回 per-cell 路径,
  否则 moss.log, 裁决在属性里, 不在 handler 装配里。
- `contracts.logger.bind_moss_file_handler(logger, log_file)` 是唯一绑定逻辑
  (midnight 轮换 + 按 `MOSS_FILE_HANDLER_NAME` 幂等去重)。
- `Project._ensure_log_file_handler` (首选) 调它; `MatrixLoggerProvider` (兜底, 不是
  首选) 也调它, 共享同一 handler name 去重, 退化路径只绑 base moss.log —— 不再持有
  第二份轮换/命名真源 (曾因此漏改 `when='d'`)。

**7. 不用 QueueHandler/QueueListener。** 那是**单进程**内的异步日志
(`queue.Queue` 是线程内内存队列), 不解跨进程轮换竞争 —— 解决竞争靠的是唯一文件名,
不是队列。真跨进程单写者要 `SocketHandler`/`multiprocessing.Queue` + 独立 logger 进程,
更重且当前无需求。非阻塞写是另一个问题, 若未来实测"文件 IO 卡 duplex 循环"再议。

## Implementation Notes

改动 (2026-09-16, deepseek-flash):

| 文件 | 改动 |
|---|---|
| `core/blueprint/matrix.py` | ABC 加 `wait_close()`; `arun` 的 exit_signal 改等 `wait_close()`; `run()` 装 SIGTERM→`close()` (主线程, finally 还原) |
| `matrix/matrix_impl.py` | 实现 `wait_close()` |
| `core/blueprint/project.py` | `_CONTROL_FLOW_EXITS` + `register_control_flow_exit()`; `__exit__` 跳过登记类型; `log_file` 属性做 per-cell 命名裁决; `_ensure_log_file_handler` 改为调 `bind_moss_file_handler` |
| `contracts/logger.py` | 新增 `MOSS_FILE_HANDLER_NAME` + `bind_moss_file_handler()` (midnight 轮换 + 幂等去重) |
| `matrix/providers/logger_provider.py` | 恢复为兜底: 复用 `bind_moss_file_handler` + 同名去重, 退化路径绑 base moss.log |
| `cli/main.py` | `register_control_flow_exit(typer.Exit)` |
| `core/blueprint/environment.py` | `log_file` 属性 (供 `project where` 显示日志路径) |
| `cli/project_cli.py` | `moss project where` 增 `Log File` 行 |
| `core/blueprint/host.py` | `run_until_closed` 增 SIGTERM → `self.close()` (镜像 SIGINT) |
| `cli/ghost_run.py` | `_run_ghost_headless` 增 SIGTERM → `ghost_runtime.close()`; 澄清收尾由 main() 的 `async with` 反卷 |
| `cli/moss_as_mcp.py` | SIGTERM → `signal.default_int_handler` (走 asyncio.run 取消路径, `async with moss_host.run()` 正常收尾) |

验证 (端到端, 非纸面): `moss nodes run .moss/system_test_nodes/signal_sender` 起来后
`moss nodes kill node/signal_sender/...` → 日志写入
`moss.node__signal_sender.log` (稳定名, 无 uid), 收尾完整 (provider loop cancelled /
channel closed / topic publish loop stopped / session closed / Subprocesses stopped),
账本删除, 前台 CLI 退出码 0, moss.log 零新增, 无 ERROR。
`moss --ai nodes install <不存在路径>` → 不再产生 ERROR traceback, 退出码仍为 1。
测试: `tests/ghoshell_moss/matrix/project` + `tests/ghoshell_moss/default/core/blueprint`
+ `tests/ghoshell_moss/matrix` 共 495 passed。

**未做 / 待办:**

1. **moss.log 仍是多写者 (host + 并发 CLI 短命令) / 同名非 singleton cell 共享文件** —
   per-cell 唯一名已消除"node 子进程 vs owner"的主竞争, 但 host + 恰好跨午夜的 CLI 命令、
   以及同名非 singleton 并发仍是窄边角竞争, 概率小, 未加锁。要彻底需给 moss.log 的
   轮换加 flock, 或非 singleton 也拆实例名 — **未动, 待定**。
2. **存量 stale 账本** — 本次修复前硬杀的遗留 (5 条 `llm_judge_probe`), 可
   `moss nodes prune` 清。(用户已确认无需处理。)
3. **host/ghost/mcp 的 headless SIGTERM 已补齐** — `run_until_closed` / 
   `_run_ghost_headless` / `moss_as_mcp` 三处 headless 入口都已装 SIGTERM (模型
   自迭代场景: headless 启动后 kill 优雅退出, 不留孤儿进程)。TUI 面不走 kill, 未动。