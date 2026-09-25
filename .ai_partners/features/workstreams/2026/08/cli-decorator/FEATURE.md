---
created: 2026-08-13
depends: []
description: '@ghoshell_moss.decorators.cli — 把真实 CLI 命令包成签名即契约的普通函数 (code-as-prompt)。'
milestone: null
priority: P1
status: completed
status_note: decorator + moss_cli dogfooding landed; git safe wrapper is a separate
  future task
title: Cli Decorator — 环境依赖 decorator 引入
updated: '2026-08-13'
---

# Cli Decorator

> Use `moss features set-status cli-decorator <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

memento agent 的唯一工具是 `sandbox_exec` (任意 Python)，授权面过大。`@cli` decorator 是一种
**快速可授权方式**：把真实 CLI 命令包成普通可调用函数，函数签名即工具签名，反射穿透到模型
(code-as-prompt)。agent 在 sandbox 里 import 工具，即声明授权；调用时在宿主侧执行。

这不是新动机。ghostos (2024) 的 `GhostFunc` decorator 已是同型先例：
`GhostOS/libs/ghostos/ghostos/prototypes/ghostfunc/decorator.py` — 构造时注入 Container，
包装后函数签名即契约，执行委托给 driver。`@cli` 把「执行」换成 SubprocessFacade 的 exec 模式
子进程，依赖从 project-level IoC 取。

前置基建已落地并提交：`contracts/subprocesses.py` 拆分 spawn-only facade 面 + owner 治理面；
`SubprocessesImpl` 惰性启动；provider 加 `SubprocessFacade` alias — decorator 是纯消费方，
只依赖 facade，生命周期方法在类型层不可见。

## Design Index

- 关键设计文档: `design/`（未建）
- 关键讨论记录: `discuss/`（未建）
- 前置基建: `contracts/subprocesses.py`, `core/subprocesses/_impl.py`, `project/providers/subprocesses_provider.py`

## Key Decisions

### 1. `prefix` 必选位置参数，`name=` 可选覆盖
最初草稿是 `@cli(name='git')`（名字兼当可执行）。落地时拆开：`prefix` 是命令前缀
(字符串或 argv 列表)，`name` 只覆盖工具显示名，默认取被装饰函数名。
理由：point 2 要求「命令前缀」是核心输入，argv[0] 必须是可执行文件。

### 2. 函数本体 = 纯声明 (签名 + docstring)，运行时不被调用
行为通过注册参数实现：`input_filter` (入参过滤, argv→argv, 可抛异常) + `output_processor`
(出参加工, 三元组→三元组, 形状不变) + `facade`/`cwd`/`timeout` (spawn 配置)。
**类型注解 `-> tuple[int, str, str]` 是模型看到的界面，也是真实返回** — 装饰器总是返回
(code, stdout, stderr)，output_processor 必须保持形状，否则 code-as-prompt 撒谎。

**入参过滤与出参加工不对称**：入参过滤是结构性的 (决定执行什么)，必须住在工具里 — 否则
module-level 工具无法自包含；出参加工大半是展示性的 (cap / exit tail)，channel 可以自己
做。所以 output_processor 是可选便利，展示格式化留在 seat。

### 3. `-h`/`--help` 只在注册 `help=` 时拦截
`help` 接受 str 或 callable。未注册时 `-h`/`--help` 原样传给子进程（如 `echo -h`）。
拦截只匹配「单独成参的 `-h` 或 `--help`」，不做参数内任意位置匹配。

### 4. 所有依赖惰性加载，定义在 decorator 内部
`SubprocessFacade` / `Project` 等只在首次调用工具时才 import 并从 project IoC 取。
模块 import 阶段零副作用。理由（memento sandbox 语义）：agent .py 编译期是唯一允许新模块
加载的时刻 (recording `__import__`)，工具宿主依赖不该成为 agent 编译期的 import 面；
调用时 wrapper 的 frame globals 是宿主 `cli.py` 模块 dict，内部 `from ... import` 走真实
`__import__`（非沙箱 replay），所以沙箱里调工具也能在宿主侧拿到 IoC。

### 5. 执行时序: wait 之后必须先 wait_drained 再读
`process.wait()` 返回后 drain 协程可能还没读到 EOF，直接读 stdout/stderr 会静默丢输出
（短命令尤其明显）。顺序必须是 execute → wait → wait_drained → read。
超时路径同理：`asyncio.wait_for` 取消的是 wait 协程，子进程还在跑 — 必须 `proc.stop()`
兜底否则成孤儿。超时返回 `(124, 已捕获输出, "[timeout after Xs, stopped]")`。

### 6. 工具消费 IoC，不声明契约
tools 声明保留 "no IoC contracts" 本意：tools 不声明 ioc abstract 类，但可消费
project-level 契约 (SubprocessFacade)。

### 7. 模块级 vs 局部函数：同一 decorator，两个绑定模式
- **module-level**：绑定固定配置 (默认 facade → project IoC 单例)，可导入 = agent 授权面，
  code-as-prompt 位置稳定。适用于共享能力。
- **局部糖形式**：decorator 做边界，closure 绑定注入的 `processes`/`cwd`/`timeout`。适用于
  channel-bound 工具，保留注入缝。**moss_cli dogfooding 用了此形式**。
- 决定因子：工具是否要可导入/共享。channel 命令天然 channel-bound，用局部糖；agent 工具用
  module-level。

### 8. ctx 分支：剪掉 — "ctx = outgrown the decorator"
per-call 可变配置 (cwd/timeout) 没有参数空间。contextvars ctx 推理后剪掉：
- 薄 ctx (per-call cwd/timeout 覆盖)：无消费者 + 双配置通道歧义 (定义期绑定 vs ctx 谁赢)。
- 厚 ctx (拿 request + 注册回调)：middleware 架构，是「超出 decorator 承载力」的哨兵 —
  需要 request-scoped 状态 + 回调的工具应升级为接口/类，不是 decorator 语法糖。
- MOSS 已有 ChannelCtx (contextvars)，channel-bound 工具可复用，不新建平行 ctx。
- 真出现 per-call 需求时 vehicle 现成，重开成本低。

### 9. dogfooding moss_cli 验证
exec 命令收缩为 `@cli(...)` 声明 + channel 薄包装：手动 subprocess 处理 (execute/wait/
wait_drained/超时/输出组装) 全部被 decorator 吸收。channel 只留生命周期 + instruction +
exec_cmd 薄包装 (empty 检查 + 展示格式化)。**per-call `timeout` 参数从 exec 签名删除** —
timeout 定义期绑定，per-call 可变需要 ctx (已剪)。实现 cli channel 提速目标达成。

## Implementation Notes

- `python -m moss` 目前不合法：`ghoshell_moss` 无 `__main__.py`，`moss` 是 console script
  (`ghoshell_moss.cli:main_entry`)。前缀用 `["moss", ...]` 或 `[sys.executable, "-m", "<module>"]`。
  是否补 `__main__.py` 让 `python -m moss` 合法——待定，未实现。
- 测试环境相关：decorator 的 standalone/facade/cwd 用例依赖仓库 `.moss` workspace
  (facade=None 时走 Project.discover)。moss_cli channel 用例注入 SubprocessesImpl，不依赖 workspace。
- 文件: `src/ghoshell_moss/decorators/{__init__,cli}.py`, `tests/ghoshell_moss/decorators/test_cli.py`,
  `src/ghoshell_moss/channels/moss_cli.py` (dogfooding 重构)
- 动机 (helper/warm 数据)：channel helper 要 gather 全部命令跑 `moss --ai all-commands`
  的 help 集合 — 干净的 bash 糖 (exec 模式无 shell) 是生产 warm 数据的载体。这是 decorator
  的真实驱动，不是装饰性需求。
- 原型验证了 9 个评估点全 PASS：反射穿透、单独运行、PyCommand 反射、channel 运行、help 拦截、
  IoC 取数。原型在 /tmp，未入库。
- macOS 上 `/tmp` 是 `/private/tmp` 的符号链接 — facade/cwd 测试断言用 `Path("/tmp").resolve()`。