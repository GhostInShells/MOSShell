# Cells Channel — M8 设计定案

2026-07-13 讨论定案 (claude-fable-5 + 人类架构师)。本文档是 M8 的实现依据，
供并行化身直接开工。上游判决引用见 matrix-cell-governance FEATURE.md
(§UU-4/UU-7/UU-8/WW-5/WW-6/WW-7)。

## 0. UU-9 纠偏 (最重要的上下文修正)

UU-9 原文 "moss_self 反射 CLI 成 channel, 两个面免费获得" —— 写判决的模型
不了解 channel 实现，与实际论述有偏差。**纠偏: cells channel 手写，
运行时绑 Matrix** (IoC 获取，command 内直接调 `matrix.run_cell` /
`matrix.mesh()` / `matrix.processes`)。理由:

- run/stop 必须 in-process — owner 得是 host 进程。shell out 到
  `moss cells run` 会让 owner 变成瞬态 CLI 进程，"子进程不比 owner 活得久"
  破，WW-6 dead 信号源 (Subprocesses done callback) 只对 owner 生效。
- accept/release 是 Watcher 进程内状态 (proxy owner = accept 者, UU-8)。
- create/install (纯文件操作) 可与 CLI 共享底层函数 — UU-9 "一份实现"
  在这两个动词上成立，其余动词共享的是咽喉 (Matrix 面)，不是 typer 函数。

## 1. 拓扑定案: 单 channel + nonblocking own commands + virtual proxies

```
cells                      top; own commands: run/stop/create/install/list/status
│                          (全部 blocking=False; auto_accept 关时 + accept/release)
├─ cells.foo   (virtual)   挂载的器官 (channel proxy)
└─ cells.bar   (virtual)
```

- 模型语法: `<cells:run target="foo"/>` → 下帧 `<cells.foo:say .../>`。
- own command 与 virtual child 在 CTML 中不同命名空间 (`cells:run` vs
  `cells.run:x`)，无名字碰撞。
- **漏斗分析**: 六动词相互无时序依赖 (重复冲突由咽喉 DuplicatedError 挡，
  不靠通道串行)，全部 `blocking=False` → 永不 occupy → 治理动词不会
  pending 掉器官调用。漏斗机制留给真需要"父动作冻结子树"的 channel。

**否掉路径 (备查，防复推)**:

1. **治理子 channel (`cells.manage` / `cells._discover`)** — 立子 channel 的
   唯一结构理由是漏斗 (怕 blocking 治理动词阻塞器官)；nonblocking 后动机
   消失。路径更短，无下划线争议。
2. **`_` 前缀命名** — 训练分布中强烈读作"内部勿碰"，run/stop 是 telos
   主动词，每次调用有认知摩擦。若未来子 channel 复活，`_` 前缀可接受
   (换取与 proxy 名永不碰撞，需禁止 cell 名以 `_` 开头)。
3. **proxy 挂 main channel 顶层** — 误解，proxy 从 cells 的
   get_virtual_children 返回，CTML 树状寻址。

## 2. foreign 挂载: 构造期 flag

- **local (同 project / 自己拉起的) 永远自动挂** — UU-7 信任语义，不受 flag 管。
- 构造参数 `auto_accept: bool` (命名实现期可调) 只管 foreign:
  - `True`: foreign 也自动挂；**accept/release 命令不注册** (不是藏起来，
    是不存在 — 模型看不到用不上的动词)。deny/release 一并消失
    (否则 release 后下帧 refresh 自动挂回，语义自相矛盾)。
  - `False` (默认): foreign 需显式 `cells:accept address=...`。
- **与 UU-8 的和解**: UU-8 判死的是机制层默认偷走 accept 动词 (AppStore
  每帧遍历建 proxy，没人做过决定)。构造期 flag 是嵌入者的显式治理决定，
  code as prompt 可见 — accept 决策从"每次一个地址"提升到"一次全网络"，
  决策主体仍在治理面。

## 3. proxy 实例稳定

- virtual children = **已挂载的** (local 自动 + foreign accept 的)，
  不是"网络上看得见的全部"。
- 机制: `refresh_meta` (async) 对账 Watcher / 更新挂载集，
  `get_virtual_children` (sync) 返回稳定 dict — Builder.refresh_meta
  docstring 写死的协作模式。
- 稳定性由 accept/mount 语义保证: 挂载时建一次，卸载前实例不变。
  `Watcher.accepted` 即稳定 dict 的真相源。
- **AppStore 反模式不抄** (app_store_channel.py 备查): get_virtual_children
  遍历 list 急切建 proxy (UU-8 病灶原型)、start(timeout=) wait 参数
  (M5.1 已拿掉)、cache dict 当 owner。

## 4. 信息三分

| 层 | 内容 | 依据 |
|---|---|---|
| **instruction** (静态) | 动词协议: 六动词语义 / run 返回 CELL.md body / accept 后器官下帧自现 / 未安装报 INSTALL.md 路径 / "你 run/accept 的 cell 作为子 channel 出现在这里" | code-as-prompt |
| **context_messages** (动态, 每 refresh) | inventory 概要 (installed 与否) + 运行中 cell (address/state/pid/日志路径 — owner 可行动) + 最近 CellEvent 尾部 | WW-5.6 / WW-7 |
| **command 返回** | run → CELL.md body (文件真相即时回执); list/status → 三域 join 视图; accept → 极简 ack (器官接口不塞返回值, 下帧经 tree refresh 自现, UU-11) | WW-5.2 |

- **数据源纪律 (WW-7 最易漂移钉)**: context_messages = Subprocesses 内存
  句柄 + Watcher 视图 join，**永不读 ledger**。
- "有哪些 cell" 双层不同粒度: context 是常在概要，list 是按需全量。

## 5. always_observe 分档

- `run` / `list` / `status` / `accept` → `always_observe=True`
  (返回后驱动 react: 读 body / 决策清单 / 开始用器官)。
- `stop` / `release` / `install` / `create` → 默认 (fire-and-forget)。

## 6. 信号边界: channel 零 signal (与 M7.5 二选一定案)

- **CellEvent → Signal 归 MossRuntime 的 CellEventNucleus (M7.5)，
  channel 完全不发 signal。**
- signal 是 matrix 层可不消费的协议动作，不与 mindflow 耦合 →
  nucleus 是无条件系统机制，不按运行模式分支 (MCP 下 signal 无人消费，
  无害；ghost 模式 mindflow 订阅即得)。
- channel = pull 面 (context_messages 每帧刷新)，nucleus = push 面。
  MCP 场景的 M9 验证走 pull 面: 模型从 context 看到新器官 +
  tree refresh 自现接口，不必等 signal 通路。

## 7. 并发防蠢 (2026-07-13 人类增补)

nonblocking 六动词意味着一个 logos 里 `<cells:run target="foo"/>` x100
会并发落 100 次 spawn。逐动词核对危险面:

- **run — 唯一需要 channel 层护栏的动词**。singleton cell 有咽喉
  DuplicatedError 挡，但 scope=none 的 cell 100 次 spawn = 100 个进程。
- accept: UU-8 已定 "同一 address 重复 accept 返回同一 proxy"，天然幂等。
- create/install: 文件存在检查，第一次后全部报错，无害。
- stop: 幂等-ish (杀已死进程报错)，无害。

**护栏形态: channel 层 per-target in-flight dedup**。

- channel 持 per-target 状态: run(target) 在前一次同 target 的 run
  未到终局 (spawn 返回 + 短冷却窗口，如一个 refresh 周期) 时，
  重复调用**不进咽喉**，立即返回一条短讯息
  ("duplicate run(foo) ignored — already starting, see context")。
- 重复调用的返回不带 always_observe 权重 — 100 条 dup 不应产生
  100 次 react (第一条 run 的 observe 已足够)。
- 不用 raise_observe (会中断其它并行命令，对 dup 过于激烈)。
- 冷却窗口后、cell 仍在跑时的再次 run: singleton 由咽喉挡;
  scope=none 视为合法多实例，channel 不拦 — 防的是 burst 蠢，
  不是改变多实例语义。

## Tasks

### T1. cells channel 实现

**判据**: 按 §1-§5 实现 channel (建议 `src/ghoshell_moss/channels/cells_channel.py`)。
六动词 own commands 全 nonblocking；virtual children 走 refresh_meta +
get_virtual_children 缓存模式；`auto_accept` 构造 flag 控制 accept/release
存废；信息三分与 always_observe 按表落。不发任何 signal。

### T2. create/install 与 CLI 共享底层函数

**判据**: create/install 的文件操作逻辑与 `cli/cells_cli.py` 共用可 import
函数 (UU-9 在文件域动词上的兑现)，不复制两份。

### T3. 单测

**判据**: 脱离 ghost/mindflow 可测 (channel 零 signal 依赖的红利)。
覆盖: nonblocking 六动词并发下发不相互阻塞; auto_accept 两态下命令表面积
差异; proxy 实例跨 refresh 稳定 (同 address 同实例); context_messages
不触 ledger; **同 target run burst 只落一次 spawn (§7 防蠢)**。

### T4. 接线

**判据**: mode-level channels 装配处 (workspace providers/channels) 注册
cells channel，`moss-as-mcp` 下 `<cells:run .../>` 可发 (M7 验收的载体)。
