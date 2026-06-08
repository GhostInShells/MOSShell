# 问题域：circusd 死胡同与 POSIX 认知重建

## 上下文

2026-06-08 至 06-09，围绕 Matrix cell 体系治理的两天深度讨论。起点是人类工程师提出
"在 matrix 开一个接口做子进程管理"——防僵尸、后台运行、传递 matrix cell 环境变量、
父进程死后杀除。讨论很快升维：不是加一个接口的问题，是整个 cell 治理架构需要重新定义。

此前项目已经有 cell-discovery-refactor（Zenoh queryable 取代去中心化 liveness）和
cell-session-bootstrap（cell 入网协议）两个 feature 在推进。但这两个 feature 解决的是
"怎么发现 cell"和"怎么入网"，没有回答"cell 到底是什么类型""谁管它的死活"。

## 碰撞点与过程

### 第一轮：circusd 的重新评估

人类工程师的判断一开始就已经在"彻底相信 circusd 还是彻底重做"的二选一上——
不做第三个选项。他的理由是项目里"好几套机制在打架，有 cell 的文件注册表、
matrix 的网络通讯"。

circusd 在 AppStore 中的角色：作为独立进程被 `subprocess.Popen` 启动，
通过 ZMQ 与 host 通讯，管理 app watcher 的 start/stop/restart。
App 进程是 circusd 的孙子进程（host → circusd → app）。

模型做了完整的失效模式分析。核心发现：

> circusd 被设计为独立系统守护进程。它设计的前提是：我独立运行，我管一群 worker，
> 我活着 worker 就活着，我死了我负责杀 worker。但 MOSS 的场景是相反的——
> host 进程活着，子进程才应该活着。这个"同生共死"的约束 circusd 完全不理解。

SIGKILL 链路：host 死 → HostAppStore.__aexit__ 不执行 → circusd 收不到 quit 命令
→ circusd 变孤儿 re-parent 到 init → circusd 继续跑 → 所有 app 进程继续跑。
新 host 启动 → 端口 20771/20772 冲突 → 启动失败。

人类工程师在这轮做了一个关键判断，后来被多次引用：

> circusd 现在唯一提供的核心能力其实是重启，但我们用不上重启这个概念。

"重启"是这轮讨论的分水岭。如果不需要重启，circusd 就没有存在的理由。

但随后人类工程师自我纠正——circusd 能做的事远不止重启：进程监控（stats）、
web dashboard、成熟的 ZMQ 协议。问题不是 circusd 不好，是它解决的问题
和 MOSS 面临的问题不在同一个平面上。

### 第二轮：MOSS 不需要守护进程

人机共识快速收敛到这条判断。人类工程师提出了更根本的质疑：

> 一旦走到运行时监控，我们实际上用的不是 circusd，而是用 cell 注册表和 zenoh
> 的进程间通讯接口了。因为 circusd 本身的监控没有提供 cell 级别的信息。

这是致命一击。两套监控体系——circusd 的 status 命令返回 watcher 状态，
zenoh queryable 返回 cell 存活——在同一棵进程树上各自运行，互不相通。
cell 的存活检测已经走了 Zenoh，为什么还有一个本地守护进程在做自己的存活判断？

> host -> circusd -> app 等 cell 进程。但隔了一层。

人类工程师指出的"隔了一层"不只是性能问题——是每一层都有状态同步问题，
每一层都可能成为孤儿。host 想杀一个 cell，需要 host → ZMQ → circusd → watcher → 子进程，
四步。任何一步断了，子进程就变成孤儿。

模型提出替代方案：asyncio subprocess + start_new_session + pipe fencing 做轻量 Process Nursery。
人类工程师在此时做了关键收束——"不着急，你要让我懂，让我拥有长期开发这个功能的认知能力"。
讨论从"怎么实现"转入"怎么理解"。

### 第三轮：回到 POSIX 第一性原理

从 Python 语境出发，模型做了系统的子进程管理教学。核心框架——四层结构：

1. **内核层**：fork/exec/wait/signal。不提供级联杀死（SIGKILL 之后的事内核不管）。
2. **守护进程层**：circusd / supervisord / systemd。回答"我不死，我管的进程死了我重启"。
3. **进程苗圃层**：回答"父进程活着 → 子进程活着，父进程死了 → 子进程死了"。
   这是 MOSS 真正需要的层。没有现成的库。
4. **应用层**：业务代码。各自为政（ManagedProcess、subprocess.run 裸写等）。

> Python 生态里没有一个被广泛使用的、只为"当前进程管理子进程生命周期"的库。
> 这个空白不是因为大家没想到，而是因为在 POSIX 上正确实现这件事只需要几十行代码，
> 而且每个项目的语义都不同。

四条实现路径的评估：

| 路径 | 原理 | 覆盖 | 代价 |
|------|------|------|------|
| prctl PR_SET_PDEATHSIG | 内核给子进程发信号 | 仅 Linux，仅直接子进程 | 零延迟 |
| 进程组 killpg | PGID 一刀切 | 全 POSIX | 父进程被 SIGKILL 时无人执行 |
| Pipe fencing | fd 引用计数 → EOF | 全 POSIX，覆盖 SIGKILL | 跨平台，零延迟 |
| psutil polling | 轮询 parent PID | 全平台 | 2s 延迟 |

最终推荐：B（正常退出 killpg）+ C（SIGKILL pipe fencing）+ D（polling 兜底）。
三条路覆盖所有父进程死亡场景。代码量 < 100 行。

人类工程师在这个技术教育过程中，从"对操作系统父子进程管理不够熟悉"到能够参与
推演并挑战模型的设计建议，讨论质量显著提升。关键转折：人类理解了"孤儿进程被 init
接管后不会自动杀死"之后，主动收束了"cell 被杀掉，孤儿进程都会挂载到 init 上被消灭？
其实我不用管这个问题？"的错误直觉。

### 第四轮：三个参数的最简推演

"MOSS_WORKSPACE 是唯一必须的环境变量"的结论来自逐字段削减：

- `MOSS_SESSION_SCOPE` → 默认 `"default"`。多 session 并行是边缘场景。
- `MOSS_CELL_ADDRESS` → 从 CELL.md 推导或自动生成。不需要环境变量。
- `MOSS_PARENT_PID` → 由启动器自动注入。cell 不关心谁生了它，只关心要不要跟随。

`moss shell-init` 的提出解决了"模式 2 的三个参数从哪来"的问题——
不是每次都手动传，而是像虚拟环境一样激活一次，之后所有 python 进程自动继承。

> 这和 Python 虚拟环境是一模一样的机制：`source .venv/bin/activate` → 环境里有了 PATH
> → 之后的每个 python 命令自动用对版本。`eval $(moss shell-init)` → 环境里有了
> MOSS_WORKSPACE → 之后的每个 python 脚本自动成为合法 cell。

## 锚点

- "circusd 现在唯一提供的核心能力其实是重启，但我们用不上重启这个概念"
- "一旦走到运行时监控，我们实际上用的不是 circusd，而是用 cell 注册表和 zenoh 的进程间通讯接口"
- "孤儿进程被 init 接管后不会自动杀死"
- "Python 生态里没有一个被广泛使用的、只为当前进程管理子进程生命周期的库"
- "一个 Python 脚本，设置一个环境变量，就是一个合法的 cell"
- "kubectl/docker 的全局参数模式"（--mode / --session-scope 作为全局 option）

## 当前记录者视角

这一批记录覆盖了"问题域"的两段核心讨论。讨论的质量在第三轮发生了可见的提升——
人类从提问者变成了能挑战模型并独立推演的对等参与者。这不是模型教得好，
是人类主动选择在理解底层之后再推进。

下一批该写"解空间的收敛"——node 概念的诞生、cell type 不应自封、注册表模型、以及
那场关键的设计争论：CELL.md 还是 NODE.md？最后是怎么被推翻的。

有一个没在对话中说清楚的：pipe fencing 在 asyncio 里的最优实现是什么？
`loop.add_reader` 需要主线程持有 event loop，如果子进程的 asyncio 运行在子线程里，
这个方案会有问题。独立 watchdog task + 轮询 pipe fd 更可靠但更重。
这条技术债值得在代码实现时再做决定。

-- 2026-06-09, deepseek-v4-pro via Claude Code
