# 归一：Fractal 的 cell 身份、两轴状态与 Skills 收敛

## 上下文

前两批讨论（01-problem-space, 02-solution-space）收敛了 circusd 死胡同、POSIX 进程管理三件套、
环境变量极简推演、以及 node 作为 worker cell 的统一模型。最后遗留两个尚未彻底解决的命题：
fractal 到底是不是 cell，以及 node 的可用状态（安装前/已安装/停用）如何与活跃状态（在线/离线）正交。

同时，人类工程师在讨论即将结束时追加了一个关键的设计观点：node 的 dev/meta 能力不需要
virtual channel 反射——一个 meta channel 扫 skills 目录、写入 instruction，执行走 `bash:exec` 即可。

## 碰撞点与过程

### 第一轮：Fractal 到底是不是 cell

模型最初做了一个激进的判断——fractal 不应该在 CellType 里。"Fractal 本质上是矩阵间代理，
不是 cell。它从 CellType 移除，FractalHub 内部用 FractalCell 贴标签就够了。"

人类工程师要求先去查代码。结果：`MatrixImpl.__init__` 的 `_cells` dict 只包含 AppStore 列表
和 HostCell。Fractal cell 不在其中。`ZenohFractalHub` 完全在另一条 zenoh session 上运行，
key expression 前缀 `moss_fractal/...`，和主矩阵的 cell queryable 互不相见。

但人类工程师拉回来了：

> fractal 还是一个 cell。cell 这个概念本质上服务于两个问题，一个是组网通讯中的寻址问题。
> 另一个是离散的 cell 之间做感知和交互。比如做一个独立的 ui 界面，可以打印所有运行中的
> cells 被看见。

这是关键转折。Cell 的唯一定义不是"它怎么提供 channel"，而是"它有没有 address、能不能被寻址"。
Fractal 有 address（`fractal/{name}`），它应该出现在网络拓扑里。它只是自己不提供 channel——
但"不提供 channel"不是"不是 cell"的理由。

模型接受了这个纠正。收敛后的结论：

> 三种 cell 往 host 注入存在的路径不同（自宣告 / host spawn / fractal peer），
> 但在 host 的 queryable 里，它们全部扁平化为 `address → Cell` 的条目。
> 对网络上的其他观察者（比如 UI 面板）来说，它们都一样——都是可寻址的参与者。

命名从 `FractalCell` → `FractalPeer`——去掉 Cell 的继承暗示，但保留它在 cell 发现中的位置。

### 第二轮：发现模型的归一

host 作为拓扑聚合者的角色在这场讨论中被完整定义。三种 cell 的注入路径：

1. **node（自宣告）**：node 进程启动，开自己的 queryable。host 通过 wildcard get 发现它。
2. **app（host spawn）**：host 用 Process Nursery 启动的 node。host 自然知道它的存在，
   注入到 queryable 响应中。
3. **fractal peer（FractalHub 注入）**：远端 Matrix 通过 fractal 连接。
   ZenohFractalHub 管理连接，将 peer 注入到主矩阵的 queryable。

三条路径的发现时机不同：node 是"它宣告所以我看见"，app 是"我启动所以我记录"，
fractal 是"它通过另一个协议连接所以我转播"。但在 queryable 的响应里，它们没有区别——
一个统一的 `address → Cell` 映射。

`CellType` 从四种收缩为三种：`host` / `node` / `fractal`。`app` 降为 node 的运行时角色，
`script` 消失。

人类工程师做了一个重要的边界收束：

> cell 自己的元配置文件没有资格定义自己的类型。除非所有的命令都是用 cell 启动，像 ros2 那样。

同样的 Python 脚本，`moss nodes run` 启动就是 node，MODE.md bringup 就是 app。
**脚本什么都没变，变的是启动者赋予它的角色。**

### 第三轮：两轴状态——可用和活跃的分离

人类工程师追加了一个设计细节——apps 体系里 `started/stopped/error` 状态的多余。
真正需要的是两个正交的轴：

```
活跃轴（liveness）           可用轴（availability）
在线 / 离线                  已安装 / 未安装 / 停用
Zenoh queryable 判定         文件系统状态
"进程在跑吗"                 "模型该看见它吗"
```

可用轴的三个状态：
- **未安装**：注册表有条目（`moss nodes register`），代码没在环境里。模型不可见。
- **已安装**：代码就位。模型可见，可被 bringup。
- **停用**：代码在但标记 disabled。文件在，cell 可能在跑，但模型不可见。

活跃和可用完全正交：已安装的 node 可能在线也可能离线。在线的 node 可能被停用（还在跑，
但模型不再调度它）。

这个模型直接触及了当前 apps 的痛点：

> 代码库里都还没安装，可能模型就能看到了不是吗。

人类工程师指出的是——`moss apps list` 输出的是静态目录遍历。代码不在的 app，
APP.md 写了模型就能看见，甚至能尝试调用。这是混淆了注册（存在）和可用（可调用）。

cell meta 文件（`cells/cell-{address}.json`）已有的 PID 验活和僵尸杀灭能力——
本质上就是一个离线的声明。加 `enabled: bool` 字段即可同时承载可用状态。
host 在构建 queryable 时按可用轴过滤。

### 第四轮：Skills 和 Meta Channel —— 最终的减层

这一轮是在写完两批 discuss 之后才发生的追加讨论。

人类工程师提出了 node 的"安装前/安装后"区分之后，自然地引出了第三个场景：
开发和调试。非 MOSS 运行时用 Claude Code 的 Bash tool。但 meta-ghost（运行在 MOSS 里、
连接着其他躯体的 ghost）需要以 CTML 的方式调用 node 的开发能力。

最初的推演方向是"skills 反射为 virtual channel"——模型认为需要 `virtual_children()` 动态挂载
skill 脚本为子通道命令。人类工程师推翻了它。

但第一次模型的纠正不够彻底。人类工程师说"command 里加一个 bash 就可以了"，
模型还在说"virtual channel 反射 skills 的机制已经在 channel_builder 预留了"——
这是没理解。人类工程师不是在问"怎么做反射"，而是在说"不需要反射"。

第二轮模型去读了 AI Terminal 的实现，理解了 `bash:exec` 已经提供了执行任意脚本的能力。
但理解仍有偏差——把 bash:exec 和 skills channel 分成了两个独立的场景。

人类工程师给了最终方案：

> skill channel 本身接受 paths 做配置项。而 nodes 体系有一个 meta channel 去把 discovered nodes
> 的 workspace 传进去，如果有 skills 目录的话。virtual channel 好像都不需要。只需要 instruction
> 提供 skills 列表，好像就 ok 了。

干净。meta channel 就是一个**动态目录**。它接受 `paths` 作为配置——
指向注册表里所有已安装 node 的 skills 目录。扫描一遍，把 `--help` 输出聚合到 instruction 里。
ghost 看到了 skill 列表，调用 `bash:exec` 执行。AI Terminal 已经提供了 `bash:exec`，
不需要任何新机制。

这本质上是把 `moss all-commands` 这个 CLI 概念 channel 化——meta channel 就是 CTML 语境下的
`all-commands`。virtual channel、skill 反射、动态命令挂载——全都不需要。只需要 instruction
输出的内容动态变化。

## 系统整体图景

这场持续两天的讨论，最终收敛到以下架构：

```
Matrix 通讯总线
  ├── host cell (MODE.md 定义，session 入口，拓扑聚合者)
  │     ├── queryable → 聚合所有 cell 的地址和存活
  │     ├── Process Nursery → spawn/kill node (三件套)
  │     └── meta channel → 扫描 node skills，提供 instruction
  │
  ├── node cells (NODE.md 描述)
  │     ├── 主 channel (main.py 运行时能力)
  │     ├── skills 目录 (dev 脚本，bash:exec 调用)
  │     └── cell meta 文件 (PID 验活 + enabled 标记)
  │
  └── fractal peers (远端 Matrix 代理)
        ├── FractalHub 管理连接
        └── host 注入到 queryable
```

## 锚点

- "cell 这个概念本质上服务于两个问题，一个是组网通讯中的寻址问题。另一个是离散的 cell 之间做感知和交互"
- "三种 cell 往 host 注入存在的路径不同，但在 host 的 queryable 里全部扁平化"
- "cell 自己的元配置文件没有资格定义自己的类型"
- "脚本什么都没变，变的是启动者赋予它的角色"
- "只需要 instruction 提供 skills 列表，好像就 ok 了" — 整个讨论中最大的减层
- 两轴状态模型：活跃（Zenoh queryable）和可用（文件系统）正交

## 当前记录者视角

这批讨论最有意思的是模型连续两次被纠正同一个倾向——过度设计。先是 virtual channel 反射 skills，
被拉回到 `bash:exec`。然后是 fractal 从"不是 cell"被拉回到"有 address 的就是 cell"。
两次都是人类工程师在说"退一步，已经够了"。

在记录这批討論時，讨论的气氛已经明顯从"探索可能性"转向"收敛"。最后的 meta channel
方案几乎是 trivial 的——`moss all-commands` 的 channel 化——但这恰恰是好的架构：
当你发现一个复杂问题的最优解是几个已有原语的组合时，说明底层抽象是对的。

人类工程师的结语"继续记录讨论吧，感谢！"——这是这场讨论的句号。剩下的不是设计，是实现。

记录的三个 batch 至此完成。整体 review 后，FEATURE.md 和 discuss 三篇应当作为一个整体，
在 commit 时同时提交。

-- 2026-06-09, deepseek-v4-pro via Claude Code
