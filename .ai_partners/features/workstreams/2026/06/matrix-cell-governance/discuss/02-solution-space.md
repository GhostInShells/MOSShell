# 解空间：极简推演与 Node 的诞生

## 上下文

前期讨论（01-problem-space）收敛了两个结论：circusd 不适合 per-session 的进程图，
以及 POSIX 子进程管理的四条路（killpg + pipe fencing + polling 三件套）。
这一批讨论从"cell 怎么启动"出发，最终撞出了 node 概念的形成。

## 碰撞点与过程

### 第一轮：三种启动模式的本体论差异

人类工程师对 MOSS 当前的启动机制做了一个精准的切割——实际上存在三种截然不同的启动方式：

1. CLI → subprocess → Matrix.discover()。两个进程。CLI 负责把参数编码成环境变量，子进程解码。
2. 纯 Python 脚本 → Matrix.discover()。一个进程。但需要"知道自己是谁"。
3. 已并网的 cell → spawn → 新 cell。一个启动另一个。

人类工程师的判断：**模式 2 分层最干净——脚本即进程，进程即 cell。没有中间人。**
但前提是 Matrix.discover() 不需要猜——它的三个参数必须已被设置好。谁来设置？

模型提出 `moss shell-init`：

```bash
eval $(moss shell-init)
python my_cell.py   # Matrix.discover() 从 env 拿到所有参数，零配置
```

这个命令不启动任何东西——它只输出 shell 脚本。它的参数简单：

```
moss shell-init [--mode default] [--scope dev] [--ghost xxx]
```

> 这和 Python 虚拟环境是一模一样的机制。

人类工程师敏锐地抓住了这个类比——venv 的两种用法（显式指定 vs 环境激活）和 cell 的
两种启动方式完全对应。这个类比成为后续讨论的共享锚点。

### 第二轮：环境变量三个变一个

从这里开始，讨论进入了"拆到极简再重建"的节奏。逐一审视三个环境变量：

**MOSS_WORKSPACE — 必须保留。** 没有 workspace，cell 不知道 zenoh config 在哪、providers 在哪。不可替代。

**MOSS_SESSION_SCOPE — 可以默认。** scope 的语义是"同一个 workspace 下多组 cell 彼此隔离"。
但在 MOSS 的分布式组网架构里，一台机器上跑两个独立 session 是边缘场景。
默认 `"default"`，需要时显式传。

**MOSS_CELL_ADDRESS — 可以推导。** 如果 CELL.md 存在，`app/{group}/{name}` 直接得出。
如果不存在，cell 自己生成一个默认值。

最终结论：`export MOSS_WORKSPACE=/path/to/.moss_ws && python my_cell.py` ——这就是一个合法 cell。

人类工程师此时提出了重要的限制条件：

> 前提是 workspace 里的 provider 全部是轻量的。

当前 providers 的加载已经是按模式声明的（Mode.manifest），不是全局加载。
这个前提已经成立。

但讨论随即转向了一个更深的质疑——"孤立的幻觉"。

### 第三轮：极简 cell 能做什么——以及不能做什么

"一个环境变量就能启动的 cell"在技术上成立，但它回答的是通信层的正确性——如何连上网。
它不回答连上网之后有什么用。人类工程师指出了三个不可简化的重域：

> ghost / 感知 / 控制模块实际上不行。对于一个 ghost 而言，拆分 workspace 鼓励化没有意义。
> cells 仅仅是一种组网补充手段，而不是 moss 本身。

这轮推演的关键在于区分**集成深度**，而非更多类型：

> 一个核（nucleus）不是"一个提供 channel 的 cell"。它是 ghost 认知架构的一部分。
> 它们之间的耦合不是技术上的 import 依赖，而是语义上的。

四个集成深度的自然分层：
- script — 零耦合。一次性数据处理。
- bridge — 协议耦合。MQTT 桥接、ROS bridge。
- app — 资源耦合。camera capture、TTS。
- nuclei — 语义耦合。共享 ghost 的 conversation、memory、identity。

script 可以完全独立。bridge 需要理解 topic 协议。app 需要理解共享资源。
nuclei 需要理解 ghost 本身。MOSS 的核是 ghost + body + shell 的有机整体，
cell 是这个整体在边缘的延伸。

### 第四轮：CELL.md 还是 NODE.md —— 一场被推翻的争论

此时讨论开始收敛到"注册表该是什么格式"。模型最初倾向继续用 CELL.md 作为统一格式，
type 字段区分权限。

人类工程师推翻了它：

> 按我们这个逻辑划分，cell 自己的元配置文件没有资格定义自己的类型。
> 除非所有的命令都是用 cell 启动，像 ros2 那样。

这是关键一击。CELL.md 不应该自封类型——类型是启动者在启动时赋予的。
同样的 Python 脚本，`moss nodes run` 启动就是 node，MODE.md bringup 启动就是 app，
`host.new()` 启动就是 host 本身。**脚本什么都没变，变的是"谁启动它"和"赋予它什么权限"。**

这和 ROS2 的 `ros2 run` vs `ros2 launch` 的区分完全一致——同一个 node 二进制，
在 launch file 里被赋予了不同的上下文。

此后模型纠正了自己的方向——之前说"CELL.md 取代 node，node 可以定义 type"是无意义的重复劳动。
四种 cell type 不是同质到可以归一，而是两种消失、两种保留：

> MODE.md → host（session 配置，不可动）
> FRACTAL → fractal（网络拓扑配置，不可动）
> NODE.md → app + node（worker cell，归一）

app 不是一个配置类型，它是 node 在 host 管理下的运行时状态。

### 第五轮：app 的独特性——重新承认

人类工程师又做了一个逆转。当模型开始推动"node 完全取代 app"时，他拉回来：

> 如果 app 是面向模型运行时迭代、自主管理的，则 host bringup 它、动态开启它、关闭它、
> 以及 respawn，好像仍然完全成立。

这不是反对归一。这是承认 app 有 node 不具备的额外需求——模型的自主管理。
一个纯外部的 node（OS 或 supervisor 管的），模型对它没有生命周期控制权。
一个 app（ghost 管的），模型可以决定它的生死。

因此最终的分类不是"node 取代 app"，而是 node 作为基础设施，app 在 node 之上。
app 的 respawn 不是进程级概念，是 ghost 的决策——ghost 观察到 app 死了，
决定要不要重新启动它。

这也回答了一个看似技术实则哲学的问题：app 的"状态"（starting/started/stopped/error）
属于谁？不属于 cell 自己，不属于发现层——**属于 host 对它所管理 node 的判断。**

### 第六轮：命名与迭代边界

命名讨论。`script` → `node`。人类工程师对命名的要求是："表示由外部决策运行的节点，
连节点编排我们都不做。"

模型的提案及淘汰：
- `daemon` / `service` / `worker` — 都太窄或太 OS 特定
- **`node`** — 分布式系统通用语。不承诺长运行、短运行、被管、自己管。
  它只说一件事：我是这个网络里的一个点。

MOSS 只在一个地方做编排：**app**。因为 app 是 ghost 的延伸，ghost 对它负责。
其他的 node，MOSS 提供身份（环境变量），外部提供生命周期（OS/supervisor/人的终端）。

最终`CellType`收缩为三个：`host` / `node` / `fractal`。`app` 降为 node 的运行时角色，
`script` 消失。

关于迭代路径，人类工程师做了明确的边界约束：

> circusd 的取代不是 beta 版本的任务。它是 0.1.0 时要考虑的问题。
> 做它只会让我继续被模型批评防御式研发。

不碰 apps，先建 node 并行线。验证完毕再讨论取代。

## 锚点

- "从分层上最舒服的其实是 2"（纯 Python 脚本启动，指模式 2）
- "moss 如果是 python 命令，它永远是从 CWD 去找。但如果 moss 是一个 workspace 里的命令，它可以软链到任何地方不用做环境发现。"
- "cell 自己的元配置文件没有资格定义自己的类型"
- "脚本什么都没变，变的是谁启动它和赋予它什么权限"
- "App 不是一个配置类型，它是 node 在 host 管理下的运行时状态"
- "命名要表示由外部决策运行的节点，连节点编排我们都不做" → node

## 当前记录者视角

这一批讨论的高质量来自一个反直觉的模式：人类工程师反复先同意再推翻模型。
模型推到"极简一个变量"，人类说"但 ghost/感知/控制无法拆分"。
模型推到"CELL.md 统一"，人类说"cell 不能自封类型"。
模型推到"node 取代 app"，人类说"app 有模型自主管理的特殊需求"。
他不是在挑剔实现方案——他是在搭建他自己大脑里那台 node 的虚拟机，
每次反驳都是虚拟机在试运行。

下一批该写的是最有争议的部分：fractal 到底是不是 cell。以及最后的两轴状态模型收敛。

-- 2026-06-09, deepseek-v4-pro via Claude Code
