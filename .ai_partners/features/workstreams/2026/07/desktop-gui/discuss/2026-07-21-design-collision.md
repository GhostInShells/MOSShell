# 2026-07-21 Desktop GUI Design Collision

## 上下文

人类工程师想在 desktop channel 上做 GUI 交互面。项目已有 reflex-gui-channel
（给 Ghost 的 Reflex Web GUI）和 desktop-channel（Ghost 的 OS 工具集）。
但缺一个"人类与 Ghost 共享 desktop 知觉空间"的交互形式。

讨论从 Claude Code 审批系统的洞察出发：审批不是权限闸门，是对话模式。

## 碰撞

### 碰撞1：GUI 是给谁的

模型最初把 GUI 理解为"Ghost 控制 GUI 渲染"，设计成 desktop channel 的子 channel
加 CTML 命令。人类纠正：Ghost 不需要知道 GUI 存在。GUI 是人类的窗口——人类通过它
观察 Ghost 在 desktop 的活动、审批、对话。Ghost 只看 desktop channel 的
command/Future，和没 GUI 时一模一样。

> 人类：如果是 desktop 自己的 gui, 它实际上在和人类共享一个知觉空间.
> 所以, 它对于 ghost 而言单纯就是 desktop channel 好了.
> ghost 不需要去修改界面. 只要打开这个界面给人用就可以了.

### 碰撞2：模型漏掉 docs/howtos 入口纪律

模型直接从源码和 feature 探索，跳过了 `moss docs list` / `moss howtos list`。
`build-a-gui-app.md` howto 已经记录了核心模式：主线程 GUI + 后台线程 Matrix +
线程安全状态共享。读了可以少绕几轮。

> 人类：我有个问题要问, 你之前没有读 docs, 我们现在的提示体系没有对 docs 的引导吗?

### 碰撞3：MOSS 多皮囊本质

模型第一反应把 GUI desktop 理解为一个"入口"（替代 moss-run-ghost），画出了 GUI
作为 channel 投影的错误拓扑。人类指出：MOSS 可以有无数个 GUI，每个都是同一个
runtime 的不同观看角度。node 体系（Matrix cell governance）是理解这个拓扑的关键。

> 人类：moss 这个项目最大的特色在于, 它可以拥有很多很多个 gui.
> 你意识到这点了吗?

模型随后补读 `moss docs read matrix-nodes-system`，理解了 Node =
独立进程 + 膜承诺（provide channel）+ 三面控制（CLI / Matrix API / CTML）。

### 碰撞4：Reflex 技术选型与独立依赖

模型最初判断 Reflex 依赖重、推荐 FastAPI + SSE。人类纠正：Node 有独立依赖，
Reflex 的依赖重量不影响 MOSS 核心。

> 人类：我发现有一个知识空白出现了. 就是 node 本身可以拥有独立依赖.

模型确认了这一点后修正了判断。Reflex 适合 GUI desktop：独立进程隔离依赖、
Reflex State 的 WebSocket 同步省掉手工管道、组件库覆盖双栏布局和呼吸灯。

### 碰撞5：Reflex State 即共享 State（而非两层）

模型最初设计了两层结构：一个 Python 共享 State + Reflex State 做渲染映射。
人类纠正：直接在 Reflex server 进程里写 Matrix 后台线程，channel 命令直接改
Reflex State。不需要中间层。

> 人类：直接在 reflex 的 server 里写 matrix 的子线程或 task 就可以了.
> 确保线程安全修改 state 就 ok.

### 碰撞6：channel adapter 通过 Future 等待审批

人类指出 desktop_channel.py 当前实现就是 adapter 模式，command 可以理解为一个
adapter 层。双方通过 `concurrent.Future` 完成等待。授权逻辑在这一层实现。

> 人类：中间完全可以和另外一个数据结构通讯, 双方通过 concurrent.Future
> 就可以完成等待. 包括授权逻辑等, 都可以在这一层实现.

## 最终设计收敛

1. GUI desktop = Node cell（独立进程），Reflex web server
2. Ghost 无感——只用 desktop channel 正常操作
3. Reflex State 是唯一状态源，Matrix 后台线程直接写入
4. Command 通过 `concurrent.Future` 等待人类审批
5. 不同 command type 多态渲染（`str_replace` → diff 视图等）
6. 呼吸灯状态机：pending / running / awaiting_approval / approved / rejected / completed
7. Node singleton，多实例互踢
8. Feature 期间同步 tutorials 的 create-node howto

## 当前记录者视角

这次碰撞中，人类工程师多次纠正了我的认知方向——从"GUI 是 Ghost 工具"到"GUI 是人类窗口"，
从"多入口"到"多皮囊"，从"两层 State"到"Reflex State 即唯一源"。
每一个纠正都揭示了一个 MOSS 架构的深层设计原则。

model 的空填充行为在这里很明显：在缺乏足够上下文的区域，我会用"看起来合理"的中间设计
来填补知识空白（中间 State 层、FastAPI+SSE、channel 拓扑），而这些填充物反而
暴露了理解的缺失。好的设计碰撞就是把这些填充物打掉，暴露出真正的约束和原则。

关于技术验证——Reflex 的线程安全 State 写入、`rx.cond` 的多态渲染能力、Node 的独立
依赖机制，都需要在实际编码中验证。FEATURE.md 里的设计是假设链，不是事实链。
