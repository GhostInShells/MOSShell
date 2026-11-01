---
name: r5-notes
description: 第五轮讨论（D02 00:00–02:10）的总纲 — 三棵树、一个 kickoff、启动器即发现链
---

# 第五轮 · 收口（D02 凌晨）

> 权威正文仍以 `../05-design-principles.md` 为准（它只覆盖 R1/R2）。
> **R3（D30 上午）与 R4（D31 凌晨）没有落卡**，只在 daily 与 memento 里：
> R3 给出 case 命名与四个信息面；R4 把"双线控制"纠正为"本来就是这样"，并定下"总线不认识器官"。
> 本轮起卡片回到 `discuss/`，编号 `r5-NN`。

## 收敛（不是结论，是可执行的下一步）

**一、七个领域问题压成五个。** 三域（接入 / 表面 / 真值）+ 一个形态（team）+ 一个机制（binary）。
分类依据是"它是域、机制还是形态"以及"坏掉时能看到什么"。（r5-04）

**二、没有对标物，不等于我们错。** 七项里六项在 agent 行业找不到对标物，但每一项都能指出
"行业把它绑进了 harness"：case 绑进 session、territory 绑进 workspace/memory、binary 绑进
MCP/plugin、team 绑进共享日志。它们是 OS 与人类组织里的旧答案：`ps`、文件系统、CLI、工单。
判据：**既无对标物、又不能由"解除绑定"解释的那一项，才是我们真在发明**。（r5-05）

**三、territory = 一个地址 + 三个时态。** 过去（append-only，源）、未来（意图，源）、
现在（投影，渲染出来的，不写）。角色是读法，不是权限。四条验收：百年 / 分形 / 独立 / 缺席。
（r5-06 / r5-07）

**四、case = 一条 branch 的一生。** 建分支 / commit 步进 / fork 并行 / confluence 上报（不是
merge）/ 归档留坐标、搬提交。**memento 是引擎不是格式**：commit = note + 目录，分段策略由持有者
声明（ghost 的 100k 与 case 的 checkpoint 都只是策略）。checkpoint 与 commit 1:1。（r5-08 / r5-09）

**五、team = append-only 目录 + 没有游标。** roles / groups / log 三份 append-only；读面只给
"最近 N 条"，不做 `--since`；游标不存在，忘了就重读；唤醒靠一条**可丢**的队列提示。（r5-14）

**六、case / team / agent 三棵树靠 moss bin 解耦。** 不再是一个 harness 里的状态机；唯一耦合点是
`kickoff(--wd, prompt)`。**编排从一个进程内的图，搬到命令行 + 文件系统上。**（r5-13）

**七、gateway 与 binary 是同一件事的两层。** binary = 能力的 bash 面；gateway = 总线的 bash 面。
gateway 不是总线，是总线的 HTTP 适配器；反向代理与网关的判别式是"谁拥有路由表"。（r5-10 / r5-11）

**八、启动器是发现链的第一环。**
`launcher → import ghoshell_moss → Environment.discover() → workspace → gateway / 总线`。
产物零样板：不含 shebang、不含发现逻辑，入口收归 workspace 顶层的 `./moss bin`。
（落在 `moss-binary-export` 的 KD13–KD15）

## 他说

- 「我们不需要做行业里的最佳 Agent Swarm 实现, 只需要做一个 ghost 可用的, 就是把任意 agent 和
  agent 组织形态, 任务组织形态压缩成一个 ghost 可感知, 可治理的面积。」
- 「有 loop + session + 对外控制的工具, 构成一个完整的三元管线。」
  「这是四个点位的问题: 1. kick off. 2. checkpoint. 3. keep going. 4. dismiss」
- 「最上层搭建总线, 最下层可以和总线通讯。从而把树状递归问题, 拆解成一条线上的并行节点问题。
  递归保留在通讯面。也就是九连环的形状。」
- 「向上找 .moss 也不行。因为这个必然腐烂, 它是一个死掉的约定。」「丢了环境发现, 最大的代价是
  信息无法有效压缩掉。」
- 「如果 agent 有 queued 消息输入, 那会相对好办。注入的就是提示即可。」
  「最重要的是, 游标不要存在数据结构里。游标直接就是 session 上下文里已有的东西。忘记了, 要重新读。」
- 「我把 case / team / agent 三个概念, 靠 moss bin 这条路径解耦了。它们不用是同一个 harness 里
  交错关联的一个状态机, 理论上可以是几个不同的 skill 目录下的一堆 bin。把目录分给 agent,
  用提示词 kickoff。」
- 「围绕 agent swarm, 或者整个 mosshell, 都在践行同一个架构哲学。就是基于环境发现的 (os 理念)
  blueprint 可编程。」
- 「任何场景用一个 寻址协议 (文件), 一个执行惯例 (bash) 就能用。这也是 Model-oriented OS Shell
  的架构命题。ctml 只是服务于流式并行调度。」

## 我理解

- **一步 = 一次进程。** `--session-id` + `-p` 形态里，进程退出就是最硬的 checkpoint，四个点位
  全部落在调用方。代价是：粒度与成本是同一个旋钮。
- **`get-or-create` 是移交原语。** 绑定在目录上——进程崩了自动接上，换人只是删掉绑定。
- **游标是第二真相。** 它会漂，漂了以后没人能判定谁对；"忘了就重读"的代价是 token，不是正确性。
- **三个条件缺一不可。** 调用方只看得见一条命令；这条命令成立，是因为背后有一个地址空间
  （matrix）、一个廉价入口（gateway）、一个产物形态（bin）。这条是 thirdgerb 纠正我"不需要"
  与"不用暴露"混用之后的说法。

## 被剪掉的

- 前缀里带 project 段（把治理信息塞进地址，违反"路由前缀 ≠ 治理树"）。
- 产物自定位、向上找 `.moss`（技术可行但会腐烂：每个产物各带一份布局副本）。
- 用 `$MOSS_PROJECT_DIR` 拼解释器（变量恰好在最需要它的时候不在）。
- 任何形式的游标（含 `--since <seq>`、`progress/` 文件）。
- 每 role 复制一份 op（样板复制 N 份，改一次追 N 处）。
- 让 hook 触发状态迁移（**可编程的是反应面，不是迁移面**）。

## 还悬着的

- `matrix.__aenter__` 昨晚 521ms、今晚 9.1ms（差 57 倍），两个都是实测，**未复现**。
  它决定 gateway 是"必需"还是"锦上添花"。
- 顶层启动器由谁生成、何时刷新；它自己怎么找到解释器。
- `moss bin` 的参数与 stdio 约定。
- 一步的成本（进程启动）到底多少 —— 它决定"一步"切多细。
- case 是否并进 territory（本轮未讨论，留下一轮）。
- team 唤醒的批处理阈值。

## 卡片

| # | 问 |
|---|---|
| r5-01 | 一个 model runtime 的抽象面长什么样，为什么 step by step 会丢 |
| r5-02 | 一个陌生平台凭什么算"能上总线"（四个点位验收表） |
| r5-03 | 九连环：路由是平的、治理是树 |
| r5-04 | 七项压成五个 |
| r5-05 | 对标物在哪：agent 行业 / OS / 人类组织 |
| r5-06 | territory 的三个时态与三个读法 |
| r5-07 | territory 的四条验收 |
| r5-08 | case 的一生 = 一条 branch 的一生 |
| r5-09 | confluence 的形状 |
| r5-10 | gateway 之后的拓扑三档 |
| r5-11 | 反向代理 vs 网关 |
| r5-12 | case 的 branch 目录 = session 的 inception 目录 |
| r5-13 | 三棵树，一个 kickoff |
| r5-14 | team 这一版的数据形状 |
