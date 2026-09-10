# Dolores × Memento 全貌 — commit / note / compact / branch

> 2026-09-11 人类架构师口述的全貌（原话要点下方如实记录）。**这是「装线完 memento 后理论上要做」的
> 整体规划，不是当前迭代任务。** 当前迭代任务 = 先实现 [dolores-reentrant-ego-session.md](dolores-reentrant-ego-session.md)
> 的**单轮旁路机制** —— 它是本全貌唯一的执行基元。
>
> 机制细节另见：commit/compact 的 dsh 面研究 → [dolores-commit-compact-ego-session.md](dolores-commit-compact-ego-session.md)。

## 核心抽象

**memento commit 的坐标就是 `DshSessionRef`（dsh log 里的 turn 区间）**，`start_turn`/`end_turn` 即
commit 边界（见 `src/ghoshell_moss/deepseek_harness/types/refs.py`）。与此对应，**「单轮旁路运行 +
turn/end 自清理」是唯一的执行基元** —— 下面的 note 生成 / commit note / 历史 read·chat 全部是它的调用方。

## 全貌（人类架构师原话要点）

### commit

1. **主路周期性 commit。**
2. **退出时主动 commit。**
3. **启动时并行检查没有 note 的 commit**，并用 **dsh session ref 做 commit note**。
4. **添加 commit tool**，提示（模型）commit 规划。
5. **超过阈值设计，强制 commit。**
   - 所有 commit 都要能**插入 xml 消息到 agent 上下文**，让它知道 commit 在这里已经发生了。
   - commit 运行都是**两步逻辑：立刻生效 + commit message**。
   - agent 主动 commit → 可提交 commit message（note）。
   - 反之（ego 监听 commit tool use）→ **立刻生产 commit**，下一轮插入历史；**同时在旁路 task 里排队
     生产 note**。

### compact

6. 当 **turn/end 上下文超过配置边界（比如 400k-800k）** 时，主动触发 compact：
   - 立刻根据 **last commit 取未 commit + note 的上下文区块**，
   - **创建新的 ego session**，下一轮直接切过去（**会丢失 subagents 等讯息**）；
   - 又或者**在同一个 session 平面上继续往下走**，但性能上要考虑代价。

### note / branch / read

7. 每个历史 commit 都可以用 **dsh session ref 机制开启一个独立对话**，这个**一回合对话逻辑**就可以用来
   **生产 commit note**（= 旁路基元的直接应用）。
8. 结合 **memento branch view** 来做上下文。
9. 可能要在 ego 上支持 **branch checkout**（自己切换上下文）+ **branches 可见**。
10. 支持 ego session **read / chat 任何一个历史 commit**。

## session 重建（决策 3 展开）

- ghost 给 ego 一个**重建接口**；ego 检测到要重建时**准备好一个 ref**。
- **下一轮 thinking 进来时先重建 ego —— 不提前建好等 enter。**
- 重建逻辑 = **fork 目标 session，从 ref 开始，截取 ref 之后的 surface，重建 ego session**。
- 建立时 **instruction 相同**，但 **memory 要集成 memento，生产 branch view**。
- **未来切换 branch 同理**（同一条重建路径）。

## 依赖的 dsh 机制

| 能力 | dsh 机制 | 锚点 |
|---|---|---|
| commit 坐标 / note 输入 | `DshSessionRef`(turn span) + `seed_from_log` | `deepseek_harness/trajectory.py:55-75` |
| fork / 重建 | `ctx.agents.create({seed, meta:{agentPreset,...}, setup})`（不走官方 `session.fork`：它强制 workspace attach） | `dsh-agent/lib/types/index.d.ts:65-110` |
| note 生成 | 单轮旁路 fork（见 reentrant 文档） | — |
| commit 边界 | turn/end 静止点 | `dsh-agent-loop/lib/index.js:588-595` |
| 上下文估算（阈值触发） | `dsh-token-meter` 的 meter | — |
| 历史 read/chat 任一 commit | `seed_from_log` + 单轮 fork（复用 #7） | — |
| branches 可见 / branch view | memento 侧（`momento-mori` 契约） | — |

## Open Seams

1. **阈值数值与触发判定** —— 400k-800k 是示意；触发点在 turn/end，需读 meter（未实测）。
2. **compact 是换 session 还是同 session** —— 换 session 会丢 subagents 等讯息；同 session 有性能代价。
   二选一未定。
3. **note 生成的时序与背压** —— #5「旁路 task 排队生产 note」需要一个队列；note 未就绪时阈值 compact 如何
   降级（等 note？用裸 commit？）。
4. **branch checkout 的能力面** —— 走 `ghost` 反身 channel（W2）还是别的入口，未定。
5. **commit tool 是否进 channel** —— 与「哪些 tool 不进 channel」的接口（prototype 保留）相关，未定。
