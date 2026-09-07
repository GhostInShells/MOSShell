# Re-entrant Ego Session

> dolores perStep 第二阶段设计。2026-09-08 与人类架构师讨论收敛 + dsh plugin 调研。
> 目标：非当前 ego session（还原/旁路）可运行、可对话，但模型语境被锁、每轮可重入。
> 关联：`ghost-prototype-dolores` FEATURE.md；todo D5/perStep 锁 → 此机制。实施计划 2026-09-09。

## 意图

当前 perStep 锁把非当前 ego session **reject + 冻结**（`notifySessionFrozen`）。第二阶段把它升级为**可重入的旁路（bypass）**：

- 旁路 = 还原一个既有 ego session（fork 或直接用，agentPreset 都是 ego），模型**知道自己现在是旁路**，
  只能执行 回忆 / 对话 / 摘要 这类单帧任务。
- 旁路会话带一个 **mark**。每次运行把 mark 之后（上一次旁路对话产生的瞬态轮次）**删掉**，
  以 mark 为起点重新跑 —— 可重入，但**每一轮都是新的**。
- **模型看到的上下文必须被锁**：恒定 = `mark 以下（锚点历史）+ 当前这一轮`。
- 人类看到的 UI 不重要（append-origin transcript 永不塌缩，旁路轮次都活着、不消失）。

锚点/历史由 memento 体系组织（commit/segment），模型可 recap/read/agent 不同锚点 —— 这是 ego 与自己
人生锚点对话的机制。**本机制先实现，memento 之后接入**（anchor session 作发起方替代），两侧不互相阻塞。

## 关键事实

- 模型可见 surface（`deriveMessages`）**尊重 `SurfaceOp.replace`**：被遮蔽区间从派生历史删掉，
  `replaceGeneration` 触发缓存重建。这是 compact 同款机制 —— 锁模型上下文的正解。
- 人类 transcript 走 **append-origin**（`isAppendSurfaceEvent`），replace 不抹掉用户已见的轮次。
  所以只塌模型 surface，UI 天然保持所有轮次 —— 不是 reject 的"消失"。
- **历史 CTML 不是污染，是锚点的身体。** mark 以下的锚点历史（含历史 CTML 命令流）是旁路对话的
  主体上下文，**永不塌缩**；塌缩只碰 mark 以上的旁路瞬态轮次。

## 机制（plugin 侧）

一个**无状态、幂等、per-session 复用**的 fold，而不是 n 个控制器：

```
每个 ego-class 且带 mark 的 session，在其回合结束时：
  surface = agent.session.surface.nodes            # seq 数组（日志位置，append-only，不变）
  mark    = 该 session 存储的 mark seq             # durable，resume 后仍在
  若 tail seq > mark：                             # 有 mark 以上的瞬态轮次要塌
    append('user/message', 残渣节点,
           { surfaceOp: { op:'replace', start: mark后首节点seq, end: tail },
             sourceEventSeqs: [全部被遮蔽 seq] })
```

- 模型语境恒 = `锚点历史(≤mark) + 残渣 + 当前轮`。残渣**有界为 1 节点**：
  下一轮塌缩把上一轮残渣一并纳入 replace 区间，不累积。
- dsh 每 agent 串行，无竞争；n 个 session 只是同一段代码在 n 份独立状态上跑一遍。

## 调研：dsh plugin 可监听点位（已确认）

Agent 的 ctx 是 per-agent 扩展（`agent.ts` `ctx.extend({agent})`），dispatch 把 `agent` 融进 hook
信封（`dispatch.ts`），所以全局 `ctx.on('agent/<event>')` 覆盖所有 agent（含 resume 复活的历史 ego），
且能拿到 `agent.session`。身份识别复用 `doloresAgentPreset` + sessionId。

| 点位 | 语义 | 用途 |
|---|---|---|
| `agent/pre-step`（waterfall） | **每 step 一次**（`agent.ts:235`），现有 perStep 锁挂这 | ✗ 塌缩（per-step 粒度） |
| `agent/turn-stopping`（serial） | **每回合真正结束时一次**（`agent.ts:296`），本轮 assistant/tool 已全部 commit、`turn/end` 落日志前 | ✓ **塌缩挂点** |
| session 日志 turn/start·end / step/start·end | 可订阅 | 备选 / 观测 |
| `agent/status` / `agent/error` / `agent/inbox/*` | dispatch 事件 | 观测 |

塌缩动作在 turn-stopping handler 里做：`agent.session.append('user/message', 残渣, { surfaceOp: replace, ... })`。
surface 节点 seq 从 `agent.session.surface.nodes` 读。

## perStep 锁改造

pre-step 锁从「非当前 ego reject」改为：

- 当前 ego session：不变（背压等 thinking/enter）。
- 非当前 ego-class session：**不再 reject** → 放行，但**工具面提示旁路不可用**（只能用 dsh 自带，
  MOSS channel / ego tools 对旁路隐藏）；模型语境由 turn-stopping 塌缩保证。
- 无 mark 的 ego-class session：语义待定（默认视为需初始化，或 reject）—— open seam。

旁路对话**不走 thinking/enter/exit**（那是主 ego 的通道），走另一个**单次请求接口**，带要插入的
instruction（mark 可与之绑定；缺省插默认 instruction）。UI 也可能访问 ego。

## Open Seams（明日实施前钉死）

1. **塌缩边界**：turn-stopping（本轮内容 commit 后）为推荐挂点。是否仍需"运行前 pre-step 塌缩"以
   保证模型从不见残渣 —— 取决于残渣节点载荷，二选一。
2. **残渣节点载荷**：replace 必须垫一个真实节点、不能垫空。载荷应极小且惰性（"此处为已归档旁路
   过往"），或被下一轮首条消息吸收。**不得**承载/摘要历史 CTML（那是 mark 以下的内容，不塌）。
3. **mark 的家**：log-only `session/mark` 事件记 `{seq}`，存 append-only log，resume/重放天然继承。
4. **旁路入口接口契约**：单次请求接口的参数（sessionId / mark / instruction）设计为纯函数，memento
   之后只负责调用。CTML 泄漏风险：旁路无 MOSS executor，mark 绑定的 instruction 若重放历史
   CTML-first 协议会让模型产出无处落地的 CTML —— 需要旁路覆盖层。
5. **工具面降级**：ego tools 对旁路"不可用"是提示/剪裁层，不是注册 reject。失败形态要干净。
