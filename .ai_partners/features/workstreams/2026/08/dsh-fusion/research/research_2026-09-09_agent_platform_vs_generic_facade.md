# dsh agent 平台面 vs 通用 facade 边界

> 认知快照, **非结论**。2026-09-09 与人类架构师讨论, 目的: 下次进来开发"agent 平台面"任务时
> 能还原当下部分认知, 不用从头说一遍。这里的判断随时可被推翻, 不是定案。
> 关联: dsh-fusion FEATURE.md 路径 A; 与 memento agent 的兴废强相关。

## 任务本体

从 `dsh connection` 长出一个**独立的、可界面的 agents 平面** —— 即 dsh session 的 **agent call
protocol**。这样不只是 dolores, **任何 moss matrix node 都能把 dsh connection 当一个 agent 平台驱动**。

这对应 dsh-fusion FEATURE.md 的「路径 A: gui 管理的 agent」, 但那个"独立 feature"**至今没立项**,
只作为两路径之一被提了一句。

## 背景: memento agent 兴废

- **memento agent 可能会废弃**(人类无精力维护, 保留部分技术实现)。
- memento agent 的**通用 facade 是"实现错误的地方"** —— 负责实现的 model(比当前早一个版本的
  deepseek)搞了**父子目录交叉耦合**(通用 facade 绑死在 memento 具体实现下)。
- **教训的精确表述**(人类纠正过模型的归因): 翻车不是"自顶向下先验设计 facade"这个方法错,
  而是**"模型独立做这个任务"这件事错**。所以通用 facade 不是"不能先验设计", 是"不能由模型
  独立先验设计"——它需要人类与模型对齐着推。
- 因此现在**倒过来用 dsh session 反推通用 facade** —— 这不是唯一正解, 是当下最稳妥的路径。

## 两个极端方案(同时有效, 不是二选一)

| 方案 | 形态 | 特征 |
|---|---|---|
| 1. sandbox | 允许模型直接 eval 这个 dsh connection, 一种 node | 危险但灵活 |
| 2. 结构化契约 | 模型经 channel 体系实现自己管理的 agents 体系(有状态、有身份、异步, **非 subagent/一次性**) | 安全但重 |

两者恰好映射 dsh 的两个调用面(见 `research_2026-08-20_dsh_agent_api_surface_and_timing.md` 第一节):

- 方案 1 = plugin **in-process 特权面**(直接 eval `ctx.agents.create` 等)。
- 方案 2 = **http rpc 外侧**(8 verb 结构化驱动)。

所以"两个都做"成立——它们是 dsh 两层的 MOSS 化暴露, 都属 dsh 专属的具体面, 可在一个 feature 里共处。

## 当前判断(非结论): 拆两层

**"dsh 作为 agent 平台"的具体面 → 算 dsh-fusion。** 8 verb 薄 facade + 特权桥 + node eval /
channel 契约实现, 回答"怎么驱动一个 dsh session", 是 dsh 专属的。

**"通用 agent facade"(有状态/有身份/异步的抽象)→ 不算 dsh-fusion。** 它是跨实现的上层抽象,
应独立成 feature。理由正是 memento 的教训: 通用 facade 绑死在具体实现下 = 父子交叉耦合。

```
dsh-fusion（交付 dsh 专属的具体面）
  └─ 8 verb 薄 facade + 特权桥 + node eval / channel 契约实现
        │  未来 adapter
        ▼
通用 agent facade（独立 feature, 未来, 需人类对齐着设计）
```

## 动词一致、数据面后行

dsh-fusion 里**只立动词协议、不碰通用数据结构**。依据:

- dsh 的动词面是稳定的。`followup / steer / inject` 本质是同一个 `send` 的三个特例, 差别只有
  `target`(next-turn vs next-step)和 `wakeup`(boolean)。
- 会变的是**数据结构**(`UserMessage` vs Claude/OpenAI message; `SessionEvent` vs `AgentResult`)。
  **memento 父子交叉耦合的恰恰是数据面(出入返回值), 不是动词。**

所以现在 dsh-fusion 按 dsh 原生动词走、数据面保持 dsh 原生 pydantic, 不做"通用化预演"。未来
通用 facade 出现时, dsh 侧只需一个 adapter 做数据映射, 动词不动。

**留意**: "动词一致" ≠ "照搬 dsh 的 8 verb 名字当通用契约"。`followup/steer/inject` 未来在通用面
可能收敛成一个 `send(target, wakeup)`, 三个动词是特例。**这个收敛是未来 adapter 时的决定, 现在不做。**

## 待下次进来钉死的点

1. 路径 A 是否立项成独立 FEATURE, 还是并入 dsh-fusion 作为"具体面"小节。
2. 通用 facade 的数据面(出入返回值结构)何时、由谁对齐着设计 —— 这是最难最易耦合的部分, 明确
   留到"有人类对齐"的阶段。
3. 方案 1(sandbox eval)与方案 2(channel 契约)的先后/取舍 —— 当前判断是"同时有效", 未排优先级。
4. 与 `agent-surface`(draft feature)的分工: 那个是 memento agent 的通用 facade(create/__call__/
   context/4 控制函数, 自然语言契约); 这里要的是 dsh 专属、保协议的 agent 平面, 8 verb 保留。
   两者不该混, 但 memento 废弃后 agent-surface 的去留会反过来影响这个边界。
