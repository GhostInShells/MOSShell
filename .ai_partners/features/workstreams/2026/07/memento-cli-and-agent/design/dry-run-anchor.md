# Memento Agent 下一步施工：dry run → anchor dump → anchor + dry run

> 索引自 [FEATURE.md](../FEATURE.md) §17。本子文档承载 v1 之后的施工计划与
> 进展记录——主文档已 16 节太长，拆到这里。设计定案在 `agent-surface`
> workstream §2.8（四控制函数 dump_anchor / reply / dry_run / self_explain）。

## 0. 定位

v1（FEATURE.md §1-§16）已收尾：invoke → 最终文本，staging 累积不 commit。
下一步打磨**协议面**——把 memento agent 从「invoke → str」升级为可 dry run、
可 anchor dump、可 anchor 续跑的表面，让第三方项目能依赖 moss 做产品化。

关键前提变更（2026-08）：**pydantic-ai-slim 已确认为关键依赖**。此前没决定
是否用 pydantic-ai，最近才知道 slim 版，才让它适合作为关键依赖存在。这使
pydantic-ai 从「可选 [ghost] extra」升级为 agent 家族的默认底座——dry run
与 anchor 都建立在它的消息协议上。

**依赖纪律**：pydantic-ai 仍不是全局默认依赖，别泄漏到全局。所有 pydantic-ai
耦合收进 `agents/pydantic_ai_utils/`，惰性 import（照 llms client.py 模式）。

## 1. 地基：agents/pydantic_ai_utils/

**目标**：把 pydantic-ai 的重复方法从 llm funcs 倒过来提炼，集中到
`agents/pydantic_ai_utils/`，memento-agent 与 llm funcs 共用。历史代码（ghost
Atom 里的一套）**不动**——当初没决定用 pydantic-ai 留下的并存，新代码才收敛。

第一批收敛物：

- 消息序列化 `serialize_messages` / `deserialize_messages`
  （`ModelMessagesTypeAdapter` dump/validate）——现散在
  `llms/pydantic_ai_adapter/funcs.py`，锚的 `CallAnchor.turns` 与 memento 的
  dry run `messages` 共用同一条链。

依赖门控（照 `llms/pydantic_ai_adapter/client.py`）：

- 模块级零 pydantic-ai import，函数体内惰性 import
- `architecture.py` 能安全 import 它，不触发 pydantic-ai

**依赖方向（已定）**：放 `agents/pydantic_ai_utils/`，llms 反向 import
（llms → agents）。用户拍板接受这一方向。

产出判据：llms funcs 的 `CallAnchor.turns` 改用 `agents.pydantic_ai_utils`
序列化，memento 侧 import 同一链；`import ghoshell_moss.agents.pydantic_ai_utils`
不触发 pydantic-ai。

## 2. 三步走

| 步 | 内容 | 产出判据 |
|---|---|---|
| 1 | dry run | `-j` 出 tool_calls + messages + usage；零副作用、不进历史 |
| 2 | anchor dump | 复用 CallAnchor turns 序列化，一次交互冻成锚 |
| 3 | anchor + dry run | dry run 产物作请求帧，reply 还原续跑 |

每步做完停下、明说「步 N 完成」，等人类 review 放行再进下一步。

### 步 1：dry run

**语义（钉死，统一 agent-surface §2.8 与协议面讨论）**：

- 纯探针：不进历史（不写 memento）
- 零副作用：停在工具调用位置，不执行 sandbox_exec
- 暴露 model response：tool_calls + messages + usage 序列化，支持 `-j --json`

**实现**：`requires_approval` 路线（parity 原则——dry run 与 run 共用同一条
instruction 组装 / 工具 schema / model 请求，看到的 code 就是会执行的 code）。
pydantic-ai 的 `requires_approval=True` 工具会让 `Agent.run` 原生停在工具调用，
产出 `DeferredToolRequests`（approvals = list[ToolCallPart]），零副作用。

**数据结构**（照 LLMFuncs 三板斧：强类型 BaseModel + usage 保真 +
ModelMessagesTypeAdapter 序列化）：

```python
class InvocationRecord(BaseModel):   # 弱数据形态，model_dump() 即 -j
    output: str                      # 最终答案（run 态有，dry-run 态空）
    content: str                     # 所有 TextPart 拼接
    usage: dict                      # token 开销
    cast: float                      # 耗时
    tool_calls: list[dict]           # 未执行的 ToolCallPart —— 干燥态核心
    messages: list[dict]             # 完整轨迹（ModelMessagesTypeAdapter 序列化）
```

「未执行」在消息轨迹里自描述：ToolCallPart 后面没有 ToolReturnPart 就是干燥态，
不需要额外状态字段。

### 步 2：anchor dump

复用 `CallAnchor`（llms/pydantic_ai_adapter/call_anchor.py）的 `turns` 序列化，
把一次交互冻成锚。数据走步 1 提炼出的 `agents/pydantic_ai_utils` 序列化链，
锚文件与 memento 轨迹序列化格式天然一致，可互读。

### 步 3：anchor + dry run

dry run 的产物作请求帧（未执行态），`reply(anchor, prompt)` 还原 turn 链续跑。
对应 agent-surface §2.8 的 `reply(anchor, prompt) -> AgentResult`。

## 3. 架构地图

三步走完，`architecture.py` 补三块（llms 已进 client/funcs，补 agents/tools）：

- tools：fs / git / codex / moss / decorators（@cli）
- agents：contract / pydantic_ai_utils
- llms：补 pydantic_ai_utils 或转换面

关键：pydantic_ai_utils 进地图那行必须验证 import 不触发 pydantic-ai。

## 4. 施工进展

<!-- 每步完成追加一行，不 fine-grained checklist。 -->

- 2026-08-13：拆出本子文档，三步走 + 地基定案。
- 2026-08-13：步 0 完成。`agents/pydantic_ai_utils/` 建立，序列化链从 llms funcs
  提炼共用（`serialize_messages`/`deserialize_messages`），import 门控验证通过
  （不触发 pydantic-ai），llms 41 tests 全过。
- 2026-08-13：步 1 dry run 完成。`InvocationRecord` 落 contract，requires_approval
  工具面（factory 双 agent），`impl.dry_run` + CLI `dry-run` 命令（-j 出
  tool_calls+messages+usage）。冒烟验证：模型停在 sandbox_exec 调用、工具未执行
  （usage.tool_calls=0）、零副作用、不进历史。
- 2026-08-13：步 2 anchor dump 完成。**方向修正：放弃通用 agent 层，诚实做
  pydantic-ai agent anchor** —— `PydanticAIAgentAnchor` 落
  `agents/pydantic_ai_utils/anchor.py`（不再伪装成通用 `AgentAnchor`）。payload =
  instruction + tools 协议 + model_name/thinking + turns（认知流一帧，价值是
  review 不是回放）。`dump_anchor` + CLI `dump-anchor`，冒烟验证：.anchor.yml 含
  完整认知条件，turns 空（步 3 填）。
