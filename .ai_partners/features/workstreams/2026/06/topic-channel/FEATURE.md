---
title: Topic Channel — TopicModel 即 Channel 的双向 CTML 接口
status: draft
priority: P1
created: 2026-06-10
updated: 2026-06-10
depends:
  - topic-ringbuffer
  - topic-cache
milestone:
description: >-
  泛型 Channel 工厂：传入 TopicModel 子类，自动获得 context messages（TopicWindow 流式消费）
  和 publish(text__) 命令。Ghost 通过闭包"看见"结构化 topic 数据，通过 CTML tag body 写入 JSON。
  零样板代码，任何 TopicModel 一行变成可交互 Channel。
---

# Topic Channel

## Motivation

当前 Topic 体系已有完整的协议层（TopicService / Publisher / Subscriber）和消费层
（DequeTopicWindow），但 Ghost 仍然无法直接与 topic 交互。要让 Ghost 感知某个
topic 流（如 RuntimeEvent、LogTopic、传感器数据），开发者需要手写 Channel：
创建 window、写 context 生成函数、写 publish wrapper、管理生命周期。每个 topic
类型重复一遍。

Topic Channel 消除这个样板。它不引入新抽象——只是把 TopicModel + TopicWindow +
Channel Builder 三者已有的接口用一条泛型工厂函数缝合。任何 TopicModel 子类，
一行 `new_topic_channel(Model)` 变成 Ghost 可感知可写入的 Channel。

这延续了 Code as Prompt 的哲学：TopicModel 的 pydantic schema 就是对 Ghost
的接口声明——不需要额外的 JSON Schema 转换或手写 prompt。

## Design Index

- `new_topic_channel()` — 唯一公开 API
- 读路径：TopicWindow.values() → format_item 闭包 → context messages（被动感知）
- 写路径：publish(text__: str) → CTML tag body 接收 JSON → TopicModel.model_validate() → TopicService.pub()
- 命令签名暴露 TopicModel 的 JSON schema（通过 `comments` 参数），模型从签名学会数据格式
- TopicService 通过 IoC 获取（`CommandUtil.force_get_contract(Matrix).topics()`）
- on_item hook：topic 到达时回调，Nucleus 可借此决定是否生成 Impulse（Topic Channel 自身不涉 Signal）
- 典型集成：PerceptionNucleus 监听多 topic → `nucleus.as_channel` → Ghost 全域感知

### Transport 分层

Topic Channel 只负责 Topic transport。Signal 是另一条并列的 transport，有独立的优先级协议。
两者都通向 Nucleus，Nucleus 聚合后产生 Impulse → Mindflow → Ghost。

```
Topic (广播)  ──→ Topic Channel (窗口 + context) ──→ Nucleus ──→ Impulse → Mindflow → Ghost
Signal (优先级) ──────────────────────────────→ Nucleus ──→ Impulse → Mindflow → Ghost
```

Topic Channel 不做 Signal 的事——它不判断优先级、不抢占。它维护窗口、提供 context、
暴露 on_item hook。Nucleus 通过 hook 决定什么值得生成 Impulse。

### 关于 Nucleus 与 Transport 的讨论

> 以下为人类工程师原话。

Signal 是有优先级协议的，它本质和 Impulse 非常接近，是一个没有聚合加工的 impulse。

但 channel 之间通讯可以直接通过 topic。signal 是 -> ghost(mindflow)，而 topic 是广播协议。
如果 nucleus 硬是只走 signal，就必须做 监听 topic 端口 -> 发送 signal -> nucleus -> impulse，
会发现拿到 topic 后返回 signal 本身就很像 nucleus 该做的事情。

所以 Nucleus 从来就不是假设只有 signal 一种通讯方式。Nucleus 最重要的是发送 Impulse。
Signal 是一种目的明确的优先级协议而已。

## API

```python
def new_topic_channel(
    model: type[TopicModel],
    *,
    name: str = "",
    max_size: int = 100,
    topic_name: str = "",
    format_item: Callable[[TopicModel], str] | None = None,
) -> MutableChannel:
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | (必填) | TopicModel 子类。其 pydantic schema 定义了 topic 的数据结构 |
| `name` | `""` | Channel 名称（CTML 中的 tag）。为空时从 `model.__name__` 推导为 snake_case |
| `max_size` | `100` | TopicWindow 窗口大小。也是 context messages 的最大条目数 |
| `topic_name` | `""` | 覆盖默认 topic name。为空时使用 `model.default_topic_name()` |
| `format_item` | `None` | 闭包：TopicModel → 可读字符串。为空时默认用 `model_dump_json(indent=0, exclude_none=True)` |

### 返回的 Channel 行为

**context_messages**：每轮 refresh 时调用 `window.values()`，对每个 item 调 `format_item`，
生成一条 `Message`。Ghost 在上下文中"看见"最近 max_size 条 topic 数据。

**publish(text__: str)**：唯一命令。使用 CTML `text__` 约定——模型在 open-close tag 体内
写入 JSON 数据。命令从 `text__` 解析 JSON，构造 TopicModel（`model.model_validate(data)`），
转为 Topic 通过 TopicService 发布。`always_observe=False`（发布是"行动"不是"信息"）。

命令的 `comments` 参数动态生成：从 `model.model_json_schema()` 提取字段信息，
渲染为注释行。模型看到函数体中的 `# Schema:` 注释，知道 tag body 中该写什么 JSON。

模型使用示例：
```xml
<topic:runtime_events:publish>
{"event": "cell_started", "address": "node/tools/web-fetch", "pid": 12345}
</topic:runtime_events:publish>
```

**startup**：从 IoC 获取 Matrix → TopicService，调用 `create_window_for(model, max_size=max_size)`，
`await wait_started()`。如果 TopicService 不可用（非 Matrix 环境），channel 标记为 unavailable。

## Key Decisions

### 1. format_item 是闭包，不是协议

`format_item` 是一个普通 `Callable[[TopicModel], str]`，不是 ABC 或 Protocol。
原因：
- 不同 TopicModel 的数据形态差异巨大——RuntimeEvent 关心 event+address+pid，
  LogTopic 关心 level+message，传感器关心数值+单位。无法用一个默认格式化覆盖所有场景。
- 闭包让调用方完全控制 token 用量。可以输出完整 JSON，可以只输出摘要行，
  可以根据 Ghost 的认知需求定制视图。
- 默认值 `model_dump_json()` 提供开箱即用的行为，但预期生产使用都会传自定义闭包。

对比拒绝的方案：
- 在 TopicModel 上加 `.format()` 方法：污染数据模型。格式化是 Channel 的视图逻辑，
  不是 TopicModel 的职责。同一个 RuntimeEvent 在不同的 Channel 实例中可能需要
  不同的格式。
- 用 Jinja2 模板字符串：引入不必要的依赖和复杂度。闭包就是 Python，够用。

### 2. publish 使用 CTML text__ 约定 + JSON schema 注释

`publish(text__: str)` — 模型在 CTML open-close tag body 中写 JSON 数据。
命令的 `comments` 参数从 `TopicModel.model_json_schema()` 动态生成，告诉模型
JSON 的字段、类型和含义。

不用动态构造函数签名（与 TopicModel 字段一一对应），原因：
- CTML `text__` 是自然的多行数据入口——适合 JSON/YAML/代码等结构化内容。
- 保持命令接口稳定——无论 TopicModel 有多少字段，publish 始终是 `publish(text__: str)`。
- 模型通过 comments 中的 schema + context messages 中的样例，两种路径学会数据格式。

### 3. max_size 同时控制 window 和 context

同一个 `max_size` 控制窗口容量和 context messages 条数。原因：
- 两个值在语义上高度相关——Ghost 能"看见"的就是 window 里的。
- 如果引入独立的 `context_limit`，会产生"window 有 100 条但 context 只展示 20 条"
  的情况，Ghost 看到的不是完整的窗口状态，造成认知偏差。
- 想减少 token 用量？减小 `max_size`。简单、可预测。

### 4. TopicService 从 IoC 获取，不在构造时注入

TopicService 在 channel startup 时通过 `CommandUtil.force_get_contract(Matrix).topics()`
获取，不在 `new_topic_channel()` 时传入。原因：
- Channel 构造和 Channel 运行是两个阶段。构造时 TopicService 可能还不存在
  （Matrix 尚未启动）。
- 与 MOSS Channel 生命周期一致——依赖在运行时解析，不在构造时绑定。
- 如果不在 Matrix 环境中（纯 CTMLShell 场景），channel 标记 unavailable，
  Ghost 可以看到 channel 存在但不可用。

### 5. 不做 subscribe 命令

第一版只有 `publish`，没有 `subscribe` / `poll` 命令。原因：
- 读路径完全由 context messages 覆盖——Ghost 不需要手动 poll，window 自动推送
  到上下文。
- 如果后续需要主动轮询（如"等待下一个特定 topic"），再加 `poll(timeout: float)` 命令。
  但那是 pull 模式，context messages 是 push 模式，两者服务不同场景。

## Future Extensions

### 运行时调参命令

增加 `set_max_size(n: int)` 命令，允许 Ghost 在运行时调整窗口大小。
需要 TopicWindow 新增 `resize(n)` 接口——`DequeTopicWindow` 内部 deque 重建即可。
场景：Ghost 发现重要事件时临时扩大窗口保留更多上下文，常态下缩小节省 token。

这不是 v1 范围。TopicWindow 接口稳定后再加。

## Open Questions

- `format_item` 的返回值是否应该允许返回 `Message` 而不仅是 `str`？（如返回图片、富文本）
  当前 YAGNI——topic 数据主要是结构化文本，str 足够。
- 是否需要 `with_context` / `with_publish` flag 来禁用部分功能？当前不需要——
  如果不需要 context，设置 `max_size=0` 即可；不需要 publish，未来可以考虑加 flag。
- 多 topic channel？一个 channel 监听多个 topic？当前不需要——需要多个 topic
  就创建多个 channel，组合进 main channel。Unix 哲学：do one thing well。

## Implementation Notes

- 存放位置：`src/ghoshell_moss/channels/topic_channel.py`
- 测试：`tests/ghoshell_moss/channels/test_topic_channel.py`，使用 QueueBasedTopicService
- 依赖：仅 `ghoshell_moss.core`（concepts.topic + blueprint.channel_builder + blueprint.matrix）
- 遵循 channels/CLAUDE.md 的 L1 构建层级（`new_channel()` + Builder）
- Status: alpha
