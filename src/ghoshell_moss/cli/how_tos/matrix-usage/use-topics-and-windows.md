---
title: Use Topics and Topic Windows
description: 在 Matrix 体系中使用 Topic 协议做跨进程 pub/sub 通讯，以及用 TopicWindow 消费"最新 N 条"数据。面向 app 开发者和 Ghost 开发者，帮你判断什么时候用 topic、什么时候用 window、怎么用。
---

# Use Topics and Topic Windows

## 背景

Matrix 提供了五条通讯路径。选对路径是第一步：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.session:Session
```

| 路径 | 模式 | 何时用 |
|------|------|--------|
| `output` | 单向广播 OutputItem | 系统对外的展示消息 |
| `signal` | Mindflow 感知信号 | 驱动 Ghost 三循环 |
| `file` | 文件级读写 | 大块数据、持久化 |
| `stream` | 字节流 pub/sub | 实时流式数据（logos） |
| **topic** | 强类型 n×m 广播 | 结构化事件、跨进程通讯 |

Topic 是唯一提供**强类型 + 多对多 + 自解释 schema** 的路径。适合监控数据、状态变更、日志、对话上下文等场景。

Topic 协议的核心抽象和 TopicWindow 接口：

```bash
moss codex get-interface ghoshell_moss.core.concepts.topic:TopicService
moss codex get-interface ghoshell_moss.core.concepts.topic:TopicWindow
```

## 什么时候用 Topic

满足以下**任意一条**就该用 topic：

- 多个消费者需要同一种数据（一对多广播）
- 生产者和消费者在不同进程（跨进程通讯，Zenoh 透明传输）
- 需要类型安全和 schema 自解释（`moss manifests topics` 可发现）
- 数据是结构化的、事件级的（秒级大脑事件，不是高频传感器帧）

不适合 topic 的场景：
- 高频原始数据（如音频采样点）→ 用 stream 或直接 Channel 通讯
- 仅需展示给人类 → 用 session output
- 驱动 Ghost 思考 → 用 signal

## 发现可用的 Topic

```bash
# 列出环境中所有已声明的 topic 类型
moss manifests topics

# 查看某个 topic 的类型定义和 schema
moss manifests topics system/error
```

如果没有找到需要的 topic，参照 howtos 中的 manifest 注册指南，在 `MOSS.manifests.topics` 下声明 TopicModel 子类。

## Pub/Sub 基础用法

在 app 中通过 Matrix 获取 TopicService：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.matrix:Matrix
```

关键步骤：

1. **发布** — 创建 publisher 或直接 `service.pub()`
2. **订阅** — `service.subscribe_model()` 返回 Subscriber，`async with` 托管生命周期
3. **消费** — `await subscriber.poll_model()` 阻塞获取下一条

搜索 topic 相关测试了解完整用法。

## TopicWindow："最新 N 条"消费模式

当消费者只需要**最近 N 条数据**而不是全量历史时，用 TopicWindow：

```
moss codex get-interface ghoshell_moss.core.concepts.topic:TopicWindow
```

### 适用场景

- 监控面板 — 显示最新的 CPU/内存/延迟指标
- 对话上下文窗口 — TTS 和 ASR 之间的滚动上下文
- 日志 tail — 显示最近的日志行
- 波形图 — 最近的音频帧

### 创建和消费

从 TopicService 创建：

```python
# 创建窗口（订阅自动开始）
win = service.create_window_for(ErrorTopic, max_size=20)
await win.wait_started()

# 非破坏性快照 — 监控面板的核心 API
latest = win.values()       # list[ErrorTopic], index 0 最旧, -1 最新

# 时间戳轮询 — 无回调的轻量检查
if win.changed_at() > last_seen:
    refresh(win.values())

# 回调 — 数据到达时自动通知
win.on_change(lambda w: render(w.values()))
# 去抖：等说话人停顿 1 秒后再处理
win.on_change(lambda w: transcribe(w.values()), debounce=1.0)
# 节流：最多每 5 秒回调一次
win.on_change(lambda w: update_chart(w.values()), throttle=5.0)
```

窗口生命周期绑定 TopicService——service 关闭时自动清理，无需手动 close。

### 何时用 Window vs 直接 Subscriber

| 需求 | 用 |
|------|-----|
| 处理每一条消息（不丢） | Subscriber.poll() |
| 看最新 N 条快照 | TopicWindow.values() |
| 数据到达时通知 | TopicWindow.on_change() |
| 去抖 / 节流触发 | TopicWindow.on_change(debounce=, throttle=) |
| 轻量轮询 | TopicWindow.changed_at() |

Window 不替代 Subscriber——它是在 Subscriber 之上的消费层抽象。

## 常见问题

### 问题：`moss manifests topics` 输出为空

Topic 声明还没被创建。参照已有 TopicModel 子类写一个：

```bash
moss codex get-source ghoshell_moss.core.concepts.topic   # 看 ErrorTopic 和 LogTopic 的定义
```

然后在 `MOSS.manifests.topics` 下声明，CLI 会自动发现。

### 问题：Window 创建后 values() 始终为空

创建后需要 `await win.wait_started()` 等待订阅激活，再发布数据。否则存在竞态：发布在订阅就绪之前完成，数据丢失。

### 问题：回调没有触发

回调从线程池调用。检查：
- `debounce` 和 `throttle` 参数是否导致延迟
- 回调内部是否抛了异常（被 logger 捕获，不会传播）

## 探索路径

```bash
moss codex list ghoshell_moss.core.topic     # core 层实现
moss codex list ghoshell_moss.matrix.topics    # host 层实现
```

搜索 topic 相关测试了解用法。查阅 howtos 了解 manifest 声明和环境发现。

## 文档目标

读者按照本文档操作，应该能够：
1. 判断一个通讯需求该用 topic、window、还是其他路径
2. 通过 `moss manifests topics` 发现可用的 topic 类型
3. 从 TopicService 创建 Subscriber 或 TopicWindow 并消费数据
4. 知道何时用 values()、changed_at()、on_change() 三种消费模式
5. 在 `tests/ghoshell_moss/topics/test_window_protocol_suite.py` 中找到可运行的参考代码
