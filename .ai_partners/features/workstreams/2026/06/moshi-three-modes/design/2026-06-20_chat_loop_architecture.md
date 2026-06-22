# Moshi 对话环路 + 架构解耦设计

## 背景

Moshi 当前聊天是 echo 原型。需打通 用户↔Ghost 的对话环路，同时解决 Moshi 与 Reflex 的耦合问题。

## 架构演进

### v1：Topic 方案（已废弃）

ChatTopic pub/sub，Ghost 被动读 TopicWindow。基础设施全通但 Ghost 不主动来取消息。

**废弃原因**：聊天是"事件"而非"数据"。Topic 是被动数据结构。

### v2：Moshi ↔ Reflex 分裂为两个 Matrix Cell（备选，暂不采用）

两个独立 cell，Ghost 跨 channel 编排命令。

**搁置原因**：Ghost 需要两步操作（读快照→存档），增加幻觉风险。

### v3：单 Channel + 模块化（当前代码结构）

一个 "moshi" Channel 承载所有命令，模块化拆分代码。Signal 驱动对话。

### v4：Topic 桥接解耦（当前设计方向，参考 sensors 模式）

**参考实现**：`.moss_ws/apps/sensors/`

audio_capture 和 listener 是两个独立 App，通过 Topic + Matrix 直接通信：

```
audio_capture 进程                   listener 进程
  │                                    │
  ├─ MatrixAudioTransport              ├─ MatrixAudioTransport
  │   ├─ pub PCM 音频流                │   ├─ 消费 PCM 流
  │   └─ topic_window(AudioRuntimeTopic)│   ├─ pub_topic(SpeechTopic)
  │                                    │   └─ add_signal(AudioSignal)
  └────────────────┬───────────────────┘
                   │
          Matrix/Zenoh Topic 总线
          （零 Ghost 参与）
```

**应用到 Moshi**：

```
Reflex App（渲染 cell）                  Moshi App（课程逻辑 cell）
  │                                      │
  │  页面变更 → pub PageSnapshotTopic     │
  │  (debounce 500ms)                    ├─ topic_window(PageSnapshotTopic)
  │                                      │   └─ save_chapter 时读最新快照
  │                                      │
  ├─ topic_window(ChapterLoadTopic)      │
  │   └─ 收到 → 渲染到白板               │
  │                                      ├─ start_teaching → pub ChapterLoadTopic
  └──────────────────┬───────────────────┘
                     │
            Matrix/Zenoh Topic 总线
            （零 Ghost 参与）
```

**新增两个 Topic**：

```python
class PageSnapshotTopic(TopicModel):
    """Reflex 页面状态快照"""
    layout: str
    fields: dict          # {title, sub_title, main_text, annotations, images_count}

class ChapterLoadTopic(TopicModel):
    """Moshi 请求 Reflex 加载章节"""
    course_name: str
    chapter_index: int
    chapter_data: dict    # {title, sub_title, main_text, annotations, images}
```

**save_chapter 改造**：不再读 `_SNAPSHOTS`，改为读 `PageSnapshotTopic` TopicWindow。

**switch_to_chapter 改造**：不再操作 QUEUE，改为 pub `ChapterLoadTopic`。

### 方案对比

| | v2（Ghost 编排） | v4（Topic 桥接） |
|---|---|---|
| Ghost 参与 | Ghost 跨 channel 编排 | **零 Ghost 参与** |
| 参考实现 | 无 | **sensors 已验证** |
| 改动量 | 大 | 中（两个 Topic + 消费端） |

## 当前架构（v3，代码实现中）

### 进程拓扑

```
Reflex 进程 (:3000) — lifespan → moss()
  ├─ Matrix.discover()
  ├─ 注册 Channel "moshi"（渲染 + 课程 + 聊天命令）
  ├─ context_messages（课程状态 + chat-instruction + 模式感知信息隔离）
  └─ 内部 HTTP 桥接 (:9733)
        ├─ POST /_internal/chat_in → InputSignal.add_signal()
        └─ GET  /_internal/chat_stream → SSE（asyncio.Event 驱动）

main.py (:9731) — HTTP proxy，不碰 Matrix
Ghost 进程 — 发现 Channel "moshi"
```

### 对话环路（Signal 驱动）

```
用户输入 → main.py → :9733/_internal/chat_in
  → InputSignal().to_signal(Message(text))
  → matrix.session.add_signal(signal)     # Zenoh "MOSS/default/signals"
  → Ghost InputSignalNucleus → Impulse → Attention
  → 读 context_messages → 看到 chat-instruction
  → <moshi:chat_reply>回复</moshi:chat_reply>
  → _SSE_QUEUE.append + _SSE_EVENT.set()
  → SSE 推送 → main.py pipe → 浏览器
```

## 关键架构决策

| # | 决策 | 状态 |
|---|------|------|
| 1 | 聊天走 Signal 而非 Topic | ✅ 实现 |
| 2 | 删除 ChatTopic/TopicWindow | ✅ 实现 |
| 3 | context 按模式信息隔离（preparing 才发 Reflex 命令信息） | ✅ 实现 |
| 4 | 暂不拆分 cell，模块化组织代码 | ✅ 当前策略 |
| 5 | Moshi ↔ Reflex 通过 Topic 桥接（v4） | ⏳ 设计完成，等解耦阶段实现 |

## 踩坑记录

| # | 现象 | 根因 | 解决 |
|---|------|------|------|
| 1 | main.py `Matrix already started` | Matrix cell 单实例锁 | HTTP 桥接 |
| 2 | ChatTopic/_CHAT_WINDOW 被 hot-reload 重置 | 模块级变量 | 换 Signal（不依赖模块变量） |
| 3 | Ghost 不主动读 context | Topic 是数据非事件 | Signal 唤醒 |
| 4 | `aspect-video` 裁切聊天面板 | CSS 布局 | `h-full` |
| 5 | logger 不可见 | `getLogger(__name__)` 无 handler | `getLogger("moss")` |
| 6 | `to_signal()` keyword error | 参数是 `*messages` 位置参数 | `to_signal(Message(...))` |
| 7 | `Text()` positional arg | 需 keyword `Text(text=...)` | `Text(text=...)` |

## 实现进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 1.1 | 删除 ChatTopic | ✅ |
| 1.2 | Signal 发消息 | ✅ |
| 1.3 | 删除 TopicService/TopicWindow | ✅ |
| 1.4 | SSE 改用 asyncio.Event | ✅ |
| 1.5 | context chat-instruction + 信息隔离 | ✅ |
| 1.6 | chat_reply prompt 调优 | ⏳ 待调 |
| 2 | 讲课流程梳理 + 跑通 | ⏳ 进行中 |
| 3 | Topic 桥接解耦 | ⏳ 设计完成 |
