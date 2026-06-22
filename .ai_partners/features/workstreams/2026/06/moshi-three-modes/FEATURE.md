---
title: Moshi Three Modes — 魔师内容传递协作系统
status: in-progress
priority: P1
created: 2026-06-20
updated: 2026-06-22T22:00
depends: []
milestone:
description: >-
  开发 Moshi 的三大场景模式（备课/人练/AI主讲），先跑通讲课流程，再推进解耦。
---

# Moshi Three Modes

## 当前优先级

**双阻塞**：bug #23（字幕链路不通）和 bug #24（Ghost 不调 advance_point 导致卡住）必须解决才能继续集成测试。

## 当前状态（2026-06-22 会话，第五轮 — 集成测试受阻）

**v2 代码实施已完成**（8 个文件），但手工集成测试暴露了两个新问题，均阻塞 Phase 2b 闭环。

### 集成测试日志分析（moss.log, 2026-06-22 16:48-16:54）

运行命令：`moss-run-ghost echo --mode default`（注意：是 `default` 不是 `show`）

关键日志行：

| 行号 | 时间 | 内容 | 含义 |
|------|------|------|------|
| 846 | 16:48:55 | `MOSS_MODE_NAME=default` | Moshi 进程 mode=default |
| 849 | 16:48:55 | `enable_topic=False, topic_path=moshi/subtitle` | ConfigStore 未命中覆盖（default 模式下无 subtitle 配置） |
| 852 | 16:48:55 | `字幕回调已注入 Speech（HTTP 旁路，同进程）` | 回退链三层全失败，走入 HTTP 旁路 |
| 876 | 16:48:58 | `lecture_start Signal 已发送, points=5` | 讲课启动正常 |
| 930-949 | 16:49:00-14 | TTS batch 创建→连接→接收→完成 | TTS 工作正常，~12s 音频 |
| 950 | 16:49:14 | `wait 28.38s for playing` | 音频播放队列预估 28s |
| 958 | 16:49:43 | `speaker running=False` | 播放完成 |
| 964-966 | 16:49:44 | `interpreter stopped`, `compiled=0 done=1` | **Ghost 只执行了 1 个命令（`__content__`），没有 `advance_point`** |
| 966-1045 | 16:49:44→16:54 | 只有 SSE keep-alive 重连 | **Ghost 完全静默，无新输出** |

### 字幕回退链分析：三层全失败

```
Layer 1: ConfigStore.get(SubtitleTopicConfig)        → enable_topic=False ← 模式覆盖未生效
Layer 2: Environment.discover().moss_mode_name       → "default"
         → import MOSS.modes.default.configs          → 无 subtitle_topic_config ← 配置只在 show 模式
         → getattr 返回 None → isinstance 跳过
Layer 3: HTTP 旁路 new_subtitle_callback()           → 只在同进程有效 ← Ghost 进程 Speech 未收到回调
```

**结论**：三层回退链无一有效。要打通字幕链路，至少要修复 Layer 1（ConfigStore 把 show mode 的覆盖应用到 Ghost 进程），或直接用 `--mode show` 并确保 ConfigStore 正确工作。

### v2 实施完成项（不变）

| # | 步骤 | 文件 | 状态 |
|---|------|------|------|
| 1 | 注册 `TTSSpeechServiceProvider` | `manifests/providers.py` | ✅ |
| 2 | 定义 `SubtitleTopic` TopicModel | `topics/audio.py` | ✅ |
| 3 | 定义 `SubtitleTopicConfig` ConfigType | `core/speech/subtitle_config.py`（新） | ✅ |
| 4 | 注册 Topic + Config 到 manifests | `manifests/configs.py`, `manifests/topics.py` | ✅ |
| 5 | 扩展回调签名 | `stream_tts_speech.py` | ✅ |
| 6 | 改造 `SpeechServiceProvider.factory()` | `speech_service_provider.py` | ✅ |
| 7 | Mode config 中启用 | `modes/show/configs.py` | ✅ |
| 8 | Moshi 侧 Topic 消费协程 | `moss_in_reflex.py` | ✅ |
| 9 | HTTP 旁路 fallback | `moss_in_reflex.py` | ✅ （三层回退） |
| 10 | 集成测试 | 跨进程手工测试 | 🚫 受阻于 #23 #24 |
| 11 | 清理旧 HTTP 旁路 | — | ⏳ |

**本轮会话新增踩坑**：
- bug #23：字幕回退链三层全失败（ConfigStore 模式覆盖不生效 + mode=default 无配置 + HTTP 同进程无效）
- bug #24：Ghost 不调 advance_point — 叙述完第一段后静默，无后续 Signal → 卡死
- 附注：第一轮测试（16:46）还发生了 `attention aborted during execute` + `RuntimeError: MossRuntime is not running`（MossRuntime 提前退出），第二轮未复现

## Design Index

- 对话环路架构（含踩坑）：`design/2026-06-20_chat_loop_architecture.md`
- **讲课模式完整方案**：`design/2026-06-21_teaching_mode.md`
- **bug #22 修复方案 v1**：`.moss_ws/apps/ui/moshi/tmp/subtitle.md`
- **bug #22 修复方案 v2（最终采纳）**：`.moss_ws/apps/ui/moshi/tmp/subtitle_v2.md`
- TTS 字幕方案调研：`.moss_ws/apps/ui/moshi/research-tts-subtitle-progress.md`
- 产品方案：`.moss_ws/apps/ui/moshi/魔师Moshi (3).md`
- App 代码入口：`.moss_ws/apps/ui/moshi/main.py`
- 核心模块：`.moss_ws/apps/ui/moshi/moss_in_reflex/`
- 独立 Reflex App：`.moss_ws/apps/ui/reflex/`
- Topic 桥接参考实现：`.moss_ws/apps/sensors/`

## Key Decisions

### 聊天：走 Signal（决策已定，基础设施完成）

- 删除 ChatTopic/TopicWindow，用 InputSignal 唤醒 Ghost
- SSE 改用 asyncio.Event 驱动
- context_messages 按模式做信息隔离
- **新增**：teaching 模式下不注入 chat-instruction，杜绝 Ghost 自言自语

### Signal vs Topic（已验证，来自 sensors 参考实现）

**Topic 不能通知 Ghost。** sensors 的标准模式是 Topic + Signal 同时发布：
- Topic：跨 app 数据共享（Zenoh pub/sub，Ghost 不感知）
- Signal：唤醒 Ghost（mindflow → Nucleus → Impulse → Attention）

因此在讲课流程中，Moshi↔Reflex 数据桥接用 Topic（Phase 4），唤醒 Ghost 永远用 Signal。

### 讲课编排权归服务端（已实施）

**不让 Ghost 做编排决策。** `/_internal/lecture/start` 在服务端原子完成：
load_course + 强制切 lesson 布局 + 渲染首章 + 初始化 LectureBrain，
然后发 Signal 通知 Ghost。Ghost 只需叙述 + 调 advance_point。

编排逻辑在服务端内部，Phase 4 解耦时只需换底层通信机制（Queue → Topic），编排本身不变。

### advance_point 后发 Signal 驱动继续（本轮新增）

Ghost 输出 CTML 命令后倾向于停止生成（function call = 回合结束）。参考 `.moss_ws/apps/genkits/image` 的 `emit_generation_signal` 模式，`advance_point` 执行完毕后发 `InputSignal(Priority.NOTICE)` 唤醒 Ghost 继续叙述。Signal 携带当前 active 段落文本，Ghost 可直接继续。

`lecture_ended` 不发 Signal——讲课结束，Ghost 应停止。

### ChannelCtx 不在 HTTP handler 中可用

aiohttp 的 HTTP handler 运行在裸 asyncio 上下文中，没有 MOSS 的 ChannelCtx。
在 `moss()` 初始化阶段捕获 `_course_storage`、`_registry` 等引用，
供内部 HTTP handler（`_internal_*`）使用。channel 命令仍然用 `ChannelCtx.container()`。

### 数据桥接：Moshi ↔ Reflex 通过 Topic（Phase 4 方案）

- 参考 `.moss_ws/apps/sensors/` 的实现——audio_capture 和 listener 两个独立 App
  通过 Topic + Matrix 直接通信，不经过 Ghost
- Moshi 和 Reflex 拆分到两个独立 cell，用 Topic 桥接页面状态和章节加载

### 翻章同步：AudioRuntimeTopic 门控（Phase 2 设计）

**问题**：Ghost 裸文本走 `__main__.__content__`（TTS），`advance_point` 走 `apps.ui_moshi` channel。跨 channel 并行执行导致 TTS 还在播上一章，翻章已触发。

**约束**：不能强行把 TTS 搬到 moshi channel（narrate 方案违反跨 channel 并行架构，且 `chunks__` 跨协程迭代触发 bug #19/#20）。

**方案**：复用 sensors/listener 的 `AudioRuntimeTopic` 门控模式。`MiniAudioStreamPlayer` 已发布 `AudioRuntimeTopic(device_name="speaker", running=True/False)`。`advance_point` 通过 `TopicWindow[AudioRuntimeTopic]` 订阅，翻章前等待 `running=False`。

**不改**：`__main__.__content__`、跨 channel 并行、Ghost 裸文本输出方式。

### 句级字幕：TTSSpeechStream subtitle_callback（Phase 2 设计）

**问题**：需要 TTS 实际朗读的文本（同语音完全一致），逐句显示为字幕。

**约束**：裸文本→TTS 管道在核心 speech 模块内，moshi channel 不可见。

**方案（方案 C）**：在 `TTSSpeechStream`（`src/ghoshell_moss/core/speech/stream_tts_speech.py`）注入可选 `subtitle_callback(text, is_final)`。`_buffer()` 累积文本按标点（`。！？；\n`）切句入队，`_play_loop()` 消费 `tts_batch.items()` 时逐句回调。moshi 侧注入回调写 SSE → 字幕同步显示。

**为什么不选其他方案**：
- 方案 A（_feed_stream 层分句）：CTML parser chunk 不按句子对齐，且 feed 时机早于音频播放
- 方案 B（SpeechTopic 整段字幕）：10-30 秒延迟，讲课不可用
- 方案 D（时间估算）：文本→音频非线性，精度太低

### Context 调整：每轮一个 talking point

**改动**：`context_messages.py` 讲课指令从 "调完立即继续叙述，不要停顿" 改为 "调完 advance_point 后停止输出，等待系统 Signal 后再继续下一段"。

**原因**：Ghost 每轮只讲一个 talking point。TTS 播完 → advance_point 门控放行 → Signal 唤醒 → 下一轮。一轮一点，时序自然对齐。

## Implementation Phases

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | Signal 对话环路 + 讲课骨架 | ✅ 核心环路跑通 |
| 2a | TTS 翻章同步（AudioRuntimeTopic 门控） | ✅ 已实施（2026-06-22） |
| 2b | 句级字幕（SubtitleTopic 跨进程 Topic 总线） | ✅ 已实施，待集成测试 |
| 2c | 弹幕（飞书消息 → SSE） | 待开始（方案需调整） |
| 3 | 语音打断 + 总结 | 待开始 |
| 4 | Topic 桥接解耦 | 设计完成，待实现 |

## Phase 2b 修订：SubtitleTopic 跨进程 Topic 总线（v2）

> 方案分析：`.moss_ws/apps/ui/moshi/tmp/subtitle_v2.md`
> 核心变更：用 Zenoh Topic 总线替代 HTTP 旁路（`new_subtitle_callback()` → `/_internal/subtitle_in`）
> 过渡策略：旧 HTTP 旁路代码保留，通过 `SubtitleTopicConfig.enable_topic` 开关控制

### v2 全链路数据流

```
Reflex 进程声明 SubtitleTopicConfig(enable_topic=True, topic_path="moshi/subtitle")
       │
       ▼ ConfigStore (workspace/configs/) ← 文件系统，两端共享
       │
Ghost 进程 SpeechServiceProvider.factory() 读配置 → 创建 _publish_subtitle 闭包
       │
       ▼ _subtitle_callback(text, is_final, batch_id)
       │    └─► SubtitleTopic.to_topic() → TopicService.pub(topic)
       │         └─► Zenoh session.put(key_expr, json) [线程池]
Zenoh 总线  ← MOSS/{session_scope}/topics/moshi/subtitle
       │
       ▼ Reflex 进程 _consume_subtitle() 协程 poll_model()
       │    ├─► 压入 _SUBTITLE_QUEUE (async Lock)
       │    └─► _SUBTITLE_EVENT.set()
       │         └─► _internal_subtitle_stream (SSE) → main.py proxy → Browser
```

### v2 实施步骤

按依赖关系排列，标注风险等级（🔴高 🟡中 🟢低）：

| # | 步骤 | 文件 | 风险 | 说明 |
|---|------|------|------|------|
| 1 | 注册 `TTSSpeechServiceProvider` | `manifests/providers.py` | 🟢 | 1 行 import。当前未注册导致 IoC 无法发现 |
| 2 | 定义 `SubtitleTopic` TopicModel | `src/ghoshell_moss/topics/audio.py` | 🟢 | ~20 行，与 `AudioRuntimeTopic` 同文件。字段：text, is_final, batch_id。topic_type="core/speech/subtitle" |
| 3 | 定义 `SubtitleTopicConfig` ConfigType | `src/ghoshell_moss/core/speech/subtitle_config.py`（新） | 🟢 | ~25 行。字段：enable_topic: bool = False, topic_path: str = "moshi/subtitle" |
| 4 | 注册 Topic + Config 到 manifests | `manifests/topics.py`, `manifests/configs.py` | 🟢 | 各 1 行 import |
| 5 | 扩展回调签名 → `(text, is_final, batch_id)` | `stream_tts_speech.py` | 🟡 | `_subtitle_callback` 签名 + `TTSSpeechStream.__init__` 新增 `batch_id` 参数。上游调用方（`SpeechModule` / `__content__`）需追溯一轮 |
| 6 | 改造 `SpeechServiceProvider.factory()` | `speech_service_provider.py` | 🟡 | 读 `SubtitleTopicConfig` → 获取 `TopicService` → 创建 `_publish_subtitle` 闭包（用 `TopicService.pub()` 直接发布，不用 Publisher 抽象） |
| 7 | mode config 中启用 | `modes/show/configs.py`（或 default） | 🟢 | `SubtitleTopicConfig(enable_topic=True, topic_path="moshi/subtitle")` |
| 8 | Moshi 侧 Topic 消费协程 | `moss_in_reflex.py` | 🟡 | `_consume_subtitle()` 协程：`TopicWindow` → `subscriber.poll_model()` → `_SUBTITLE_QUEUE` → `_SUBTITLE_EVENT.set()`。避免 `on_change` 回调（线程池→事件循环桥接问题） |
| 9 | 保留旧 HTTP 旁路，config 开关控制 | `moss_in_reflex.py` | 🟢 | `enable_topic=True` → Topic 路径；`False`（默认）→ HTTP 旁路。过渡期两套共存 |
| 10 | 集成测试 | 跨进程手工测试 | 🟡 | Ghost 进程 ↔ Reflex 进程，确认字幕跨 Zenoh 到达 |
| 11 | 清理旧 HTTP 旁路 | `moss_in_reflex.py`, `main.py` | 🟢 | 删除 `new_subtitle_callback()`、`/_internal/subtitle_in` 端点、`force_fetch(Speech).set_subtitle_callback()`。Topic 路径稳定后执行 |

### v2 关键设计决策

**放弃 Publisher 抽象，改用 `TopicService.pub()` 直接发布**（解决问题 2）：
- `TopicService.pub()` 无需预先建立连接，每次调用即时 pub
- 线程安全已验证（内部 `run_in_executor`）
- 避免了 `ZenohTopicPublisher` async context manager 生命周期绑定问题
- 代价：失去 `declare_publisher` 预声明优化，但字幕消息频率低（每句一次），可忽略

**使用专用消费协程替代 `on_change` 回调**（解决问题 4）：
- `TopicWindow.on_change` 在线程池执行，不能操作 `asyncio.Lock`/`asyncio.Event`
- 消费协程 `_consume_subtitle()` 天然在事件循环上运行，无线程安全问题
- 与 `AudioRuntimeTopic` 的轮询模式一致

**保留旧 HTTP 旁路**（解决问题 5）：
- `SubtitleTopicConfig.enable_topic` 默认 `False`，保持向后兼容
- 确认 Topic 路径稳定后再清理旧代码

### v2 未解决的开放问题

1. **`batch_id` 上游来源**：需在调用 `new_tts_stream()` 时传入。确认是 `SpeechModule` 层还是 `__content__` 命令层生成
2. **TopicWindow 底层 Subscriber 访问**：消费协程需要 `window._subscriber`（私有属性），或需在 `TopicWindow` ABC 上暴露 `poll()` 方法
3. **Zenoh 断连恢复**：`ZenohTopicService` session 断连时的重连行为需确认
4. **多 Ghost 实例**：同一 `session_scope` 下多 Ghost 同时说话时，字幕 topic 混在一起。`batch_id` 可用于前端过滤但需前端配合

### Phase 2a：TTS 翻章同步（AudioRuntimeTopic 门控）✅

| # | 内容 | 涉及文件 |
|---|------|---------|
| 2a.1 | `moss()` 初始化时创建 `TopicWindow[AudioRuntimeTopic]` | `moss_in_reflex.py` |
| 2a.2 | `advance_point` 翻章前等待 `device_name="speaker"` 的 `running=False`（150ms 轮询） | `moss_in_reflex.py` |
| 2a.3 | context 指令改为"调完停止，等 Signal 再继续" | `context_messages.py` |

### Phase 2c：弹幕（后续）

| # | 内容 | 涉及文件 |
|---|------|---------|
| 2c.1 | `check_messages` 实际对接飞书 channel | `moss_in_reflex.py` |
| 2c.2 | 飞书消息 → `_DANMAKU_QUEUE` → SSE 弹幕 | `moss_in_reflex.py` |
| 2c.3 | 弹幕频率控制（缓冲 50 条，堆积 20+ 丢弃非@旧消息） | `moss_in_reflex.py` |

## Phase 1 实现进度

| # | 内容 | 状态 |
|---|------|------|
| 1.1 | chapter_data 增加 speaker_notes 字段 | ✅ |
| 1.2 | LectureBrain 状态机（lecture_brain.py） | ✅ |
| 1.3 | 新命令（advance_point、check_messages、resume_lecture） | ✅ |
| 1.4 | 课程列表 API（/_internal/courses → main.py proxy） | ✅ |
| 1.5 | 前端讲课页面（teachingL0.html） | ✅ |
| 1.6 | Ghost context 更新（speaker_notes + 约束 + Signal 驱动） | ✅ |
| 1.7 | 字幕通道 | 延至 Phase 2 |

### 新增 API 路由

| 路由 | Method | 说明 |
|------|--------|------|
| `/api/courses` | POST | 课程列表（proxy → /_internal/courses） |
| `/api/lecture/start` | POST | 讲课启动（proxy → /_internal/lecture/start） |

### 新增 Channel 命令

| 命令 | 说明 |
|------|------|
| `advance_point` | 段落推进，返回 point_advanced / chapter_advanced / lecture_ended。执行后发 Signal 唤醒 Ghost |
| `check_messages` | 检查飞书消息（Phase 1-2 桩） |
| `resume_lecture` | 暂停后恢复讲课 |

### 新增文件

| 文件 | 说明 |
|------|------|
| `moss_in_reflex/lecture_brain.py` | 讲课状态机，不依赖 Reflex |

## 踩坑记录

| # | 现象 | 根因 | 解决 | 日期 |
|---|------|------|------|------|
| 1 | main.py `Matrix already started` | Matrix cell 单实例锁 | HTTP 桥接 | 2026-06-20 |
| 2 | ChatTopic/_CHAT_WINDOW 被 hot-reload 重置 | 模块级变量 | 换 Signal（不依赖模块变量） | 2026-06-20 |
| 3 | Ghost 不主动读 context | Topic 是数据非事件 | Signal 唤醒 | 2026-06-20 |
| 4 | `aspect-video` 裁切聊天面板 | CSS 布局 | `h-full` | 2026-06-20 |
| 5 | logger 不可见 | `getLogger(__name__)` 无 handler | `getLogger("moss")` | 2026-06-20 |
| 6 | `to_signal()` keyword error | 参数是 `*messages` 位置参数 | `to_signal(Message(...))` | 2026-06-20 |
| 7 | `Text()` positional arg | 需 keyword | `Text(text=...)` | 2026-06-20 |
| 8 | 前端讲课页蒙层不消失 | Ghost 讲课时不走 chat_reply | 服务端 `/_internal/lecture/start` 原子完成 | 2026-06-21 |
| 9 | Ghost 乱切布局 | context 没说用哪个布局 | 服务端强制 LayoutEvent；context 约束 | 2026-06-21 |
| 10 | Ghost 重复调 load_course | context 未告知已加载 | context 约束 + 服务端 guard | 2026-06-21 |
| 11 | 5 个 advance_point 后返回 lecture_ended 而非翻章 | Ghost 在 CTML 命令后停止生成，无新 attention 周期推进翻章逻辑 | Signal 驱动：advance_point 后发 InputSignal 唤醒 Ghost | 2026-06-21 |
| 12 | `/_internal/lecture/start` 崩溃 | `_switch_to_chapter` 调 ChannelCtx 但 HTTP handler 无此上下文 | 初始化时捕获 `_registry` 引用 | 2026-06-21 |
| 13 | Ghost 输出 `<_>` 破坏 CTML 解析 | Ghost 用 `<_>` 做段落分隔，CTML 解析器把 `<` 当标签 | context 加 CTML 卫生约束：禁止自创标签 | 2026-06-21 |
| 14 | 讲课模式下 Ghost 自言自语调 chat_reply | chat-instruction 对所有模式注入 | teaching 模式不注入 chat-instruction | 2026-06-21 |
| 15 | Ghost 调 advance_point 后静默 | CTML function call 后模型停止生成，COMMAND-RESULT 不触发新 attention | advance_point 后发 InputSignal(Priority.NOTICE) 唤醒 | 2026-06-21 |
| 16 | lecture_ended 后 Ghost 继续调 check_messages + advance_point | context 无停止规则 + advance_point 在 ended 状态抛异常 | 幂等保护 + context 明确停止规则 | 2026-06-21 |
| 17 | Ghost 用 `<moshi:xxx />` 调命令报 not found | context 写的是短名 `moshi`，运行时注册的是全名 `apps.ui_moshi` | 全部改为 `apps.ui_moshi` | 2026-06-21 |
| 18 | context 中 speaker_notes 和 lecture-state 展示矛盾 | teaching() 读持久化数据（全 pending），lecture-state 读内存（实时状态） | 移除 context-mode 的 talking_points 展示，只保留 lecture-state 的实时进度 | 2026-06-21 |
| 19 | narrate: `'NoneType' object has no attribute 'call_soon_threadsafe'` | `asyncio.create_task(_feed())` 把 `chunks__` 放到后台协程迭代，底层 janus 队列的 `call_soon_threadsafe` 找不到 event loop | 不要跨协程传递 `chunks__`，始终在主协程迭代。`SpeechStream.speak()` 官方 API 已处理此约束 | 2026-06-21 |
| 20 | narrate: `speech unavailable` | `ChannelCtx.container().force_fetch(Speech)` 拿到的 Speech 实例与 Shell `_speech_context_manager` 启动的不是同一个，`is_running()` 返回 False | 在 `moss()` 初始化时用 `matrix.container` 直接捕获（与 `_course_storage`/`_registry` 一致） | 2026-06-21 |
| 21 | narrate 方案整体否决 | 试图把 TTS 搬到 moshi channel 强制串行，违反 MOSS 跨 channel 并行核心设计。用户指正"不符合 moss 架构思想" | 不在 channel 层面强行串行。在共享状态上做协调——翻章用 AudioRuntimeTopic 门控，字幕用 SpeechStream 回调注入 | 2026-06-22 |
| 22 | subtitle_callback 注入到错误的 Speech 实例 | Reflex 进程和 Ghost 进程各有独立的 IoC Container，`matrix.container.force_fetch(Speech)` 在两个进程中返回不同的 `BaseTTSSpeech` 实例。`moss()` 中注入的 callback 设在 Reflex 进程的 Speech 上，Ghost 进程中 Shell 用的 Speech 未收到回调 → `TTSSpeechStream._subtitle_callback is None` → 字幕数据永远不到达 SSE | v2 方案：在 Ghost 进程 `SpeechServiceProvider.factory()` 中注入 `_publish_subtitle` 闭包，通过 Zenoh Topic 总线跨进程发布字幕。代码已实施，但集成测试中被 #23 阻塞 | 2026-06-22 |
| 23 | 字幕 ConfigStore 回退链三层全失败 | 三层回退无一打通：(1) ConfigStore 模式覆盖不生效，Ghost/Moshi 进程均读到 `enable_topic=False` 的默认值（根因可能是 `HostEnvConfigStoreProvider` 未正确扫描 mode config 模块，或 `search_config_infos_from_package` 未找到 instance 类型的 override）；(2) `Environment.discover().moss_mode_name` 返回 `"default"` 而非 `"show"`，`MOSS.modes.default.configs` 无 `subtitle_topic_config` 属性 → 回退导入静默跳过；(3) HTTP 旁路 `new_subtitle_callback()` 只在 Reflex 同进程有效，Ghost 进程 Speech 收不到。**当前日志确认**：`enable_topic=False` 在两个进程中都出现，走 HTTP 旁路，无字幕数据发布 | **方向 A**：确保 `--mode show` 运行，并修复 ConfigStore 模式覆盖链路（诊断 `HostEnvConfigStoreProvider.bootstrap()` → `search_config_infos_from_package`）。**方向 B（临时）**：在 `MOSS.modes.default.configs` 中重复定义 `subtitle_topic_config`，或在 `speech_service_provider.py` / `moss_in_reflex.py` 中硬编码 `enable_topic=True` | 2026-06-22 |
| 24 | Ghost 不调 `advance_point` → 叙述完第一段后静默 | 日志 `interpreter settled: compiled=0 done=1`：Ghost 只执行了 `__main__.__content__`（TTS 叙述），没有输出 `<apps.ui_moshi:advance_point />` CTML 命令。模型讲完第一个 talking point 即停止，不调 advance_point → 无 Signal → 永远不会醒来继续。context 指令（`context_messages.py:119-130`）已写明"每讲完一个要点后调 advance_point"，但 LLM 模型行为不可靠。第一轮测试（16:46）还发生了 `attention aborted during execute` + `RuntimeError: MossRuntime is not running`（MossRuntime 提前退出崩溃），第二轮未复现 | **方向 A**：增强 context 约束（如要求 Ghost 必须在同一轮输出的末尾包含 `<apps.ui_moshi:advance_point />`，否则视为违规）。**方向 B**：服务端增加超时兜底——TTS 播放完成后 N 秒若无 `advance_point` 调用，自动推进并发送 Signal。**方向 C**：检查 Ghost LLM 的原始输出，确认是模型没生成还是 CTML 解析器未识别 | 2026-06-22 |

## 架构教训

### IoC Container 是进程内对象，不跨进程共享

`TTSSpeechServiceProvider(singleton=True)` 的 singleton 作用域是单个 `IoCContainer` 实例内部——不是跨 OS 进程的。Reflex 进程和 Ghost 进程各自调用 `Host.discover()`，各自创建独立的 Host/Matrix/IoCContainer，各自在首次 `force_fetch(Speech)` 时触发 Provider 创建自己的 `BaseTTSSpeech` 实例。两个 Python 对象在不同的内存空间，互不影响。

**可跨进程工作的机制**：Zenoh Topic（`AudioRuntimeTopic`）— pub/sub 走网络协议，天然跨进程。
**不可跨进程工作的机制**：Python 函数指针（`subtitle_callback`）— 只在设置它的进程内存中有效。

**推论**：任何需要在 Ghost 进程中生效的运行时注入，必须在 Ghost 进程侧执行。Reflex 进程侧的 `matrix.container` 操作只影响 Reflex 进程自身。

Ghost 的职责是**叙述**，不是**编排**。布局选择、课程加载、章节导航——这些都是确定性操作，应放在服务端，不经过 Ghost 的理解层。把编排交给 Ghost 引入了三类问题：
1. 幻觉（选错布局、重复操作）
2. 时序依赖（Ghost 的 CTML 执行顺序不可靠）
3. 上下文污染（编排细节占用了叙述需要的 token）

原则：**确定性操作走 API，创造性工作走 Ghost。**

### ChannelCtx 的作用域

`ChannelCtx.container()` 只在 MOSS channel 命令和 context_messages 回调中可用。
HTTP handler（aiohttp）运行在裸 asyncio 上下文，需要在 `moss()` 初始化阶段捕获所需 IoC 引用。
这是一个通用模式——任何在 `moss()` 闭包内但不在 channel 命令链中的代码，都应使用捕获引用。

### CTML 命令 ≠ 回合延续

模型倾向于在 function call 后停止生成。CTML 命令返回的 `<command><result>` 是 percept 不是 Signal——它能被 Ghost 看到，但不能触发新的 attention 周期。要让 Ghost 在命令后继续工作，必须发 Signal 来驱动下一轮 mindflow。

模式：**命令执行 → 状态变更 → 发 Signal(Priority.NOTICE) → Ghost 醒来 → 读 context → 继续**。

参考实现：`.moss_ws/apps/genkits/image/main.py` 的 `emit_generation_signal`。

### Channel 短名 ≠ 运行时全名

`PyChannel(name="moshi")` 注册的短名在 app 体系下会被自动加上 `apps.<group>_<name>` 前缀。context 中引用的命令必须使用运行时全名，否则 Ghost 每次调命令都先吃一个 `INTERPRET_ERROR`。

## 下次会话起点

### 优先级 0：修复 Ghost 卡住（bug #24）

Ghost 叙述完第一段后不调 `advance_point`，整个讲课流程死锁。

**诊断步骤**：
1. 查看 Ghost LLM 的原始输出 — 确认模型是否生成了 `<apps.ui_moshi:advance_point />`。如果是，说明 CTML 解析器未识别；如果否，是 prompt 问题
2. 检查 `__content__` 命令执行后 Ghost 的 mindflow 状态 — `interpreter settled` 后应有新 attention 周期，但 `observe=False` 暗示没有
3. 对比 chat 模式（聊天时 Ghost 能正常调 `chat_reply`）与 teaching 模式的行为差异

**可能修复**：
- 增强 context 约束：明确要求 Ghost "本轮输出末尾必须包含 `<apps.ui_moshi:advance_point />`"
- 服务端超时兜底：TTS `running=False` 后 5s 无 `advance_point` → 自动推进 + 发 Signal
- 如果是 CTML 解析器问题：检查 `<apps.ui_moshi:advance_point />` 语法是否被正确识别

### 优先级 1：打通字幕链路（bug #23）

**短期可用路径**（绕过 ConfigStore bug）：
- 方案 A：在 `MOSS.modes.default.configs` 中加 `subtitle_topic_config = SubtitleTopicConfig(enable_topic=True)` — 1 行改动，让 default 模式也启用 Topic 字幕
- 方案 B：在 `speech_service_provider.py` 和 `moss_in_reflex.py` 的回退代码中，不仅尝试 `mode_name` 的 config，也尝试 `"show"` 的 config — 比方案 A 更脏

**正确路径**（修复 ConfigStore 模式覆盖）：
1. 添加诊断日志到 `HostEnvConfigStoreProvider.bootstrap()` — 确认 `search_config_infos_from_package("MOSS.modes.show.configs")` 是否找到 instance 类型的 `subtitle_topic_config`
2. 若未找到 → 修复扫描逻辑；若找到但未应用 → 修复 `set_config` 调用链
3. 验证：`moss --mode show manifests configs` 输出应包含 `subtitle_topic: SubtitleTopicConfig(enable_topic=True)`

### 优先级 2：ConfigStore 根因修复

独立 feature：ConfigStore 模式覆盖（`is_override=True` 的 ConfigType instance）不生效。影响范围超出字幕——任何在 mode configs 中定义的 instance 覆盖都无效。需要端到端追踪：`PackageManifests` 扫描 → `ConfigInfo(is_override=True)` → `HostEnvConfigStoreProvider.bootstrap()` → `set_config()`。

### 测试命令

```bash
# 正确命令（用 show mode）
moss-run-ghost echo --mode show

# 验证 ConfigStore
moss --mode show manifests configs | grep subtitle

# 日志关键字
grep -E "subtitle|advance_point|interpreter settled|回退导入" .moss_ws/runtime/logs/moss.log
```
