# 讲课模式完整技术方案

> **实施策略**：按 FEATURE.md 优先级——先单体跑通讲课流程（Phase 1-3），最后 Topic 桥接解耦（Phase 4）。每 Phase 产出可体验的端到端功能。

## 用户故事（完整流程）

1. 用户在模式选择页点击 **"讲课"**
2. 弹出课程列表，用户点击选择一门课
3. 页面切换到讲课模式：**全屏 Reflex iframe（白板） + 底部字幕区**
4. AI 自动开讲（TTS），字幕同步展示
5. 讲课时飞书群消息以**弹幕形式从右飘过**，AI 在飞书群回复（不打断讲课）
6. 人类**语音提问**→ AI 暂停讲课 → 回答语音问题 → 继续讲课
7. 全部讲完后，AI 总结所有问题（飞书群 + 语音打断）

## 核心设计：双重视图

Reflex 白板是「观众视图」，Ghost 看到的是「演讲者视图」——两者主题相同，内容不需要逐字对应。

| | 观众视图（Reflex 白板） | 演讲者视图（Ghost context） |
|---|---|---|
| **内容** | 标题 + 要点 + 图片（投屏用） | 每段演讲要点、过渡词、关键数据、预期时长 |
| **给谁看** | 听众/学员 | 主讲人（AI 或人类） |
| **数据来源** | `slide_data` | `speaker_notes` |
| **生成时机** | 备课阶段 AI 生成 | 备课阶段 AI 生成（与 slide_data 同时） |
| **练习模式映射** | — | 演讲者手卡（📌要点 / 🔄过渡 / 📊关键数据） |

### speaker_notes 的生成流程

在备课（preparing）阶段，Ghost 为每个章节生成内容时，同时产出两份数据：

1. **slide_data**（观众视图）— 投屏用的标题、要点、图片
2. **speaker_notes**（演讲者视图）— 讲课时 Ghost 自己的"手卡"

Ghost 调 `save_chapter` 时，slide_data 和 speaker_notes 一起持久化到章节 JSON 中。讲课阶段 Ghost 从章节数据中读取 speaker_notes 作为叙述指南。

**章节数据结构**：

```python
chapter_data:
  slide_data:      # 观众视图（已有，原 chapter_data 内容）
    sub_title: str
    main_text: str
    annotations: list[str]
    images: list[str]          # locator 列表

  speaker_notes:   # 演讲者视图（备课阶段 AI 生成，与 slide_data 同时持久化）
    talking_points: [
        {"id": "0", "text": "开场：欢迎各位", "status": "done"},
        {"id": "1", "text": "要点1：核心用户场景", "status": "done"},
        {"id": "2", "text": "要点2：小步实验验证", "status": "active"},
        {"id": "3", "text": "要点3：沉淀指标看板", "status": "pending"},
    ]
    # status: done=已讲完, active=正在讲（唯一）, pending=待讲
    transitions: list[str]     # 过渡词
    key_data: list[str]        # 关键数据
    estimated_duration: int    # 预计讲解时长（秒）
```

**段落驱动 + 间隙回复**：每个 talking_point 讲完后 Ghost 调 `advance_point` 标记完成。段落间隙调 `check_messages` 批量回复飞书消息。

**打断恢复**：context 中展示 `✓ done / → active / … pending` 的段落状态列表，Ghost 直接从 `→` 标记处继续。

**advance_point 兜底**：若 Ghost 未在合理时间内调 `advance_point`（当前 active 段落持续超过 estimated_duration × 1.5），LectureBrain 自动将当前段落标记为 done 并推进到下一段，同时向 Ghost context 注入提示"上一段已超时自动推进"。

## 架构概览

### Phase 1-3：单体架构（当前阶段）

Moshi 和 Reflex 在同一进程内，通过内部 Queue + 命令直接通信。讲课状态机、TTS 控制、字幕生成均在 moshi Channel 内完成。

```
┌─ 浏览器 ──────────────────────────────────────────────────────────────┐
│  ModeSelect → CourseSelect → LecturePage(iframe + subtitle + danmaku) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP + SSE (/api/chat, /api/chat/stream,
                               │             /api/courses, /api/mode)
┌─ main.py (:9731) ──────────────────────────────────────────────────────┐
│  HTTP proxy：不碰 Matrix                                               │
│  路由：/api/courses（课程列表）、/api/mode（发 Signal）                  │
│  新增 SSE 频道：/api/chat/stream/subtitle、/api/chat/stream/danmaku    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP POST :9733/_internal/*
┌─ Moshi + Reflex（同一进程）────────────────────────────────────────────┐
│  Channel "moshi"：课程逻辑 + 讲课控制 + 聊天 + 渲染                     │
│  LectureBrain：讲课状态机（idle→loading→lecturing→paused→ended）       │
│  TTS 控制 → 字幕文本 → SSE 推送 Subtitle                               │
│  飞书群消息 → 弹幕 → SSE 推送 Danmaku                                  │
│  语音打断 ← AudioSignal → pause → answer → resume                      │
│                                                                        │
│  内部通信（Queue，不经过 Matrix）：                                     │
│    _switch_to_chapter → QUEUE.put(LayoutEvent) → Reflex 渲染           │
│    Reflex 页面变更 → LayoutSnapshot → save_chapter 读取                │
│    Subtitle/Danmaku → _SSE_QUEUE → main.py pipe → 浏览器              │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 4：Topic 桥接解耦（未来）

```
┌─ 浏览器 ──────────────────────────────────────────────────────────────┐
│  ModeSelect → CourseSelect → LecturePage(iframe + subtitle + danmaku) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP + SSE
┌─ main.py (:9731) ──────────────────────────────────────────────────────┐
│  HTTP proxy：不碰 Matrix                                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP POST :9733/_internal/*
┌─ Moshi Cell（课程逻辑）────────────────────────────────────────────────┐
│  Channel "moshi"：课程逻辑 + 讲课控制 + 聊天                            │
│  LectureBrain：讲课状态机                                               │
│  TTS 控制 → pub SubtitleTopic                                          │
│  飞书群消息 → pub DanmakuTopic                                         │
│                                                          │             │
│                   Topic Bridge（不经过 Ghost）             │             │
│  ┌────────────────────────────────────────────────────────┼──┐          │
│  │ pub ChapterLoadTopic → Reflex 渲染章节                  │  │          │
│  │ sub PageSnapshotTopic ← Reflex 页面快照                  │  │          │
│  │ pub SubtitleTopic → Reflex 字幕                          │  │          │
│  │ pub DanmakuTopic → Reflex 弹幕                           │  │          │
│  │ pub LectureStateTopic → Reflex 状态同步                   │  │          │
│  └────────────────────────────────────────────────────────┼──┘          │
└───────────────────────────────────────────────────────────┼────────────┘
                                                            │ Matrix/Zenoh
┌─ Reflex Cell ──────────────────────────────────────────────────────────┐
│  Channel "reflex"：纯白板渲染                                          │
│  topic_window(ChapterLoadTopic) → 渲染章节                             │
│  topic_window(SubtitleTopic) → 字幕展示                                │
│  topic_window(DanmakuTopic) → 弹幕动画                                  │
│  pub PageSnapshotTopic（debounce 500ms）                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Topic 定义（Phase 4 启用，Phase 1-3 用内部 Queue 等效实现）

### 1. ChapterLoadTopic — Moshi → Reflex（章节渲染）

```python
class ChapterLoadTopic(TopicModel):
    course_name: str
    chapter_index: int = 0
    total_chapters: int = 0
    chapter_data: dict   # slide_data 部分（观众视图）
    topic_type → "moshi/chapter_load"
```

Phase 1-3 等价实现：`_switch_to_chapter()` → `QUEUE.put(LoadChapterEvent)`。

### 2. PageSnapshotTopic — Reflex → Moshi（页面快照）

```python
class PageSnapshotTopic(TopicModel):
    layout: str
    fields: dict   # {title, sub_title, main_text, annotations_count, images_count}
    topic_type → "moshi/page_snapshot"
```

Phase 1-3 等价实现：`LayoutSnapshot.refresh()` 读取当前 Reflex State。

### 3. SubtitleTopic — Moshi → Reflex（字幕同步）

```python
class SubtitleTopic(TopicModel):
    text: str                # 当前字幕文本
    chapter_index: int = 0
    is_final: bool = False   # True=本段讲完, False=流式增量
    topic_type → "moshi/subtitle"
```

Phase 1-3 等价实现：写入 `_SUBTITLE_QUEUE`，SSE 推送到浏览器。

### 4. DanmakuTopic — Moshi → Reflex（弹幕）

```python
class DanmakuTopic(TopicModel):
    sender: str              # 发送者昵称
    text: str                # 弹幕文本
    color: str = "white"     # 颜色标签（white/green/yellow/red）
    source: str = "feishu"   # feishu / system
    topic_type → "moshi/danmaku"
```

Phase 1-3 等价实现：写入 `_DANMAKU_QUEUE`，SSE 推送到浏览器。

**弹幕频率控制**：飞书群消息先入缓冲窗口（最近 50 条），按到达时间戳均匀发射。若缓冲堆积超过 20 条，丢弃最旧的非@消息，保留 @消息优先展示。

### 5. LectureStateTopic — Moshi → Reflex（讲课状态）

```python
class LectureStateTopic(TopicModel):
    state: Literal["loading", "lecturing", "paused", "ended"]
    course_name: str = ""
    chapter_index: int = 0
    total_chapters: int = 0
    topic_type → "moshi/lecture_state"
```

## 讲课状态机（LectureBrain）

```
idle → loading → lecturing ⇄ paused → ended
                            ↑
                    语音打断或人类控场
```

| 状态 | 行为 |
|------|------|
| idle | 等待开始讲课指令 |
| loading | load_course + 渲染首章（slide_data 推送白板，speaker_notes 进 context） |
| lecturing | TTS 播放 + 字幕同步；Ghost 按 speaker_notes 逐要点讲，讲完后自动翻页 |
| paused | TTS 暂停；Ghost 回答问题；恢复时重新读 speaker_notes，自行判断从哪继续 |
| ended | TTS 停止，总结飞书+语音问题 |

**核心数据**：

```python
class LectureBrain:
    status: str = "idle"
    current_course: str = ""
    current_chapter: int = 0
    points: list[dict] = []       # 当前章节 talking_points（从 speaker_notes 加载）
    questions: list[dict] = []    # [{source:"voice"/"feishu", text:"...", sender:"..."}]
    point_started_at: float = 0   # 当前 active 段落开始时间（用于超时检测）
```

**新命令**：

```python
# advance_point — 标记当前 active→done，下一个 pending→active
# 全部 done 时返回 "chapter_done"（Ghost 据此翻页）
# 兜底：若超过 estimated_duration * 1.5 未调用，LectureBrain 自动推进
<moshi:advance_point />

# check_messages — 读 feishu_window，有消息就回复，返回消息数
# Ghost 在段落间隙调用
<moshi:check_messages />
```

## 前端页面流

```
┌─ 模式选择页（/）───────────────────────────────────────┐
│  ┌─ TopBar ───────────────────────────────────────┐    │
│  │  [备课]  [练习]  [讲课]                         │    │
│  └────────────────────────────────────────────────┘    │
│  ┌─ 内容区 ───────────────────────────────────────┐    │
│  │  默认：显示 Logo/提示文字                        │    │
│  │  点击"讲课"后→ 弹出课程选择列表                   │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
                        │ 点击具体课程
                        ▼
┌─ 讲课页面（/lecture?course=xxx）───────────────────────┐
│  ┌─ Reflex iframe（全屏白板）─────────────────────┐    │
│  │  :3000，嵌入，占满整个页面                        │    │
│  │  弹幕层：文字从右飘过，覆盖在白板上方              │    │
│  └──────────────────────────────────────────────┘    │
│  ┌─ 字幕区（底部，半透明背景）─────────────────────┐    │
│  │  AI 主讲字幕，TTS 所说文字同步展示                 │    │
│  └──────────────────────────────────────────────-┘    │
│  ┌─ 退出按钮（右下角，3s 无操作半透明）──────────────┐  │
│  │  ✕  仅此一个操作入口                               │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

前端基于现有 `static/teachingL0.html` 改造——该文件已有弹幕 CSS 动画和演讲者手卡 UI 组件，可直接复用。

## TTS 音频通道

TTS 音频不经过浏览器——由服务端 Volcengine TTS 直接合成播放（通过 MOSS TTSManager + AudioPlayer provider）。浏览器端通过字幕同步跟进内容。

流程：
```
Ghost 生成叙述文本 → <moshi:speak text="...">
  → TTSManager.synthesize(text) → AudioPlayer 播放音频
  → 每句播放时 pub SubtitleTopic（或写 SSE queue）→ 浏览器字幕更新
```

如需在浏览器端播放音频（例如远程听课场景），可在 Phase 2 评估增加音频流推送，但当前以服务端播放 + 浏览器字幕同步为主方案。

## 交互数据流

### 流 A：开讲（Phase 1-3 走 Queue，Phase 4 走 Topic）

```
前端 POST /api/mode {mode:"lecture", course:"xxx"}
  → main.py → :9733/_internal/course/start_teaching
  → LectureBrain.status = "loading"
  → CourseManager.load_course() + start_teaching()
  → _switch_to_chapter(0) → QUEUE.put(LoadChapterEvent) → Reflex 渲染
  → speaker_notes 注入 Ghost context
  → Ghost 开始叙述 → TTS 启动
```

### 流 B：段落推进 + 翻页

```
Ghost 即兴叙述当前 active 要点 → TTS 播放 → 字幕同步
Ghost 认为本段讲够了：
  → 调 <moshi:check_messages />
    → 有飞书消息就回复，没有就跳过
  → 调 <moshi:advance_point />
    → LectureBrain：active→done，下一段→active
    → 如本章全部 done：自动翻页（Ghost 不参与翻页决策）
    → 返回 "point_advanced" / "chapter_advanced" / "lecture_ended"
  → Ghost 继续讲下一段（或结束）

兜底：若 advance_point 超时未调用（estimated_duration * 1.5），
      LectureBrain 自动推进并在 context 注入提示。
```

**翻页决策权在 LectureBrain，不在 Ghost**。`advance_point` 内部判断是否翻页：

```python
# advance_point 内部逻辑
if 下一段 is None（本章全部 done）：
    if 还有下一章：
        _switch_to_chapter(current + 1)  # Phase 4: pub ChapterLoadTopic
        LectureBrain.current_chapter += 1，加载新的 talking_points
        return "chapter_advanced"
    else：
        LectureBrain.status = "ended"
        return "lecture_ended"
else：
    return "point_advanced"
```

### 流 C：字幕同步

```
TTS 播放每个句子：
  → 写 SSE queue：SubtitleTopic(text="...", is_final=False)   # 流式增量
  → 写 SSE queue：SubtitleTopic(text="...", is_final=True)    # 本句结束
  → main.py SSE pipe → 浏览器 → 更新字幕区文本
```

### 流 D：飞书弹幕 + 段落间隙回复

```
飞书群消息到达 → Feishu Channel → Moshi 消费
  → 写 SSE queue：DanmakuTopic（弹幕，即时显示，不经过 Ghost）
  → 飞书消息存入 feishu_window
  → 弹幕频率控制：缓冲最近 50 条，超 20 条堆积时丢弃最旧非@消息

Ghost 讲完一个要点（advance_point 后，翻页前）：
  → 调 <moshi:check_messages />
  → 读 feishu_window → 批量回复 → 继续讲
```

### 流 E：语音打断 + 恢复

```
用户语音输入 → ASR → AudioSignal(SPEECH_FINAL, priority=WARNING)
  → 当前讲课 Attention 被 abort
  → LectureBrain.status = "paused"
  → LectureBrain.questions.append({source:"voice", text:"..."})
  → TTS 停止（当前 active 段落保持 active）

  → Ghost 新 Attention：回答问题
    → context 包含：
        段落进度：
          ✓ 要点1：核心用户场景
          → 要点2：小步实验验证  ← 继续这个
          … 要点3：沉淀指标看板

        已播字幕（当前 active 段落的最后 5 句）：
          1. "我们来谈谈增长策略中最重要的实验方法"
          2. "核心在于用最小成本验证最大假设"
          3. "举个例子，某电商平台通过A/B测试..."

    → Ghost 回答问题

  → Ghost 调 <moshi:resume_lecture>
    → context 同样提供段落进度 + 已播字幕
    → "请从第 3 句之后继续讲，不要重复已讲内容"
    → Ghost 从断点后继续即兴叙述
    → TTS 恢复 → 字幕同步
```

**恢复信息全部由已有数据派生，不额外存状态**：
- 段落进度：读 `talking_points[*].status`（`done` / `active` / `pending`）
- 已播字幕：读 SubtitleTopic 窗口最近 5 条

### 流 F：讲课结束 + 总结

```
Ghost 讲完最后一章
  → LectureBrain.status = "ended"
  → 收集所有飞书问题 + 语音打断问题
  → 生成总结（通过 chat_reply 返回给人类）
```

## 实施步骤

### Phase 1：单体讲课骨架（产出：可从头讲到尾）

| # | 内容 | 涉及文件 |
|---|------|---------|
| 1.1 | 扩展章节数据结构：chapter_data 增加 speaker_notes 字段 | `course_manager.py`, `course_storage.py` |
| 1.2 | 实现 LectureBrain 状态机（idle→loading→lecturing→paused→ended） | `moss_in_reflex.py`（新建 `lecture_brain.py`） |
| 1.3 | 新增命令：`start_teaching`、`advance_point`、`check_messages`、`resume_lecture` | `moss_in_reflex.py` |
| 1.4 | 课程列表 API 对接 `CourseStorage.list_infos()` | `main.py` |
| 1.5 | 前端讲课页面（基于 `teachingL0.html` 改造：iframe + 字幕区 + 弹幕层 + 退出按钮） | `static/teachingL0.html` 或新建 |
| 1.6 | Ghost context 更新：teaching 模式下注入 speaker_notes + 段落进度 | `context_messages.py` |
| 1.7 | 字幕推送通道：SSE `/api/chat/stream/subtitle` | `main.py`, `moss_in_reflex.py` |

### Phase 2：TTS + 字幕 + 弹幕（产出：有声音有互动）

| # | 内容 |
|---|------|
| 2.1 | 接入 MOSS TTS 系统（Volcengine TTSManager），新增 `speak` 命令 |
| 2.2 | TTS 每句字幕 → SSE subtitle 推送 → 浏览器字幕同步 |
| 2.3 | 弹幕推送通道：SSE `/api/chat/stream/danmaku` |
| 2.4 | 飞书消息 → 弹幕（频率控制：缓冲 50 条，堆积 20+ 丢弃非@旧消息） |
| 2.5 | 段落间隙飞书回复（check_messages 实际对接 feishu channel） |

### Phase 3：语音打断 + 总结（产出：完整讲课体验）

| # | 内容 |
|---|------|
| 3.1 | Audio Signal 打断流程：识别 SPEECH_FINAL → LectureBrain pause |
| 3.2 | 打断后恢复：context 注入段落进度 + 已播字幕窗口 |
| 3.3 | 讲课结束总结：收集 questions → Ghost 生成总结 → chat_reply 返回 |
| 3.4 | 断线重连恢复：重连时从 LectureBrain 当前状态 + 当前章节恢复 |

### Phase 4：Topic 桥接解耦（产出：架构清洁）

| # | 内容 |
|---|------|
| 4.1 | 定义 5 个 Topic Model（ChapterLoad / PageSnapshot / Subtitle / Danmaku / LectureState） |
| 4.2 | Reflex Cell 改造：去掉课程逻辑，只保留渲染 + Topic 消费 |
| 4.3 | Moshi Cell 改造：去掉 Reflex 渲染代码，改用 Topic 发布 |
| 4.4 | save_chapter 改用 PageSnapshotTopic TopicWindow |
| 4.5 | _switch_to_chapter 改用 ChapterLoadTopic.pub() |
| 4.6 | Subtitle/Danmaku 改走 Topic，Reflex cell 内 topic_window 消费后 SSE 推送 |

## 不改动的部分

- main.py HTTP proxy 基础结构（Phase 1-3 沿用，Phase 4 可能微调）
- Signal 对话环路（Phase 1 已完成）
- CourseManager / CourseStorage
- context_messages 模式感知机制
- Reflex 布局系统（Layout、EventModel、build()）

## Phase 1 实现记录（2026-06-21）

### 已实现

| 组件 | 文件 | 说明 |
|------|------|------|
| 章节数据结构扩展 | `course_manager.py` | `save_chapter()` 新增 `speaker_notes` 参数；`get_speaker_notes()` |
| LectureBrain 状态机 | `lecture_brain.py`（新文件） | idle→loading→lecturing⇄paused→ended，超时检测，问题收集 |
| 讲课命令 | `moss_in_reflex.py` | `advance_point`（段落推进+自动翻页）、`check_messages`（桩）、`resume_lecture` |
| 课程列表 API | `moss_in_reflex.py` + `main.py` | `/_internal/courses` → `POST /api/courses` proxy |
| 讲课启动 API | `moss_in_reflex.py` + `main.py` | `/_internal/lecture/start`（原子编排）→ `POST /api/lecture/start` proxy |
| 前端讲课页面 | `teachingL0.html` | Reflex iframe + 字幕区 + 弹幕层 + 课程选择弹窗 + SSE |
| Ghost context | `context_messages.py` | teaching 模式展示 speaker_notes + 约束（不切布局、不重复 load）；preparing 模式提示生成 speaker_notes |

### 架构决策

1. **编排权归服务端**：`/_internal/lecture/start` 原子完成 load + 布局 + 渲染 + 状态机初始化 + Signal，不经过 Ghost 理解层。
2. **ChannelCtx 作用域**：aiohttp HTTP handler 无 ChannelCtx。在 `moss()` 初始化阶段捕获 `_course_storage`、`_registry` 等引用供内部 handler 使用。
3. **Signal 唤醒 Ghost**：编排完成后用 `InputSignal`（非 Topic）通知 Ghost。Topic 不能唤醒 Ghost——这是 sensors 参考实现验证过的模式。
4. **advance_point 后发 Signal 驱动连续叙述**：模型倾向于在 function call 后停止生成，COMMAND-RESULT 是 percept 不是 Signal，不能触发新 attention 周期。参考 `.moss_ws/apps/genkits/image` 的 `emit_generation_signal` 模式，`advance_point` 执行完毕后发 `InputSignal(Priority.NOTICE)`，携带当前 active 段落文本。`lecture_ended` 不发 Signal。
5. **context 命令引用用运行时全名**：`PyChannel(name="moshi")` 在 app 体系下注册为 `apps.ui_moshi`，context 中所有 CTML 命令引用必须用全名。
6. **teaching 模式隔离聊天功能**：`chat-instruction` 仅在 idle/discussing/preparing 模式注入，teaching 模式不注入——防止 Ghost 在讲课中自言自语调 `chat_reply`。

### Phase 1 调试实录（2026-06-21 第二轮）

在端到端测试中发现并修复了以下问题，Phase 1 核心环路（叙述→推进→翻章→结束）最终跑通：

| # | 问题 | 修复 |
|---|------|------|
| 1 | channel 名不匹配（`moshi` vs `apps.ui_moshi`）→ 每次命令调用先报错 | context 全部改为 `apps.ui_moshi`（6 处） |
| 2 | context-mode 与 lecture-state 重复展示 talking_points | 移除 context-mode 的 points 列表，只保留 lecture-state 的实时进度 |
| 3 | chat-instruction 对所有模式生效 → Ghost 讲课中自言自语 | `if _COURSE_MGR.mode != "teaching"` 守卫 |
| 4 | Ghost 输出 `<_>` 做段落分隔 → CTML 解析器崩溃 | context 加 CTML 卫生约束 |
| 5 | Ghost 调 advance_point 后静默 → function call 是回合终点 | advance_point 末尾发 InputSignal(Priority.NOTICE) |
| 6 | lecture_ended 后 Ghost 继续调命令 → 报 UNKNOWN_ERROR | advance_point 对 ENDED 状态幂等保护；context 加停止规则 |
