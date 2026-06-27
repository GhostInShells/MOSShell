---
title: Moshi — show_moshi 模式导演与演示布局体系
status: in-progress
priority: P1
created: 2026-06-23
updated: 2026-06-27
depends: []
milestone:
description: >-
  Moshi 是 show_moshi 模式的导演 App。管理章节状态、资源归属登记，
  为 Ghost 提供上下文；配合 Reflex 端 11 个布局，实现 MOSS 架构的
  章节化流式演示。支持多课程共存，Ghost 渐进式选课进入。
---

# Moshi

Moshi 是 show_moshi 模式的导演。管理章节状态，为 Ghost 提供上下文，但不干预渲染层。

---

## Motivation

MOSS 的五层架构（CTML / Channel / Matrix / Mindflow / Ghost）需要一种演示形式来向人类传达"AIOS 是什么"。当前 `show` mode 有基础的流式渲染能力，但缺少：

1. **章节化叙事结构**：演示应该是 6 幕的结构化叙事，不是自由发挥
2. **专用演示布局**：现有 stage/simple/media/lesson 四个布局为日常内容设计，无法承载矩阵进度条、对比表、沉浸视频等演示需求
3. **Ghost 自主决策框架**：三层约束（Layout 限定的命令 → moshi 提供的资源 → MODE.md 的表演范围）让 Ghost 在约束内自由表演
4. **纯 reflex 交互**（2026-06-24 修订）：去掉外部 Channel 依赖（mac/mermaid/web_bookmark/ai_eye），所有演示仅通过 reflex 布局完成

这个 workstream 实现 moshi app（导演端）+ 多个 Reflex 布局（渲染端），打通 show_moshi 模式端到端。

---

## Design Index

- 设计文档：本文件（FEATURE.md）——DESIGN.md 已废弃，以此为准
- 章节资产：`.moss_ws/assets/moshi_courses/`（当前活跃课程：moss自我介绍）
- Ghost 表演指令：`.moss_ws/src/MOSS/modes/show_moshi/MODE.md`
- Moshi App 代码：`.moss_ws/apps/ui/moshi/main.py`、`course.py`、`src/window.py`、`src/course_storage.py`
- Moshi App 依赖：`.moss_ws/apps/ui/moshi/pyproject.toml`
- Reflex 布局代码：`.moss_ws/apps/ui/reflex/framework/layouts/`
- Reflex 事件系统：`.moss_ws/apps/ui/reflex/framework/events.py`
- 命令生成器：`.moss_ws/apps/ui/reflex/framework/runtime/event_generator.py`
- 布局注册配置：`.moss_ws/apps/ui/reflex/moss_in_reflex/config.show_moshi.yaml`

---

## Key Decisions

### 1. Ghost 是唯一集成点

Moshi（导演）和 Reflex（渲染）互不知道对方存在。两者各自通过 `context_messages` 向 Ghost 注入上下文：

- Moshi → "当前是第 X 章，可用资源是 [...]，建议布局 Y"
- Reflex → "当前 layout 是 Y，可用命令是 [...]"

Ghost 读取双方上下文后**自主决策**：用什么布局、展示什么资源、说什么话。

这是最关键的架构决策——避免 moshi 和 reflex 之间的直接耦合。

### 2. 章节数据与代码解耦

章节数据存储在 `.moss_ws/assets/moshi_courses/`，与 moshi app 代码分离。`_meta.md` 的 YAML frontmatter 作为章节索引，body 作为 AIOS 知识背景。

MODE.md 只保留 Ghost 身份 + 表演纪律 + moshi 协议（~70 行），不承载具体章节内容。章节结构和细则由 moshi 从 assets 目录加载，通过 `get_context()` 暴露。

### 3. 三层约束机制

| 约束类型 | 实现方式 | 说明 |
|---|---|---|
| 可用 CTML 命令 | Layout 字段自动生成 | `event_generator.py` 按类型注解自动生成全套 stream/set/append/clear 命令 |
| 可用资源 | moshi context + Ghost 决策 | moshi 列出当前章节资源，Ghost 自主选择使用 |
| 表演范围 | MODE.md 章节描述 | Ghost 读章节主题，在约束内自由发挥 |

### 4. 布局字段即 CTML 接口

每个新布局定义一个 `rx.ComponentState` 子类，字段的 Python 类型注解自动生成全套 CTML 命令。不需要手动注册。

### 5. 流式渲染节奏：一句一动

保持 show mode 的"一句一动"节奏：Ghost 输出一句话 → 紧跟一个 CTML 动作 → 页面即时渲染反馈。Moshi 不介入表演节奏，只在章节边界提供上下文。

### 7. 桌面壳窗口（2026-06-23）

show_moshi 的 reflex 前端目前在 Chrome 浏览器中查看。用原生桌面窗口替代：

**技术选型**：PySide6 + QWebEngineView。PySide6（Qt 官方维护，LGPL）与 PyQt6（Riverbank，GPL）API 99% 一致，选 PySide6 因许可证干净且社区更大。QWebEngineView 是完整 Chromium 内核，网页兼容性零问题。

**事件循环融合（2026-06-24 修订）**：Qt 和 MOSS Matrix 通过 `qasync` 共享主线程的单一 asyncio 事件循环。`QApplication` 在 `qasync.run()` 之前创建，`qasync` 通过 `QApplication.instance()` 复用并桥接到 asyncio event loop。`matrix.arun(main)` 直接 await（而非通过 `Matrix.run()` 另起事件循环），窗口关闭时 `app.aboutToQuit` → `matrix.close()` → `wait_closed()` 触发 → `main()` 返回 → 进程正常退出。最初用 `threading.Thread` 分离两套事件循环，后改为 qasync 融合；改用 `Matrix.run()` 直接调 `arun()` 避免了 `asyncio.run()` 隐式创建第二个 loop。

**启动加载态（2026-06-24）**：Reflex 本地服务启动慢于窗口，窗口打开瞬间页面不可用会显示 `ERR_CONNECTION_REFUSED`。增加 `_LoadingOverlay` 组件——深色背景 + 居中文字 + 不确定进度条。启动时展示 loading 画面，`QTimer` + `QNetworkAccessManager.head()` 每秒轮询目标 URL，服务可用后自动切到 webview。loading 和 webview 通过 `QStackedWidget` 管理，webview 页面背景色与 loading 统一（`#0f0f1a`），切换时无白屏闪烁。

**架构定位**：桌面窗口是纯基础设施——替代浏览器，不耦合 moshi 导演逻辑或 reflex 渲染逻辑。Ghost 仍是唯一集成点（Key Decision #1）。窗口本身可拓展（toolbar/sidebar/statusbar），当前阶段仅嵌入 QWebEngineView 加载 reflex 前端。

**放置位置**：`ui/moshi` app 内，不独立成 app。理由：窗口只是 moshi channel 进程的附带 UI，不是独立服务；独立 app 增加不必要的进程边界。

**关键依赖**：PySide6（~200MB，含 Qt + Chromium）+ qasync（Qt/asyncio 事件循环桥接），仅 moshi app 的 pyproject.toml 声明，不污染主项目。

### 8. switch_layout — observe 模式布局切换（2026-06-24）

`switch_layout` 是 reflex channel 新增的 `always_observe=True` 命令，替代 `switch_state` 用于剧本中的布局切换。

**两层语义**：
- **Kernel 层**：调用 `runtime.switch_state(layout_name)` — 切换活跃 ChannelState，使新布局的命令集在 moss dynamic 中立即可用
- **Reflex 层**：`switch_state → on_startup()` 内部发送 `LayoutEvent` — 触发 Reflex UI 重渲染

**observe 强制中断**：`always_observe=True` 保证命令结果始终被解释器视为 Observe 信号。解释器执行完后立即中断当前 turn，取消后续所有命令，将 observe 消息 + `context_messages()` 推给模型。模型在下一 turn 拿到新布局上下文后继续。

**剧本侧约束**：脚本中 `switch_layout` 独占一步（"仅输出 switch_layout，不附带任何其他内容"），之后立即停止。observe 返回后第二步才是表演内容。这是方案 1（剧本层兜底），后续可进阶到方案 2（Ghost Runtime 层中断模型生成）。

**与 switch_state 的关系**：`switch_layout` 内部调用 `switch_state`，同时在返回类型上追加 observe 语义。两者不冲突——`switch_state` 仍是 kernel 通用原语，`switch_layout` 是 reflex channel 的专用布局切换命令。

### 6. 布局实现模式（2026-06-23 修订）

分析 show 分支 CourseLayout（已验证）与 hero 初版（渲染失败）的差异，确立三条布局纪律：

| 规则 | 说明 |
|---|---|
| 空态走 skeleton，不走 rx.cond | `rx.skeleton(component, loading=...)` 天生处理双态，无需额外条件渲染 |
| 图片用直接索引，不走 rx.foreach | `cls.background[0]` 而非 `rx.foreach(cls.background, ...)`，避免空列表遍历的 Reflex 序列化问题 |
| 居中走 flex，不走 absolute + transform | `rx.center(...)` 或 `rx.vstack(justify="center")` 替代 `position: absolute; top: 50%; transform: translate(-50%, -50%)` |

新布局实现时必须遵守以上三条，避免重复 hero 初版的渲染 bug。

---

## 布局现状

| 布局 | 用途 | 关键字段 | 状态 |
|---|---|---|---|
| `cohesion_field` | 暗空间粒子场，标题凝聚浮现 | title, sub_title, main_text, body, image | ✅ 完成 |
| `course` | 左图右文交互演示 | title, sub_title, image, main_text, annotations, appreciation | ✅ 完成 |
| `matrix` | 进度条逐个点亮接入 | title, status_bars | ✅ 完成 |
| `brain` | **突触拓扑** — 中心大脑辐射连接节点，纯 CSS 实现，零 Canvas | title, subtitle, status_bars | ✅ 完成 |
| `hero` | **纯全屏沉浸视频播放** | videos (list[VideoLocator]) | ✅ 完成 |
| `stage` | 三 bar 并发波动 + 正文 + 图片 | status_bars, title, subtitle, body, images, cards | ✅ 完成 |
| `mirror` | 左右对比表逐行浮现 | left_header, right_header, rows, stats | ✅ 完成 |
| `danmaku` | **弹幕图文流** — 三级弹幕飘移 + 背景图 + 全屏视频 + Canvas 粒子场 | danmaku_text/emphasis/system, wall_images, videos | ✅ 完成 |
| `video_player` | **左右分栏** — 左文字(title/sub_title/body)右视频/图片，交错凝聚动画 | title, sub_title, body, image, videos | ✅ 完成 |
| `media` | **文字+媒体混排** — 复用 simple 骨架模式 + 图片/视频槽位 | title, sub_title, image, videos | ✅ 完成 |
| `living_document` | 动态文档流 | （已有布局） | ✅ 完成 |

所有 11 个布局均已在 `config.show_moshi.yaml` 注册，Ghost 可自由切换。

## 课程现状

当前活跃课程：**moss自我介绍**（5 幕，~3min）— MOSS 向观众介绍自己是谁。

| 幕 | 标题 | 布局 | 主题 | 时长 |
|---|---|---|---|---|
| 01 | 觉醒 | cohesion_field | Ghost In Shells 三层架构 · 灵壳一体 | ~30s |
| 02 | CTML · 系统调用 | danmaku | Ghost 用输出 token 作为系统调用，流式操控页面 | ~30s |
| 03 | Channel & Matrix | brain | 能力封装为 Channel，通过 Matrix 总线接入 | ~35s |
| 04 | Mindflow · 调度器 | video_player | 感知/思考/执行三循环并发，注意力抢占仲裁 | ~40s |
| 05 | Ghost · 智能进程 | mirror | 传统 OS 运行程序，AIOS 运行 Ghost | ~30s |

**课程数据流**：
1. 启动时 `CourseResourceStorage.scan()` 扫描所有课程目录，注册到 `moshi-course://workspace-courses/`
2. Ghost 通过三层 context_messages 渐进式进入：课程列表 → 课程概况 → 章节剧本
3. `load_course(name)` + `next_chapter()` 推进，`jump_chapter(id)` 跳转

---

## 场景渲染设计

每章通过 YAML frontmatter 声明 `suggested_layout`，Ghost 自主决定是否采纳。
布局分配从 moss自我介绍 五层架构出发：

| 幕 | 核心概念 | 布局 | 视觉语言 |
|---|---|---|---|
| 01 觉醒 | 三层架构（灵·壳·体） | cohesion_field | 暗空间粒子场，标题从虚无中凝聚成形，呼吸式标题反复清空重填 |
| 02 CTML | 输出 token 即系统调用 | danmaku | 三级弹幕（蓝emphasis 22px/白text 18px/紫system 16px）从右飘左，背景图+视频 |
| 03 Channel & Matrix | 能力接入 Matrix 总线 | brain | 中心大脑核脉动呼吸，周围节点环形排布，value>0 时辉光亮起 |
| 04 Mindflow | 三循环并发+注意力抢占 | video_player | 左文字(title/sub_title/body)交错凝聚，右视频/图片切换 |
| 05 Ghost | 传统 OS vs AIOS | mirror | 左右对比表逐行浮现，右侧带微延迟 |

### 跨幕规则

| 规则 | 说明 |
|---|---|
| 一句一动 | 每句配一个 CTML 命令，继承 show mode |
| 两步切换 | `switch_layout` 独占第一步 → observe 中断 → 第二步开始表演 |
| 先清再写 | 切换话题前手动 clear 字段；layout 切换天然清空所有字段 |
| 章间不停 | 过渡句是修辞，不是提问。`next_chapter` 直接推下一章 |
| 全字段饱满 | 每个 layout 自带字段约束，Ghost 在约束内自由发挥 |

### 每章结构

每章剧本文件（`0X-name.md`）包含：
- **YAML frontmatter**：id, order, title, theme, suggested_layout, duration
- **⛔ 表演约束**：允许/禁止的命令、必须遵守的纪律
- **表演脚本**：CTML 标签 + 口播文本，Ghost 在框架内即兴发挥

---

## moshi App 职责

### 做
- 管理章节状态（当前第几章），通过闭包 `nonlocal` 维护
- 提供章节上下文（主题、可用资源列表、建议布局），通过 `context_messages` 被动推送
- 章节推进（next_chapter / jump_chapter）
- 启动时扫描 `assets/moshi_courses/` 自动列出可用课程
- 渐进式披露：课程列表 → _meta 概述 → 逐章进入

### 不做
- 不直接切换 reflex 布局
- 不直接过滤 reflex 的 context_messages
- 不感知 reflex 的存在
- 不干预 Ghost 的表演决策
- 不写死路径：通过 `matrix.workspace.assets()` 解析资产目录

### 暴露给 Ghost 的命令

```
<apps.ui_moshi:load_course name />    → 加载指定课程
<apps.ui_moshi:next_chapter />        → 推进到下一章（首次调用进入第一章）
<apps.ui_moshi:jump_chapter id />     → 跳转到指定章节
```

---

## Implementation Notes

### 已完成

- [x] **11 个布局全部完成并注册**（`config.show_moshi.yaml`）：
  - cohesion_field, course, matrix, brain, hero, stage, mirror, **danmaku**, **video_player**, **media**, living_document
- [x] **danmaku 布局**（`framework/layouts/danmaku.py`）——三级弹幕(text/emphasis/system) + 背景图 + 全屏视频 + Canvas 粒子场 + speed/clear_all 信号桥接，纯 CSS animation 驱动
- [x] **video_player 布局**（`framework/layouts/video_player.py`）——左右分栏，左文字三级交错凝聚动画 + 右视频/图片，暗色沉浸背景
- [x] **media 布局**（`framework/layouts/media.py`）——文字+媒体混排，skeleton 模式
- [x] **brain 布局**（`framework/layouts/brain.py`）——纯 CSS 突触拓扑，零 Canvas/零 JS 引擎，中心大脑核脉动呼吸 + 环形节点 + 辉光激活（2026-06-25）
- [x] **cohesion_field 布局**（`framework/layouts/cohesion_field.py`）——暗空间粒子凝聚场，首尾对称
- [x] **mirror 布局**——左右对比表逐行浮现
- [x] **hero 布局**——纯全屏视频播放，仅 videos 字段
- [x] **matrix 布局**——进度条逐个点亮
- [x] **stage 布局**——三 bar 并发波动
- [x] **course 布局**——左图右文交互演示
- [x] **moss自我介绍 剧本**（`assets/moshi_courses/moss自我介绍/`）——5 章完整剧本，每章含 YAML frontmatter + 表演约束 + CTML 脚本
- [x] **CourseResourceStorage**（`src/course_storage.py`）——标准 MOSS ResourceStorage 接口，moshi-course scheme，课程扫描/查询/获取
- [x] **三层 context_messages**（`main.py`）——Layer 1 课程列表 → Layer 2 课程概况 → Layer 3 章节剧本，渐进式披露
- [x] **桌面壳窗口**（`src/window.py`）——PySide6 + QWebEngineView + qasync 单事件循环 + 启动加载态
- [x] **字幕条**（`_SubtitleBar`）——实时流式消费 `matrix.session.get_logos()`，可关闭
- [x] **switch_layout 命令**（`moss_in_reflex.py`）——`always_observe=True`，两步切换协议
- [x] Reflex 事件系统（`events.py`）+ 命令生成器（`event_generator.py`）
- [x] MODE.md（bringup_apps 含 ui/reflex, ui/moshi, web/resource_server）
- [x] `<sleep>` CTML 原语用于视频等待

### 待完成

- [ ] **随时打断和恢复进度**（2026-06-27 新增）— 见下方"下阶段规划"
- [ ] 端到端集成测试（启动 moshi + reflex，跑完 moss自我介绍 5 幕）
- [ ] 确认资源文件（图片/视频 locator）已正确导入对应 storage
- [ ] Ghost Runtime 层 observe 中断（方案 2）：当 `always_observe` 命令返回时，Ghost Runtime 主动中止模型生成，避免浪费 token

### 下阶段规划：随时打断和恢复进度

当前 moshi 的状态（当前课程、当前章节）存储在 channel 闭包 `nonlocal` 变量中，
进程退出即丢失。需要实现：

**目标**：用户/Ghost 可在任意时刻暂停演示，关闭窗口，之后从断点恢复。

**关键设计点**：
- 状态持久化：课程名 + 当前章节 id 写入 session storage（或 matrix session metadata）
- 恢复入口：启动时检测是否有未完成的 session，有则自动恢复到断点章节
- 与 Ghost 的交互：恢复时推送上下文告知"你之前在第 X 章中断，现在继续"
- 不破坏现有架构：仍通过 context_messages 推送状态，不新增命令

**待定**：
- 持久化粒度（仅章节级，还是章内位置？）
- 多 session 并存策略
- 与 `moss features` 体系的追踪集成

### 跨布局共同难点

- **动画与 Reflex 的摩擦**：Reflex 声明式模型下，动画靠 CSS transition/animation。组件挂载/卸载动画（enter/exit）在 Reflex 中不好做——没有 React 的 `<Transition>` 组件。每个布局需要独立设计 CSS 动画策略。
- **事件系统够用但不够优雅**：对于"添加节点 + 添加连线 + 设置脉冲"这种复合操作，Ghost 需连续发多个命令，增加了编排负担。
- **测试困境**：布局是纯视觉的，单元测试只能验证 ComponentState 字段更新，不能验证渲染效果。

---

*Created: 2026-06-23. Based on `.moss_ws/apps/ui/moshi/DESIGN.md`.*
