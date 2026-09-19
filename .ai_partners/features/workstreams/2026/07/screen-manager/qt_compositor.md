---
title: QT Compositor — 自有合成器后端
node: nodes/screens/screen
created: 2026-09-13
updated: 2026-09-19
status: design-locked
---

# QT Compositor

> screen-manager 的子文档：**自有合成器后端**（QML 场景图）。2026-09-13 重设计的结论，
> 是全部 KD 的原文出处 —— 其中 KD1–KD4、KD7、KD10–KD11 描述**语义**，两个后端共享；
> KD5–KD6、KD8–KD9、KD12 是**自有合成器特有**的实现约束。
> 其余：旧设计见 [FEATURE.legacy.md](FEATURE.legacy.md)；碰撞轨迹见
> `discuss/2026-09-13_screen_formal_design_collision.md`；
> 运行原型 `nodes/screens/qt_screen` **不改不删**，新版 node 从头开始。

## Motivation

旧方案把 screen 定位成「web 合成器」：窗 = URL，QML 场景里塞 WebEngineView。它跑通了
（S1–S3），但有几处被实战否掉：前排缩略图条不是排列机制、webview 是唯一的内容种类、
background 只是占位、人类没有布局权、QA 没有落点。这一轮不修旧，而是**重新规划整个方案**。

核心转向一句话：**screen 是「屏幕资源空间」，不是「web 窗口管理器」。**

- 自有合成器仍是脊（UI 真相在自己状态机里，动画时长即 command 的 await 边界——时间一等公民）。
- 但「资源」从 URL 泛化为**任何可窗口化的东西**：owned（自己渲染）+ managed（真实 OS 窗口，枚举/摆位/激活/关闭）。
- 模型是布局第一公民，但人保留主权；background 可被模型运行时编程；内容是**流式的活表面**，不是死文件查看器。

## Key Decisions

### KD1 命名：screen（否决 desktop-gui / moss-gui）

`desktop-gui` 撞语义——本项目 `desktop` 已等于「OS 操作面」（desktop-channel：文件/进程/系统），
新 node 恰好也碰 OS 窗口，两个 desktop 会互相吃掉。`moss-gui` 目视不可解。
落 `screen`：它就是 Ghost 的屏幕，从「web 合成器」扩到「屏幕资源空间」没改变它是什么。
**边界钉死：desktop = OS 操作，screen = 可见窗口空间。**

### KD2 屏幕资源空间：owned / managed 二元

- **owned 资源**：自己创建/渲染（QML item、WebEngineView、图片、视频）。可合成、可动画、可换皮，美学属于这层。
- **managed 资源**：别的 app 的真实窗口。macOS 不能 reparent 跨进程窗口（fromWinId 不可用），
  所以只能枚举/摆位/激活/关闭，**捕获（截图）是可选的 managed 能力，不是本体**。
- 这不是对旧「OS 级 WM 否决」的反转，是**泛化吸收**：被否的是「本体是像素回路」，不是「永远不许碰 OS 窗口」。

**事实纠偏**：PyPI 没有 `getwindow` 包。macOS 可用的是 `pywinctl`（`MacOSWindow` 走 AppleScript 管
别的 app，需辅助功能权限；watchdog 子模块可监听窗口变化）。`pygetwindow` 在 macOS 上是残废的。

### KD3 工业风：master-detail + 分格

七种候选风格，映射为专业名后其实是**一个设计空间里的七个点**：

| 俗称 | 专业名 | priority 通道 |
|---|---|---|
| 工业/IM | master-detail + tiling WM | 序位 |
| 科幻球面 | carousel / spherical projection | 景深+尺度 |
| 桌面 | stacking WM (WIMP) + spatial canvas | z-order |
| 卡牌塔罗 | grid / card layout + choreographed transition | 栅格序位 |
| 细胞 | force-directed graph layout (d3-force) | 图中心性 |
| 瓷砖 | treemap / space-filling | 面积 |
| 气泡 | circle packing / perimeter | 半径 |

**选工业风的理由**：功能化、priority 可感知最强（序位）、与现有实现连续（solo/split 就是 1分/2分）。
依据（Cleveland & McGill 1984 感知任务准确度排序 + Stevens 幂律）：**位置 > 长度 > 面积 > 颜色**，
面积被系统性低估（幂律 ~0.7），是偏置通道。所以 priority 编码**必须走位置，不走面积/颜色**。

构图：**左 = 物料列表（master），中 = 屏幕区（分格），右 = QA 收件箱**。

### KD4 两档排序（lexicographic）

- **第一档 = 语义级别**（log level 式）→ **外观通道**：有序分类，用色阶 + 徽标（ERROR 红 / WARN 琥珀 / INFO 中性 / DEBUG 暗），
  配明度-饱和度梯度让「更严重」可读出序。别用纯色相（名义通道）。
- **第二档 = 排序强度** → **位置通道**（连续量喂位置，最准）。
- 排序 = `(level, strength)`。**level 变 = 跨档大位移 + 外观变（同帧）**；**strength 变 = 同档微移**；
  **纯外观变（badge 计数）= 不动**。让「动」始终承载信息（calm technology）。

### KD5 动效分两类 + object constancy

| 动画 | 触发 | await? | 机制 |
|---|---|---|---|
| 重排移动 | priority 馈送，自发 | 否 | 隐式（ListView displaced / Behavior） |
| 物料→全屏物化 | 模型 command | 是 | 显式 Transition + animation_finished resolve |

- **object constancy 是硬要求**：`ListView` + `ListModel.move()` 按模型身份复用 delegate，重排时 item 只移动不销毁重建。
  **禁用 `Repeater` + 静态 model**（重排会销毁重建全部 delegate，constancy 归零）。
- hover 暂停重排：`HoverHandler` 挂起；恢复时**只应用最新态、动画一次**，不重放积压队列。
- WebEngineView 动 transform（scale/position），**不动 width/height**（会触发独立进程里的页面重排）。

### KD6 background：模型可编程（f/g + state 两函数模型）

```
visual = f(state, time)     ← 函数帧：每帧 GPU 跑（shader），模型不写
state' = g(state, event)    ← 响应函数：事件触发、低频，模型运行时写
```

- 方案一（选 N 种 + create）与方案二（函数帧）是**同一架构的两层**，不是二选一。
- **shader 可编程但需 QSB 烘焙**（Qt6/RHI，GLSL → .qsb → Metal 的 MSL 等）；**运行时编程走 QML**
  （`QQmlComponent.setData(string)` + `create()`，QML 本就是运行时解释语言，`errorString()` 做运行时校验）。
- 第一版 = **非人化 shader 基底**（球/波纹，一个全屏 `ShaderEffect`，energy/emotion → uniform）。
  Live2D = 后续 opt-in 重模式（唯一可能卡的，别做默认）；功能化（logos+input）不是背景，是 HUD。
- **性能硬约束**：全屏持续动的东西必须是 shader（不是 QML item / Canvas / Text 海）；单 uniform 驱动；
  空闲冻结（同一窗口内任何一层动 → 整个窗口逐帧重绘 → 永不 idle）；折半分辨率；量测用 `QSG_RENDER_TIMING=1`。

### KD7 布局：fullscreen 模态 + 按数分格

- **fullscreen**：模态，一次一个；QA 有覆盖权；左侧 field 收束。
- **主区域按节点数分格**（N ≤ 4，webview 有界）：

| N | 形状 | 专业名 |
|---|---|---|
| 1 | 满格 | monocle |
| 2 | 左右半分 | horizontal split |
| 3 | 左主(通高) + 右列两格 | master-stack |
| 4 | 2×2 四象限 | grid / BSP |

- **顺序 = 优先级**：第一个节点进主格（N=3 左边、N=4 左上）。分格本身就是优先级地图。
- 命令面最小：`arrange(有序 id 列表)` / `unarrange(id)` / `fullscreen(id)` / `exit_fullscreen()`。
  象限名（left-top…）是**读回词汇**，不是命令词汇。
- **删旧 front strip**：那是缩略图点选器（点选全屏），不是排列。被真分格取代。

### KD8 channel 体系

```
screen
  objects      open(kind, address, label) / close(id) / list()
               └─ virtual_children: #mail, #docs, ...   ← 每个物化对象一个子 channel
  layouts      只摆位，与 object id 合用
  background   背景控制
```

- `object_manager` 与 `objects` **合并为一个 hub**（sandbox-hub 模式：open/close/list + virtual_children）。
- 子 channel 以短 id 命名，**address 是 join key**（延续旧决策）。

### KD9 四 kind 控制面（第一版）

每个物化对象有自己的子 channel = 控制面。screen 只摆位（layouts），内容操作全在对象自己的 channel。

| kind | 控制面 | 成本 |
|---|---|---|
| `http:` | WebEngineView | 重（已有） |
| `text:` | set/append/clear + font_size/family/color/align/wrap/line_spacing | 轻 |
| `image:` | load + cross-fade transition + scale_mode | 轻 |
| `video:` | play/pause/stop/seek/rate/volume/loop/position/duration/state | 中 |

- **markdown 必须流式**：`stream(chunks__)` 增量呈现；动效三件套 = caret + 平滑滚动 + chunk 淡入。
  **性能**：LLM 流式 chunk 密，不能逐字符重解析整篇——缓冲 + 限速重渲染（20–30fps）。
- **video 用 `QMediaPlayer` + `QAudioOutput`**（Qt6 音量在 QAudioOutput 上）。**音频播放是同一机制**，
  `video:` 顺带覆盖音乐。**打包坑**：`QtMultimedia` 在 PySide6-**Addons** 不在 Essentials。
- **截图统一能力**：所有 owned view 可截图，结果生成 image 物料 → **回流入 field**（闭环，无需新机制）。
- 押后（第二类，接 eval）：`pty:`(terminal)、`qml:`(运行时自建 UI)、`canvas:`(画板)、`3d:`、`capture:`、`avatar:`(Live2D)。

### KD10 人机双主权

两条主权线，别混：

| | 布局 | QA |
|---|---|---|
| 第一公民 | 模型（默认） | **人（唯一权限）** |
| 人的能力 | 仅全屏开关 | 全部（打开/审批） |
| 模型的能力 | 分格/排序/全屏 | 只有 issue，不能 answer |

- **布局开关在界面（人手），不在模型手中**：开关 = 布局写权的互斥锁；主权在人，抢回控制权不需模型授权。
  「模型是布局第一公民」是默认值，不是权利。模型只把开关读成 state（`layout_mode`），遵守它。
- **人类唯一布局权 = 是否全屏化**（一个物料一个 bit）。全屏是模态，人点全屏天然覆盖模型分格，不打架。
- **sidebar 常驻**：非全屏时常驻，全屏时收束。
- **QA 是系统向的结构化 dialog**：上层是 warrant 等机制（「ask user 签发 QA → command 返回 result」）。
  模型对 QA 的全部触点 = 签发那一层的返回值；**授权动作不感知，纯交互可感知**。
- **QA 审批动作不进模型会话历史**——但答案经 QA 协议回 issuer（`qa.wait()`），这是协议回执，不是 chat turn。
  两件事同时为真：不进历史 ≠ 模型收不到答案。
- **topbar 文本输入**：人→系统的自由文本通道（独立于 QA 的结构化；QA 是系统发起，topbar 是人发起）。

### KD11 线程模型：子线程 Matrix + concurrent bridge

- node = 进程；**内部两线程**：Qt 主线程（QApplication + QML 场景）+ daemon 线程（Matrix asyncio）。
- `Matrix.run` 用 `asyncio.run`，**在非主线程安全**（`Runner` 只在主线程装 SIGINT）。
- bridge = `Signal(str)`（`Qt.QueuedConnection`）+ `concurrent.futures.Future` + `janus` 桶。
  **阻塞单向**：GUI 永不阻塞在 Matrix 上（emit/push_nowait 即走）；只有 Matrix 阻塞在 GUI Future 上
  （`await asyncio.wrap_future`）。单向阻塞 = 不可能死锁。
- **不用 qasync**：Matrix 是引擎，要在 Qt 无关上下文跑；qasync 会把 Matrix 焊死到 Qt 主线程，反向依赖 GUI。
- **遗留缺口（正规化要补）**：无优雅关闭（daemon 线程被硬杀，`aboutToQuit` 握手缺失）；`Matrix.run` 的
  uvloop 分支实际失效（`set_event_loop` 后 `asyncio.run` 另建 loop）；main.py 的 100ms QTimer 疑似多余。

### KD12 运行时 eval：codex 编译器 → Qt 主线程

- codex 工具链（`Compiler`/`Executor`/`Reflector`/`Sandbox`）：模型写 `def run_xxx(obj: xxx):` →
  编译 → 取函数体 → **在 Qt 主线程 eval 运行**。一次进入后对活对象无限直接类型化访问，杠杆比 command 化高一层。
- **Qt 主线程真实模型 = 串行执行 + 单帧提交**：多对象操作逻辑串行、视觉同帧（写不立即绘，脏节点下一帧统一渲染）。
  只有 eval 自身体量大才阻塞。**一次 eval = 一次视觉原子提交。**
- **QML 不是 MVC**：Widgets 是命令式，QML 是**声明式属性绑定**（MVVM 味）。**最大威力 = V 是 M 的纯函数、
  只写 M**——直写派生属性会打断绑定。eval 的写靶 = intent state（M），绝不写 derived（V）。
- module_eval_channel 的**子进程隔离不适用于 Qt**（Qt 对象不可跨进程）。eval 面按域分裂：QML 场景走 JS/主线程，
  非 Qt Python 域（playwright/pywinctl）走 module_eval 原样。

### KD13 新版 node 名

新版 node = **`nodes/screens/screen`**（一眼看懂，canonical）。旧 `nodes/screens/qt_screen` 保留不改
（legacy 原型，S1–S3 实现 + `demo/` 继续在那）。

## Implementation Notes

- **Qt6 / PySide6**：绑定用 PySide6（LGPL）。多媒体与 WebEngine 在 PySide6-**Addons**，不在 Essentials
  （旧 demo 的「Essentials」注释是过时的，正规化时要改）。
- **shader 工具链**：QSB（`qsb` 烘焙 `.frag` → `.qsb`）；运行时编程走 QML `QQmlComponent`，不走 shader。
- **第一版 = 功能化，不框架化**：不做风格引擎、不做 layout 抽象引擎。四个 kind + 工业风 + shader 背景 + 分格，
  一件一件写实。第二风格/第二 kind 是并列新增，不是配置通用引擎。
- 旧 `screen.py` 的 PrimeChannel + solo/split states 结构被 `objects/layouts/background` 取代，但**代码不动**——
  新实现另起目录。
