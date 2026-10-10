---
created: 2026-07-24
depends: []
description: >-
  MOSS 开箱的屏幕躯体 —— 一套 window 语义模型（item / group / arrange / fullscreen /
  veil / background）加它的三种实现（webview / qt compositor / os window）。
  语义共享、躯体分离；当前施工面 = 零依赖 webview manager。
milestone: null
priority: P1
status: in-progress
status_note: >-
  2026-09-19 工作名 screen-node → screen-manager：语义模型提升为脊柱，qt 与 webview 是两个
  实现。当前施工面 = 零依赖 webview manager（验证页先行）。qt 设计已定稿，实现待开工。
  2026-10-02 background 的音频数据面接通（W1 关闭）：雨改由真实 AudioSampleTopic 逐列驱动，
  见 web_manager.md 的「background：声音数据面」；产品化收尾项见 screen-todo.md 的 W6–W8。
  2026-10-10 定位收紧（thirdgerb）：它不是"某块屏的后端"，而是 **screen 模式启动即打开的那个工作台**
  —— 浏览器里的一个 Web 分屏桌面，别的 node 的 surface 以 item 身份被摆进去。验证期已过，
  施工面从"验证页"转成"产品化的默认面"；人的入口（顶栏 / 切组）归属待定 → screen-todo O4。
title: Screen Manager — MOSS 开箱的标准可扩展屏幕躯体
updated: '2026-10-10'
---

# Screen Manager

> 本文是**脊柱**：只放结论与索引。实现细节归子文档与验证代码，不在本文重复 ——
> 细节写进来就是第二次手工维护，必然漂移。三类读者的入口见文末「注意力分配」。
>
> 沿革：旧设计（Decision 1–11 + S1–S3）见 [FEATURE.legacy.md](FEATURE.legacy.md)。
>
> 问题清单（单一事实源）→ [screen-todo.md](screen-todo.md)。

## Motivation

最初定位「web 合成器」（窗 = URL），跑通后只剩一处站得住的核心，其余被实战否掉。
2026-09-13 翻转成「屏幕资源空间」——自有合成器仍是脊，但资源从 URL 泛化为任何可窗口化的
东西。2026-09-19 再进一步：**这块屏不是一个后端，是一套 window 语义 + 三种实现**，
其中零依赖的 webview 实现要开箱可用。

## 默认面：模式启动即打开（2026-10-10 thirdgerb 定）

screen 不是"某块屏的后端"，而是 **screen 模式启动时自己打开的那个工作台**：浏览器里的一个 Web
分屏面，别的 node 的 surface 以 item 身份被摆进去。设计目标是**桌面** —— 不是显示器，也不是播放器。
验证阶段用 Web 技术实现，理由是**最快最便宜**；qt compositor 与 os window 两档是同一语义的另两种
躯体，换躯体不改这条定位。

推论（本文其余部分据此对齐）：

- 这个页面在被打开的那一刻就是**我的桌面**，人看到的"屏"是它的 active group。
- 池子（不在任何 group 的 item）仍然存在，只是不在屏上 —— 人不看池子，模型在池子里挑。
- 多屏不是"多开几个窗口"，而是这套语义在多个显示面上的分工（W8 未验）。

## 语义模型（脊柱）

### 对象模型

| 概念 | 定名 | 语义 |
|---|---|---|
| 物料 | **item** | 一个可窗口化的东西。住在**物化池**，**模型单独可见**（人看不到池子） |
| 桶 | **group** | 有序 item 列表；**空 group 自动删除**。人看到的"屏" |
| 当前桶 | **active group** | 人可切、模型可切；改变它必须回流 |
| 全屏 | **fullscreen** | 模态，与分格正交。不是第五种分格 |
| 底层 | **background** | **空屏时可见**，Ghost 不需要争取注意力就能出现的地方 |
| 姿态层 | **veil** | 最外层遮罩 + 指向/动效。`frontend` 一词弃用 |

**不变量：一个 item 在且仅在一个 group。** 不在任何 group = 离屏（还在，只是不在屏上）。
这条不变式由后端维护 —— 模型永远不需要说"从旧桶取出"，它甚至不该知道有这个动作。

### 层栈

`background` → `stage`（分格区，唯一接收指针的物料层）→ `veil`。
z 序只描述层，**不描述物料**：物料之间的先后只体现为分格序位。

人机交互面（输入 / 通知 / 历史）不再属于 screen —— 由 `ghost-in-web`（浏览器扩展）承担。
**归属待定（2026-10-10）**：人"切哪一组"与顶部导航栏究竟住在屏幕页里、还是住在扩展里，
这两句现在对不上 —— 待裁决，见 screen-todo O4。

### 布局：两个家族 × 一个方向参数

| 家族 | 语义 | n 的范围 |
|---|---|---|
| **grid** | 等分（`cols = ceil(sqrt(n))`，末行不满则均分整行） | 1–n |
| **stack** | 一主 n 从（主格 2/3，其余在条带里等分） | 2–n |

`dir="lr" | "tb"` 决定 grid 的换行轴、stack 的条带方位。**形状 = f(n) 的派生量，
不是模型的参数**；模型只给有序 id 列表（`arrange(ids)`）。

**序位映射**：grid 第 k 个 → 第 k 格；stack 第 1 个 → 主格，第 k 个 → 条带第 k-1 位。

**感知前提（修正 KD7 的隐含假设）**：面积不承载**连续的** priority，但承载**角色**。
「第 2 比第 3 重要一点」用面积微差表达 = 偏置通道，禁止；「这个是主角」= 名义-序位混合
通道，允许。

**尺寸不进模型上下文**（2026-10-10 判据）：舞台的像素几何（视口宽高、栏宽、格距）不搬进
模型侧的 notice / context —— 模型只给**有序 id 列表**，形状由 n 派生。屏幕页确实把几何自查
上报（stdout 的 `[client]` 行），但那是**调试面**，不是感知面。理由：模型是按个数决策的，像素
只影响渲染；把一个"视口一变就变"的数字放进上下文，是花 token 买一个它并不需要的决策输入。
栏的展开 / 收起是例外且必要 —— 它改变的是**可用区这一约束本身**，而不是它的像素度量值。

### 感知两级

| 级 | 载体 | 内容 |
|---|---|---|
| 温 | **named notices** | 状态。框架按 fragment 做 delta，只重发变化的那一个 |
| 回执 | command 返回值 | **只装 notice 载不住的**：拒绝 / 失败 / 截断 / 自动副作用 |

**notice 是状态的唯一权威来源**，回执不复述状态 —— notice 已经是 delta，回执复述就是
双重 delta。

### 主权：人管"看哪"与"node 生死"，模型管"怎么排"

| | 权限 |
|---|---|
| **人** | 切分组 + 全屏开关（选择观看对象）+ 关闭 / 禁止 node（定 node 生死） |
| **模型** | 物化、上屏、分格、排序、navigate（自建 item 的 url） |

### 时间感

命令不是单纯状态变更，有**三档时长**。第二档是跨轨道共同时钟 —— 手势与语音用同一把尺，
模型才能说出"这里（1.5 秒）是入口"并且真的对齐。

| 档 | resolve 时机 | 例子 |
|---|---|---|
| 瞬时 | 立即（回执） | `open` / `arrange` |
| 有时长 | **声明的时长结束**（不是动画真实结束） | `veil.mark(item, region, duration=3)` |
| 持续 | 直到取消（返回句柄） | 常驻高亮环 |

### 一条尺（贯穿两个方向）

> 变化后，模型不知道就会做错决定 → **说出去**。
> 模型侧进 notice；人类侧回流信号；不改变模型状态的不回流。

## 三层实现（可插拔）

| 层 | 实现 | 状态 | 归属 |
|---|---|---|---|
| webview | iframe 池 + 分组 + 分格 + veil | **当前施工面**，见 [web_manager.md](web_manager.md) | 本 feature |
| qt compositor | QML 场景图合成器 | 设计定稿、实现待开工，见 [qt_compositor.md](qt_compositor.md) | 本 feature |
| os window | 真实 OS 窗口（枚举/摆位/激活） | 待建 | `moss-os-control` 的 `window_control`，**不在本 node** |

三者的关系是**语义同构、实现分层**：`window` 是抽象，webview / qt view / os window 是
三种躯体。命名里不放任何实现名 —— 那是 `desktop-gui` 当年的死法。

## Design Index

- 后端子文档：[qt_compositor.md](qt_compositor.md) · web_manager.md
  （background 的声音数据面与雨的模型 → web_manager.md「background」一节）
- 旧设计（被取代）：[FEATURE.legacy.md](FEATURE.legacy.md)
- 碰撞轨迹：`discuss/2026-07-24_screen_body_design_collision.md` ·
  `discuss/2026-09-13_screen_formal_design_collision.md`
- 旧视觉 demo：`demo/`（保留）
- 相邻轨迹（参考，非依赖）：
  - `types/topics/audio.py` — `AudioSampleTopic`（双边频谱，视效数据面）· `ClauseTopic`
  - `types/topics/vision.py` — `FaceTopic`（归一化人脸框，可直驱 veil 指示）
  - `types/topics/ghost.py` — ghost state/emotion 协议，**尚未定案**
  - `qa-exchange` — QA 协议（概念层 `core/concepts/qa.py`）
  - `module-eval-channel` — eval 杠杆的 channel 化

## 注意力分配

| 你想知道 | 去哪 |
|---|---|
| 有哪些已知问题、修到哪了 | [screen-todo.md](screen-todo.md)（单一事实源） |
| 设计是什么、为什么 | 本文 + 对应后端子文档 |
| 这个假设成立吗 | 打开对应**验证页**跑一下；事实在代码里，动机在注释里 |
| 实现怎么改 | web 档 → `nodes/screens/screen_manager/`；qt 档 → 待定（另起 node）；os 窗口档 → `moss-os-control` 的 `window_control` |
