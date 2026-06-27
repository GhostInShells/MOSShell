---
title: Reflex Layout Design — 从静态幻灯片到空间化流式视觉语言
status: draft
priority: P2
created: 2026-06-24
updated: 2026-06-24
depends: []
milestone:
description: >-
  重新设计 Reflex GUI 的布局体系：废弃 PPT 式的"页面填槽"模型，
  转向 CTML 流式本质驱动的空间化视觉语言——凝聚场、活体手稿、镜像对。
---

# Reflex Layout Design

## Motivation

现有 layout（hero / course / stage）本质是**静态幻灯片**——切换时整块替换，和 CTML 的「流式、边生成边执行」内核有割裂。Ghost 在说，画面在跳切——视觉语言和 CTML 的流式本质不统一。

本 workstream 重新设计布局体系：不再有「填入槽位」、不再有「layout 切换」（保留机制但语义转变）、不再把内容当作静态数据填入预置网格。取而代之的是**空间模型**——内容以流的方式到达时，空间做的是什么？

## Design Index

- 原型：`.moss_ws/apps/ui/reflex/prototypes/`
  - `cohesion_field.html` — 凝聚场独立原型（粒子场 + 涟漪 + 震颤 + 力布局）
  - `living_document.html` — 活体手稿独立原型（纸纹 + 墨迹光标 + 跟笔滚动）
- 实现：`.moss_ws/apps/ui/reflex/framework/layouts/`
  - `cohesion_field.py` — 凝聚场 Reflex 集成
  - `living_document.py` — 活体手稿 Reflex 集成
  - `mirror.py` — 镜像对 Reflex 集成
- 配置：`.moss_ws/apps/ui/reflex/moss_in_reflex/config.show_moshi.yaml`

## Key Decisions

### 决策 1：抛弃「填入槽位」，拥抱「空间模型」

**旧模型**：每个 layout 有固定字段（title、subtitle、body、images），Ghost 命令填充字段，页面渲染到固定位置。这是填表思维——字段名是槽位名，组件树是静态布局。

**新模型**：每个 layout 定义的是一个**空间**，而不是一个表格。空间有它自己的物理法则（暗物质粒子场 / 暖纸纤维 / 镜面对称），内容到达时触发的是一组空间响应（凝聚、墨迹、涟漪、震颤），而不是坐标定位。

**Why**：CTML 的核心是「时间是第一公民」。空间模型让时间可视化——内容不是出现，是降临。观众在视觉上感受到 Ghost 的流式本质。

### 决策 2：三种空间模型，对应三种叙事气质

| 模型 | 气质 | 空间法则 | 适用场景 |
|------|------|----------|----------|
| **凝聚场** CohesionField | 壮观、宇宙感、高科技 | 暗空间 + 粒子漂移 + 内容从边缘凝聚 + 落位涟漪 + 邻近震颤 | 架构演示、技术宣讲、开幕/收束 |
| **活体手稿** LivingDocument | 温暖、人本、安静 | 暖纸底布 + 纸纤维噪点 + 墨迹逐字落下 + 自动跟笔滚动 | 叙事性讲述、信件式交流、知识娓娓道来 |
| **镜像对** MirrorLayout | 对比、张力、对称 | 左右暗空间 + 中轴分隔线呼吸 + 行成对出现 + 右侧 0.1s 回响延迟 | 传统 OS vs AIOS、Before/After、二元对比 |

**Why 三个而非一个**：凝聚场和活体手稿气质完全相反，无法互相替代。镜像对是对比演示的专用模型，左右对称 + 延迟回响的体验无法用前两者模拟。

### 决策 3：CSS 动画驱动，不与 React 渲染模型对抗

Reflex 的渲染模型是 State 变更 → 组件树重渲染。在每帧级动画和 React 重渲染之间同步是已知难题。

**策略**：不让 JS 和 Reflex 争抢 DOM 控制权。分层分工：

- **Reflex 层**：管理组件结构和内容数据（字段值、列表项）
- **CSS 层**：管理视觉过渡（`@keyframes coalesce`、`@keyframes inkReveal`、`transition`）
- **JS 层**：仅管理纯 Canvas 效果（粒子场、纸纤维噪点），不碰 DOM 内容

核心技巧：利用 React 的 key 机制。`rx.foreach` 遍历列表时，稳定 key 使 React 复用已有 DOM 元素（CSS animation 不重播），新 key 使 React 创建新元素（CSS animation 在 mount 时自动播放）。入场动画和稳定状态由此区分，无需额外状态追踪。

**Why 不用 JS 驱动的复杂动画引擎**：原型中实现了光晕轨迹、涟漪、震颤等 JS 动画，但 Reflex 集成版暂只保留 CSS 动画 + Canvas 粒子。JS 引擎方案需要在后续迭代中解决与 React 生命周期同步的问题——当前 CSS 方案提供 80% 的视觉体验，30% 的复杂度。

### 决策 4：字段名向后完全兼容

Ghost 对着 HeroLayout / CourseLayout / StageLayout 写的脚本不应因新 layout 而失效。

**策略**：新 layout 声明所有历史字段名的超集：

```python
# 兼容全部历史命名
title: str = ""          # HeroLayout / CourseLayout / StageLayout
sub_title: str = ""      # CourseLayout
subtitle: str = ""       # StageLayout
main_text: str = ""      # CourseLayout
body: str = ""           # StageLayout
image: list[Image] = []  # CourseLayout (单数)
images: list[Image] = [] # StageLayout (复数)
annotations: list[str] = []  # CourseLayout
appreciation: str = ""       # CourseLayout
status_bars: list[dict] = [] # StageLayout
cards: list[dict] = []       # StageLayout
```

同类型字段（如 `sub_title` 和 `subtitle`）在渲染时只显示一个，避免双份。CourseLayout 命名优先，StageLayout 命名回退。

**Why**：Git 历史中的 Ghost 剧本不应因视觉升级而变成死代码。字段兼容让新 layout 对旧剧本透明。

### 决策 5：layout 切换语义保留，但用途转变

`switch_state` 机制保留，但不再是「PPT 换页」——它是「Ghost 换表达方式」。同一场演示中，Ghost 可以：
- 开幕用 hero（黑底大白字，宣告身份）
- 架构讲解用 cohesion_field（粒子降临，技术感）
- 温暖叙事用 living_document（手稿生长，人味）
- 对比用 mirror（左右镜像，张力）

`config.show_moshi.yaml` 注册所有 layout，Ghost 按叙事节奏自由切换。

## Implementation Notes

- **Canvas 噪点生成**：`living_document.py` 的纸纤维效果用 `ImageData` 逐像素随机生成，一次性写入 Canvas，不需要 rAF 循环。这是静态噪点，不是动画。
- **粒子场性能**：180（凝聚场）和 90（镜像）颗粒子，纯 Canvas 2D，不涉及 WebGL。QWebEngineView 下 60fps 稳定。
- **rx.script vs rx.html**：`rx.script()` 注入 JS 到 `<head>`（通过 Helmet），`rx.html("<style>...</style>")` 注入 CSS 到组件体内。两者在 Reflex 中均可用。
- **自动滚动**：原型中的 `scrollToBlock()` 在 Reflex 版中未实现——Reflex 的滚动容器控制需要额外的 `ref` 机制。当前用 CSS `overflow-y: auto` + `scroll-behavior: smooth` 作为过渡方案。
- **镜像对延迟回响**：CSS `animation-delay: 0.1s` 实现右侧行的微延迟，无需 JS 协调。

## Future

- **凝聚场 JS 动画引擎**：将原型中的光晕轨迹、涟漪、震颤加回 Reflex 版，解决与 React 生命周期的同步问题
- **力导向布局**：当前所有 layout 用 flexbox 居中堆叠，未实现原型中的力布局。需要服务端预计算或客户端布局引擎
- **活体手稿的跟笔滚动**：当前依赖浏览器原生滚动，后续用 `scrollIntoView` + `behavior: smooth` 或 Reflex ref 实现主动跟笔
- **更多空间模型**：星图（3D 星座空间）、脉动核心（呼吸中心 + 同心光环）在概念阶段，待原型验证
