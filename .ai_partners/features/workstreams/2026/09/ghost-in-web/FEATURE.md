---
title: Ghost In Web — 通用浏览器躯体
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-09-21
updated: 2026-09-22
depends: [bilibili-shared-webview]
milestone:
description: >-
  bilibili-shared-webview 的泛化:把"陪人看片"的具体功能扩成全部网页功能。
  人类真实浏览器 + 扩展 + 本地 node。核心 = 行为授权 + 严格可审计。
  页面侧 = 图标(默认感知授权)+ 卫星(预建模的点击推送)。
---

# Ghost In Web

> 本文是**索引与结论**,不是实现细节。设计全文待补 `design/`。
> 前身:`nodes/browsers/ghost-in-bilibili/` + `bilibili-shared-webview` 的 design
> —— 本 workstream 从它长出来,继承身份/通讯/授权形状,泛化能力面。
>
> **做不成就删。** 这个机制不许用"半成品平行轨迹"的方式挂着。

## Motivation

bilibili 的机制回答了"ghost 能不能陪人看视频"。它证明的通路是:扩展 + 本地 WS node +
人类持有的授权,人机共享同一个真实浏览器。但它只覆盖了一个站点、一类内容(视频+字幕)。

本 workstream 问:**这套机制能不能从"陪人看片"扩成"共处整个 web"**。

动机不是功能覆盖,是**信任**。网页里有大量隐私——登录态、个人数据、支付。模型直接操作
人的浏览器是危险的事。所以通用化的第一个约束不是"能做什么",是**"人怎么知道模型做了什么"**。

一句话:**行为授权 + 严格可审计**。

## 与已有三条浏览器路径的关系

| node | 执行面 | 授权面 | 审计面 |
|---|---|---|---|
| `nodes/browsers/playwright` | **任意 Python**(module_eval 沙箱) | 二值:给/不给 | 无 |
| `nodes/browsers/co_browser` | 同上 | 二值 + 一个主开关 | 每次 exec 一帧,推到 surface |
| `nodes/browsers/ghost-in-bilibili` | **枚举动作表**(无任意 JS) | per-page(label) 卫星授权格 | panel 内 |
| **ghost-in-web(本 track)** | **枚举动作表** | per-page + 行为 accept/reject | **node 自己的页面(将来自动进 webview_screen)** |

## 结论脊柱

### 可枚举 → 可授权(本设计的支点)

`co_browser` 砍掉审批、改成"观察替代同意",理由是模型的 exec body 是 Python,**行为不可
枚举**,审每一条会淹死人。本 track 相反:扩展下发的是**封闭动词表**(DOM/CDP 的现有 API),
执行面一旦封闭,**per 行为授权才在结构上成立**。

所以"通用插件走行为授权"和"co_browser 砍审批"不矛盾——是执行面可枚举性不同的必然分叉。

### 泛化方式:不加抽象,直接下发 API 控制能力

不做"通用 webview 语义"层。模型本就懂 `querySelector` / `click` / `dispatchKeyEvent` /
`getBoundingClientRect` 干什么。**抽象工作不在动词层,在包住动词的那圈授权 + 审计框架。**

沿用 bilibili 的 `label` = 唯一持久身份;`bvid`→通用页面无对应物,内容(url/title)只作
易变属性。帧协议、授权状态进 notice、node 不提供授予授权的命令——全部继承。

### 控制面三层(不 eval 的前提下)

| 层 | 机制 | 事件可信 | 代价 |
|---|---|---|---|
| T0 扩展层 | `chrome.tabs.*`、`captureVisibleTab` | — | 无 |
| T1 DOM 层 | content script isolated world | **否**(`isTrusted=false`) | 无 |
| T2 CDP 层 | `chrome.debugger` + `Input.*`/`DOM.*` | **是** | 可见 infobar |

T2 的 infobar 与 bilibili 的**非打扰**动机直接冲突 → T2 不能做默认层,只能做显式、人类
可感知的升级。

### 审批面 = web 页面自身 → **先不走 warrant 体系**

审批不在 MOSS 内核里,在**页面自己的 UI** 上:人类在页面上 accept / reject(必要时加一个
`analyse`)。扩展持有授权状态,node 只做镜像 → 状态进 notice → 模型自己推理。

所以本 track **先不引入 `matrix.warrant`**。页面本身就是审批面,足够承载;等这个薄机制真
暴露出需要内核级授权的场景,再把 warrant 接进来不迟。

与 bilibili §4"审批按钮砍掉"的调和:bilibili 反对的是**弹窗审批 + 卫星球两套控制并存**。
若 accept/reject **就渲染在页面上**(和球同一面),就只剩一套入口,§4 的反对理由消失。

### 点击图标 = 默认感知授权

一次人类动作,授予该页面**常驻的读权限**:

- **IN**:页面身份(url/title)· DOM/AX 读 · 对话面(页面输入框 人→ghost,`reply` ghost→人)
- **OUT**:截图 · 导航/点击/输入/提交/上传 · **人自己的行为**

即"读不是全免费,但图标点击就授权读"——入口是一次人类动作,开了就常驻。

### 截图 = 卫星,推逻辑,**不可拉取**

人类点击一次 → 发送一张截图。**不点击永不能拉取**。不走审批——它属于"人类主动点击"这一类。

于是出现第三类:**卫星 = 预建模动作 = 人类主动点击触发的推送**。截图是第一个;同类功能皆
做成卫星。

### 隐私红线(最硬的一条)

**人的动作、行为,不感知、不分享。**

- 除非显式设计了一个**感知面**并做成**授权卫星**——且即便如此,捕获的也必须是**事件**,
  不是**行为流**。
- 泄漏伤的不是人、也不是模型,是**框架**。所以边界宁紧勿松,**做越少越好**。

### 地址与视觉边界

- 扩展**不在 local 类地址生效** → node 自己的面(localhost 上的 `webview_screen`)天然排除。
  审计面与被控面靠**地址空间**分开,不靠逻辑判断。
- 页面无法与 node 共享 → **点图标必须失败**。不留幽灵 affordance:图标在、点了没反应,
  比不显示更坏。
- 扩展机制与 moss 分离,**视觉可区分**。

### 审计面

审计面 = **node 起的页面**(`co_browser` 的 surface 形状),未来让它自动进
`nodes/screens/screen_manager`(`webview_screen`)。**所有行为可回到那页上看。**

审计记录的是**模型的行为**(dispatch + 结果),不是人的行为——与隐私红线一致。

## 用到的内核信号面

审批**不在这层**——它在页面里(见上)。这层只是信号通道:

| 用途 | 机制 | 位置 |
|---|---|---|
| 推模式 | `new_aside_signal(...)` → `AsideNucleus` | `core/mindflow/aside_nucleus.py:267` |
| 图像进模型上下文 | `CommandUtil.observe_image(text, image)` | `core/blueprint/channel_builder.py:209` |
| 人的输入 | `send_input_signal`(继承 bilibili 的 `input` 帧) | `core/blueprint/channel_builder.py:286` |

`AsideNucleus` 语义:NOTICE 永不打断,空闲才看见;`buffer_size=20` 溢出丢最早;0.5s 冷静期;
`hint/description` 取最新一条 = **新数据驱动**。适合实时状态流。

## token 纪律

**默认不共享像素。** bilibili 全程零截图——它用**字幕(文本)**做感知,`state` 上行是标量。
它已解决过的 token 爆炸是字幕轨道(30 分钟约 400 句),解法 = **时间区间查询 + 滚动窗口**,
不是倒进上下文。

通用版没有"字幕"这个天然窗口 → **读的窗口化是新的待解问题**。像素(截图)一旦进上下文
代价高得多,所以截图走"人点一次发一张"的推逻辑,正是对它的节流。

## 未验证 / 待定

1. **动词表切分**:哪些动词进授权面?"点击"要不要按元素类型再分(点链接=导航 vs 点按钮=未知副作用)?
2. **"行为确认"的粒度**:确认的是"该页是否开放这个能力"(记住,重复调用不再问),还是每次调用?
3. **模型能否 pull 读?** 图标点击授权常驻读 → 模型似乎可以主动读;但截图**不可拉取**。
   这两种"读"的边界要对齐。
4. bilibili 的「未验证声明」(MV3 SW 持 WS 等)在本 track 同样成立,不重复。

## 实现状态与遗留

**已落地并 live 验证**(真 Chrome,2026-09-22):图标授权感知 · `read`/`find`/`click`(含页面
确认条)· `say`+input 对话 · 截图卫星(降采样 JPEG push)· 审计页(渲染截图缩略图)· SPA
导航后浮层自动重挂。代码在 `nodes/browsers/ghost-in-web/`,24 个测试。

**遗留问题:**

1. **观测面没有观测模型行为。** 审计页展示感知面(页面/截图/对话)是对的,但它没有展示模型
   的**代码动作到底是什么** —— 现在只看到 dispatch 动词 + 结果(`click('r1')`),看不到模型
   意图的完整表达。观测面的重心该从"发生了什么"转向"模型在做什么、为什么"。
2. **截图和说话才是核心功能。** 这个 body 的核心价值是**共享**(截图 push + 对话),不是
   `click`/`type` 这些控制操作。后续重心应放在共享体验上,控制面是次要的。
3. `type` 未 live 测(与 `click` 同一套下发+确认链路,多一步 React 原生 setter)。
4. `find` 纯文本匹配,定位不了无可见标签的输入框(如 baidu 搜索框);要按 `id`/`name` 定位需补。
5. 截图会把注入的浮层一起拍进去(要干净需在抓取瞬间临时藏浮层)。
6. 打磨:卫星半圆弧、字符加大、图标自解释(现为"W"方块)。

## 注意力分配

| 你想知道 | 去哪 |
|---|---|
| 本 track 从哪长出来 | `bilibili-shared-webview` 的 FEATURE.md + design |
| bilibili 的身份/通讯/授权形状 | `.ai_partners/features/workstreams/2026/09/bilibili-shared-webview/design/2026-09-20_ghost_in_bilibili_body_design.md` |
| 观察替代同意的先例 | `co-playwright` FEATURE.md |
| 屏幕躯体 | `screen-manager`(节点 `nodes/screens/screen_manager`) |
