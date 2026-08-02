---
created: 2026-07-24
depends: []
description: 为 Ghost 提供屏幕躯体的平台专属 node：数字人 background 槽 + n focus 槽 + meta 游离层的自有合成器。窗
  = URL 的最简契约，窗内操作走各 cell 自己的 channel，address 是 join key。每个平台一个独立 node（INSTALL.md
  门控）， 首个实现为 macOS PyQt6/QML。
milestone: null
priority: P1
status: in-progress
status_note: 'S3: transition + split landed — curtain animation on layout switch (Future resolved
  on animation complete, "command 返回时刻 = 视觉稳定时刻"), split layout with dual
  WebEngineView focus slots (left/right), _DEFERRED bridge mechanism for animated ops,
  context_messages split-aware (left:/right:). End-to-end verified via MCP:
  solo↔split bidirectional transition, dual WebEngineView rendering side-by-side.'
title: Screen Node — MOSS 开箱的标准可扩展屏幕躯体
updated: '2026-08-03'
---

# Screen Node

> Use `moss features set-status screen-node <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 需要一个"酷炫的标准可扩展屏幕躯体"：Ghost 并行控制多个 GUI/web UI 进程时，
用户看到的应该是一个有机整体——数字人底座、可切换的分屏布局、确定性的通知与转场动画——
而不是散落的窗口。

选型讨论（2026-07-24，三轮碰撞）确立了根本路线：**自己拥有合成器**，而非做 OS 桌面的
客人（AX/xdotool/mss 截图那条路感知回路是像素的、编排权是别人的、跨平台是三套后端）。
自有合成器下，UI 的真相在自己的状态机里，动画时长即 CTML command 的 await 边界，
"时间一等公民"精确落地。

跨平台策略不是一个实现的兼容矩阵，而是 **n 个平行 node 的自然选择**：nodes 大目录下
逐步为每个系统做一个 node，INSTALL.md 门控依赖——没装就不被 Ghost 发现，能力门控免费。
可以有很多个不同的 windows 组织协议并存。

与 moshi 的边界：moshi 是另一位架构师在独立仓库演进的导演/舞台体系。本 workstream 是
MOSS 开箱内的**独立、相对简单、符合本轮设计**的实现——不认亲、不对齐词表，只共享
CTML/matrix 地基。

## Design Index

- Key design documents: `design/`
- Key discussion records:
  - `discuss/2026-07-24_screen_body_design_collision.md` — 选型、meta 单元、桶协议、数据交换的完整碰撞轨迹
- 相邻轨迹（参考，非依赖）：
  - `matrix-resources`（draft）— `servers://` scheme 是"URL 发现"的将来时；第一期不依赖
  - `desktop-gui` — K3 线程模型（GUI 主线程 + Matrix daemon 线程）是已验证两次的模式
  - `reflex-layout-design`（draft）— 空间化布局的先行思考

## Key Decisions

### 1. 窗 = URL，最简契约；cell providing 协议是升级路径而非现在

窗的身份是 **URL**，不是 cell、不是 manifest。node 起来后自己声明"我 serve 一个 http
地址"（provide 语义/channel description 里自带，类似 playwright channel 逻辑），screen node
对来源零知识——**任何 URL 都能成为窗**，包括不是 cell 的外部页面。扩展性最大化。

**拒绝的重方案**（第一轮曾提出，被产品判断否决）：WindowManifest 独立协议。node provide
本身就有语义在，不需要额外的呈现声明协议。最多将 window 上升成 address 级别的通用协议。

**升级路径**（明确不现在做）：若 meta 单元 + 红点这套产品逻辑被验证是标准的，再上升为
matrix 级的 cell providing 协议（类似 channel provider/proxy，一切封装在 node address 下，
ghost 看 address 即看全貌 + 生命周期 + 红点）。meta item 留一个可选 `source`（cell 身份）
字段作为升级钩子，今天来自 URL 声明和手工 open，协议落地时可被自动填充。升级不推翻。

### 2. 游离的是 meta 单元，不是窗口——presence 层与渲染层二元

IM 类比：会话列表的 item 和打开的会话是两种存在。

- **MetaItem（游离层）** = 图标/条目/徽标——纯 QML 轻量元素，零 web 内容，走游离动画。
  它是"存在的最小可视记号"，将来可承载红点计数。
- **Window（挂载态）** = URL 被 focus 进槽位时才**物化**的 webview；unfocus 即退化回
  meta item（webview 销毁或休眠）。

推论：webview 数量是有界常量 `n_focus_slots + 1 (background)`，与网络里有多少活动节点
无关。"游离窗活渲染还是快照"的问题直接消失——游离层根本没有窗。

### 3. Layout 体系与 channel 树形态

每种 layout 有：一个 **background 槽**（数字人/底座）+ **n 个 focus 槽**（分屏方案定）+
其余为游离 meta 层。模型看到的 channel 树形如：

```
matrix
  screen              switch_layout / open(url, label) / close(id)
  screen.current      focus(id, slot?) / unfocus(id) / transition(...)
```

- `screen.current` 用 **states channel 惯用法**：layout 切换 = 状态切换 = command 集整体
  换血（分屏 layout 有 focus_left/focus_right，单焦点 layout 只有 focus）。渐进式披露免费。
- **occupy 语义即转场锁**：`screen:switch_layout` 占用父 channel 时整个 `screen.current`
  子树 blocked——转场期间布局命令排队，语义天然正确，零锁代码。
- webview 列表挂在 screen 的 **context messages** 上。模型用 screen 只管布局；窗内操作走
  各 cell 自己的 channel（如 terminal），**address 是 join key**。用户看到一个体系，
  模型看到两个正交的面。

### 4. 短址映射省 token

address/URL 直接进 command 参数 token 太多。沿用 `matrix_channel._refresh` 的 alias 纪律：
context messages 里维护 `短id → (label, url, source)` 映射，所有 command 参数只吃短 id。
形如：

```
layout: split_2
slots: {left: #blog, right: —}
drifting: #term(shell) #mail #docs
```

### 5. 平台间 channel 词表不预先标准化

不把 focus/unfocus/transition 定成跨平台标准词表。code as prompt 下模型每次读真实
interface，词表差异成本本来就低；macOS 与 Ubuntu node 对"焦点/分屏"的自然语义未必同构。
让两三个平台实现先各自长出来，再从活的实现里蒸馏公约数——与 matrix-resources
"信封无聊、差异留在 cell 内"同一纪律。

### 6. 首个实现：macOS，PyQt6/QML

- **场景**：QML scene graph 一层合成器。background 槽可以是 WebEngineView（web avatar
  生态：Live2D/VRM/three.js）或 QML item；focus 槽是布局锚点；meta 游离层是纯 QML 动画。
- **线程模型**：照抄 desktop-gui K3 / `build-a-gui-app` howto 已验证模式——Qt 主线程跑
  event loop + 场景，daemon 线程跑 Matrix asyncio，channel command 经线程安全桥
  （signal-slot / QMetaObject.invokeMethod）改场景状态。pygame、Reflex 之后的第三个皮。
- **转场即 command 时长**：QML animation finished 信号是 command 的 await 边界。
- node 落在 nodes 大目录下（目录名待定，见 Decision 7），INSTALL.md 声明 PyQt6 依赖。

### 7. 命名：screen

标准：**目视可解**——名字应让人一眼看出是人机交互界面，不需要解码隐喻。screen 直接
取自问题的自我表述（"屏幕躯体" = screen body），与现有词汇域全部错开：desktop（OS
操作面）、windows（OS 词汇）、stage（已被开发计划治理占用）。操作词表
focus/background/transition 本身也是平实的 UI 词汇，不依赖隐喻自洽。
**desktop channel 不需要改名**。

曾选 `stage`（剧场隐喻：聚光/换景/wings 候场）后被否决，两个理由：与 ai_partners
正在落地的开发计划治理概念冲突（模型在那里没选 roadmap 选了 stage）；太隐喻，
目视看不出是交互界面。比过的其它候选：`display`（偏输出设备，无"交互"义）、
`surface`（太抽象）、`canvas`（撞 web 技术词）、`viewport`（太 web 域内）。

### 8. 交互事件走 peek/drain 双面桶，不走 signal（对齐 g1 感知纪律）

一等原则："**交互的交付物入上下文，交互的过程不入**"——人类工程师的表述："你现在
看到了整段文字我发送给你，但你看不到我正在打'这个字'时屏幕的交互过程，不影响
你能和我产生协作。" 人在窗内的点击/拖拽属于窗背后的 node，不属于 screen；screen
层面只有布局态变更（focus/unfocus/切 layout）是值得进桶的事实。

桶协议照抄 g1 已验证的双面结构（`unitree/g1/channels/listener.py`）：

- **peek 面**：context_messages 每帧 tail-N **只读**布局事件近况，永不 drain——
  感知是无副作用的旁观。
- **drain 面**：仅由显式触发；batch 经 janus.Queue 从 GUI 线程单点汇入 asyncio。
  真实 ghost 场景 drain 后可转 signal；**MCP 场景无 signal 通路，drain 结果留在
  桶里由 context_messages 呈现**——同一协议的自然降档，不是两套实现。
- 纯感知退化形态参照 `asr.py`（无命令无 signal，peek_window 顺行遗忘）。

拒绝方案：人的操作发 signal 进 Mindflow（第一版讨论中模型的倾向）。否决理由：
交互语义难兼容，视觉共享要支持流式视觉轨迹 + mss 截图才有语义，"这个视角很容易
滑落到特别重"。同构纪律：matrix-resources "resources 投影第一期不产 signal"。

### 9. 数据交换：三通路单写者（queue + Future 做脊柱）

Qt 硬规则：场景对象只能被 GUI 线程碰——"内存共享数据对象"不存在天真版本。
架构按方向拆三条单向通路，**场景真相只有 GUI 线程一个写者**（actor 模型）：

```
控制入 (channel → GUI):  signal-slot 跨线程投递 + concurrent.futures.Future,
                         command 侧 asyncio.wrap_future await;
                         GUI 线程在动画完成回调 resolve / set_exception
渲染态 (GUI 内部):       bridge QObject 属性, QML 绑定自动渲染; channel 永不直接摸
状态出 (GUI → channel):  纯 Python 快照 (锁保护) + 事件桶, 供 context_messages
```

- **约定：所有布局 command 的返回时刻 = 视觉稳定时刻**。异常经 Future 自然流回
  CTML `<result>`。这是 CTML "时间一等公民" 在 GUI 里的兑现点。
- 跨线程桥实现直接用 janus.Queue（g1 listener 已验证：多 callback 源 sync put，
  running loop 单点 async get）。

### 10. layout = QML 组件 + 配套子 channel，成对注册；窄 bridge 契约

```
screen                 switch_layout(name) / close(id)
screen.<current>       当前 layout 的 view command 集 (StatesChannel 换血)
                       例: solo → view(url, label) / clear()
                           split → view(url, slot, label) / swap() / clear(slot)
```

- 一个 layout 是一个开发单元：QML 组件 + 子 channel 定义成对注册。每个 layout
  可以定义自己的小实现——这是本技术栈的核心优势。
- **view command 第一期只认 http**。file:// 等 scheme 是 view 参数域的扩展，不动结构。
- **窄 bridge 契约解耦 channel 与模板语法**：两者之间只有一个小协议——props 进
  （场景模型）、slot 动词进（view/clear/switch）、事件出（交互桶）。layout 的
  QML 只要实现 slot 接口，内部自由。契约窄，layout 就自由。
- layout 运行时开发（模型给自己写 layout）物理可行——QML 本就是运行时解释加载
  （`engine.load` 不需重启进程），是 MOSS "Transformative" 的具身版。第一期不做。

### 11. 验证路径：无副作用 mock node + moss-as-mcp

先于真实实现，做一个伪交互验证 node：

- 窗内容全部 mock，无任何副作用；作为 node 拉起，经 `moss-as-mcp` 暴露。
- **模型侧**经 MCP 调 screen commands 操作布局；**人类侧**在视觉上点击操作。
- MCP 体系无 signal：准备发 signal 的 batch 落桶，模型在 context_messages 里
  看到结果（Decision 8 的降档形态，等价 g1 A 键 drain 但无 signal 出口）。
- 验证目标：双主体操作在同一场景状态上汇流；command 返回 = 视觉稳定时刻；
  桶的 peek/drain 语义；StatesChannel layout 换血的模型可用性。

## Implementation Notes

- **视觉原型已验证**（2026-07-24）：`demo/screen_demo.py` + `demo/Screen.qml`，
  约 200 行 QML——background 呼吸数字人占位 + 5 个漂浮 meta 图标（含红点徽标）+
  点击物化/退化 + solo/split 切换。人类工程师体验确认"体验非常好"。
  依赖 `uv pip install PySide6-Essentials`（清华源），不进 pyproject，`uv sync`
  会清掉、重装即可；正式化时依赖进 node 的 INSTALL.md。绑定选 PySide6（LGPL，
  Qt 官方），非 PyQt6（GPL/商业双协议）。
- **Behavior vs 显式 Transition**：demo 用隐式 `Behavior` 做物化动画，但 Behavior
  拿不到干净的完成信号。command 驱动的转场必须用显式 Transition/Animation
  （有 `finished` 信号）以兑现 Decision 9 的"返回时刻 = 视觉稳定时刻"；
  Behavior 只留给浮游层这类无人 await 的自发动效。
- **桶实现的参照文件**（读了再写，不要自己发明）：
  - `src/ghoshell_moss_contrib/unitree/g1/channels/listener.py` — peek/drain 双面 + janus.Queue 全样板
  - `src/ghoshell_moss_contrib/unitree/g1/channels/asr.py` — 纯 peek 顺行遗忘的退化形态
  - `src/ghoshell_moss_contrib/unitree/g1/channels/locomotion.py` — available 门控 + context_messages 陈述能力状态
- **Stages**：

  ### S1: Dual-thread Node Scaffold (done, 2026-07-26)

  提交：2b2dbf4, 5ea8a968, 6f3cc0ac, fc21ce71, cd3bdc18

  **产出**：`nodes/screens/qt_screen/` — macOS PySide6/QML screen body node
  直接上真实 node（跳过了 Decision 11 的 mock node）。

  - **线程模型**：Qt 主线程（QApplication + QML engine）+ Matrix daemon 线程（asyncio + channel logic）
  - **Bridge**：ScreenBridge QObject — Signal(str) queued connection + concurrent.futures.Future
    跨线程 dispatch，positional args dispatch table 适配 QML 函数签名
  - **Bucket**：EventBucket — peek/drain 双面，janus.Queue 单点汇入，参照 g1 listener
  - **Channel 树**：`screen` PrimeChannel（main state: open/close/set_background/switch_layout/drain）+
    solo/split 两个 StatefulChannel sub-states（focus/front/float/clear）
  - **QML**：四槽 compositor（background placeholder + focus + front strip + float meta items），
    人类点击即时生效 + bridge.human_clicked 入桶；WebEngine 就绪但当前 placeholder 模式
  - **已验证**（MCP via moss-as-mcp）：open/focus/front/float/drain/close 全命令链路通，
    QML 界面正确响应，context_messages 信道通（snapshot QVariant→dict 转换待修）

  **目录决定**：`screens` 是品类目录（为多平台 screen 实现预留），具体 node 在 `screens/qt_screen`。

  摩擦点记录：
  - LoggerItf contract 在 node channel startup 里找不到 → 用户修复了 IoC 反绑逻辑
  - Channel API 错误使用了 `screen.main_state().command()` → 修正为 `screen.build.command()`
    （PrimeChannel.build 返回 MutableChannelState，main_state() 返回 ChannelState 无 command 方法）
  - janus exception 名：`AsyncQueueEmpty` 非 `QueueEmpty`
  - QML 函数从 Python 调用需 positional args，不能 keyword args

  ### S2: WebView + Interactive Drain (done, 2026-08-03)

  提交：待 commit (this session)

  **产出**：WebEngineView 替换 placeholder Rectangle，badge 通路闭环，snapshot 修复。

  - **WebEngineView**：focus + background slot 从 Rectangle placeholder → WebEngineView，
    由 QML Loader 按 active 条件物化/销毁。close button 改为 overlay Item 浮于 WebEngineView 之上
    （WebEngineView 不能包含 QML 子元素）。
  - **QWebChannel**：Python 侧 `QWebChannel.registerObject("bridge", bridge)` → context property
    → QML `WebEngineView.webChannel` 绑定，页面内 `qt.webChannelTransport` → `bridge.web_badge_changed()`
    通路闭环。
  - **脚本注入**：`inject_window_id.js`（DocumentCreation，定义 `__screen_window_id` 占位）+
    `badge_intercept.js`（DocumentReady，拦截 `navigator.setAppBadge()`）→ QWebEngineProfile
    全局注入，不再依赖 QML `WebEngineScript`（PySide6 6.11 QML 中不可创建）。
  - **Per-window ID**：QML `onLoadingChanged` → `runJavaScript` 注入实际 window ID。
  - **QJSValue snapshot fix**：`_to_native()` 递归调用 `QJSValue.toVariant()`，修复 QML
    `property var` → Python 的 PySide6 不自动转换问题。context_messages windows 目录和
    layout state 不再永远为空。
  - **context_messages 优化**：window 一行紧凑格式（去 `https://` 前缀，title 截断 30 字符），
    事件截断为 3 条（原 5），`bucket.start()` 从 context_messages 移除（已在 startup 调用）。
  - **URL filtering**：`open` 命令拒绝非 http/https scheme。
  - **node venv**：`pyproject.toml` + `uv sync` 独立 venv（trafilatura 模式），
    `NODE.md exec.command: .venv/bin/python`。
  - **已验证**（MCP via moss-as-mcp）：open/focus/close/context_messages/url_reject 全通路，
    WebEngineView 渲染 example.com 成功，QML 窗口无崩溃。

  摩擦点记录：
  - `WebEngineScript` 在 PySide6 6.11 QML 中不可创建（"Element is not creatable"）→ 改用
    Python `QWebEngineProfile.defaultProfile().scripts().insert()` 全局注入
  - `QtWebEngineQuick.initialize()` 必须在 `QApplication` 之前调用

  ### S3: Layout Runtime + More Layouts (done, 2026-08-03)

  提交：待 commit (this session)

  **产出**：curtain 过渡动画 + split 双焦点布局 + _DEFERRED Future 机制。

  - **Curtain 过渡**：`switch_layout(name, rid)` → black Rectangle fade-in 300ms → swap
    layoutName → fade-out 300ms → `bridge.animation_finished(rid)` → Future resolve。
    "command 返回时刻 = 视觉稳定时刻"（Decision 9）。
  - **_DEFERRED 机制**：bridge.py 新增 `_DEFERRED` sentinel + `_ANIMATED_OPS` set。
    `_execute` 对动画 op 返回 `_DEFERRED`，`_on_dispatch` 跳过立即 resolve，Future 留在
    `_futures` 由 `animation_finished(rid)` resolve。rid 通过 dispatch 签名注入 QML。
  - **Split layout**：两个独立 focus Loader（focusLeftLoader/focusRightLoader），
    active 条件 = `layoutName === "split"`，`focus_window(id, slot)` 按 left/right 分派。
    `focusRectLeft()` / `focusRectRight()` 计算左右半区几何。
  - **Top bar**：layout selector 扩展为 `["solo", "split"]`。
  - **Snapshot**：`_refresh_snapshot` 新增 `focusIdLeft` / `focusIdRight` 读取，
    layout slots 增加 `focus_left` / `focus_right` 字段。
  - **context_messages**：split 模式下显示 `left:` / `right:` 替代单 `focus:`。
  - **split state**：`focus(id, slot='left')` / `clear(slot='left')` 默认值更新。
  - **已验证**（MCP via moss-as-mcp）：solo→split→solo 双向过渡正常，双 WebEngineView
    并排渲染 example.com + httpbin.org，context_messages 正确显示 left/right。

  摩擦点记录：
  - QML `focus_window` 函数重复定义导致 "Duplicate method name" → 删掉旧 standalone 版本

  ### 后续：跨平台与 Cell Providing

  原 S4 计划 Ubuntu node + matrix-resources `servers://` 消费 + cell providing 协议，
  现决定不独立推进——与 matrix-resources、desktop-gui 等 feature 合并一起看效果。
  当前 screen node 仅 macOS + PySide6，跨平台能力门控保留在 `screens/` 目录下新增 node
  （INSTALL.md 门控）的机制中。

- 第一期不依赖 matrix-resources：URL 发现走 provide 语义自述；`servers://` 落地后
  screen 可成为它的消费者（一次 list 即恢复全部可开的窗，compact 不遗忘）。
- meta 游离层的动画（d3 式前台聚焦/后台游离的美化）只是实现升级，不动协议。
- 曾考虑并放弃的路线记录：
  - OS 级 WM（AX + xdotool + mss + PyQt6 独立窗口治理）——像素感知回路、无编排权、
    Wayland 逆潮流，降级为将来可选的窄能力 node（启动 app/激活窗口/截图观察），非本体。
  - per-window 子 channel（`screen.<window>` 挂虚拟子 channel）——把呈现拓扑和内容操作
    耦合了；窗内操作应走窗背后 cell 自己的 channel，address 关联。