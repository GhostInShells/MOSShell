---
title: Web Manager — 零依赖 webview 后端
node: nodes/screens/screen
created: 2026-09-19
updated: 2026-09-19
status: in-progress
---

# Web Manager

> screen-manager 的子文档：**零依赖 webview 后端**，当前施工面。
> 语义见 [FEATURE.md](FEATURE.md)；qt 后端见 [qt_compositor.md](qt_compositor.md)。
> 视觉效果的实现细节**不写在这里** —— 它们落在验证页代码的注释里，本文只索引。

## 定位

一个 node = 一个 HTTP 服务 + 一个 ws + 一个单文件 index.html，**与 terminal / file_editor
完全同构**。物料是 iframe（各 node 自己的本地页面），合成器只是摆位。

**零依赖是硬约束**：node 无 `.venv`，`NODE.md` 的 `exec.command = python`（父进程
`sys.executable` 驱动），只依赖 `moss[host]`。terminal / file_editor 已证明此形态可行，
screen 只是在其上再叠 Matrix + ws。

playwright 是高阶用法，不是另一套 screen —— 它驱动同一个页面，补页面内做不到的两件事：
真外部 web 内容（绕不开 X-Frame-Options 的站）与「接管人类已开的浏览器」。

## 关键决策

### WD1 iframe 池：永不 reparent，只写样式

普通 `appendChild` 搬 iframe = 重载（WebKit bug 32848 的原始行为；Chrome 133 才引入
atomic `moveBefore()`）。**物料池是扁平的，iframe 一旦创建永不换父节点**；所有视觉变化
都是改样式。需要跨层时 `moveBefore()` 是逃生口，但主线设计不需要它 —— 需要它 = 分层写歪了。

### WD2 动画：动 wrapper 的裁剪，不动 iframe 的尺寸

iframe 改 width/height 会让内部文档 reflow（与 WebEngineView 同病）。每个物料外套
`overflow:hidden` 的 wrapper：过渡动画只动 wrapper 几何（clip/reveal），iframe 保持最终
尺寸。动 transform 不 scale（scale 会糊）。

**嵌入契约：被嵌入页面必须监听 `resize` 自适应。** 合成器只改 iframe 的 viewport，
不负责内部重排 —— 不监听的页面在切布局 / 全屏时会被裁切或拉伸。实测 avatar（Live2D）
不自适应（窗口变动身体位置炸）、terminal / file_editor 同样不自适应，三者是各自节点的
修法，不是本合成器的职责。

三档工具：重排走 **View Transitions**（快照旧态交叉淡入 + 自动 FLIP，Chrome 111+ /
Safari 18+ / Firefox 144+，降级为瞬时）；物化/退出全屏走 **WAAPI**（`finished` promise
即 await 边界）；hover/徽标走原生 `transition`。

### WD3 层栈落地

`background`（shader/canvas，`pointer-events:none`）→ `stage`（分格，唯一接收指针）→
`chrome`（group rail、topbar）→ `veil`（遮罩/指向）。

- **全屏不是搬运，是插 scrim**：给物料高 z，scrim 插在它之下、其余物料之上，DOM 结构不动。
- **分组切换 = `display:none`**：不重载 iframe，状态保留，后台被 Chromium 自动节流。
- **不给子 iframe 开 `allowfullscreen`**：否则子页能抢走整块屏幕绕过合成器。
- **全屏后焦点在 iframe 内，父页收不到 Esc**：退出按钮必须是 scrim 上的 DOM。

### WD4 veil 两级

| 级别 | 语义 | await | 改模型状态 |
|---|---|---|---|
| **block** | 手势：高亮 / 箭头 / 大字 / 临时交互 | 是（时长即边界） | 否 |
| **idle** | 闲时自跑脚本 | 否 | 否（纯表现） |

**抢占规则**：block 进场 idle 让位，结束后恢复。**帧预算护栏必需**：idle 脚本可被模型写，
坏脚本会被降频或杀掉。

veil = **姿态层**（stage = 内容层）：它是唯一能跨物料作画的地方，Ghost 因此有了"指"的
能力。坐标语义为主（`item` + `region`），像素为边界（靶子在 iframe 内部时，来自截图，
需合成器吐出 stage 几何做桥）。

### WD5 background：数据面已在仓库

- **语音视效消费 `types/topics/audio.py` 的 `AudioSampleTopic`**（双边 `role`、5Hz、
  `rms_db`/`peak`/`spectrum_bins`/`waveform`，no PCM）—— 零新协议。聆听 = 底部 ECG 锯齿
  （`waveform`）+ 超阈红（`spectrum_bins`）；说话 = 字符雨。
- **不做状态驱动**（idle/thinking/speaking → uniform）：`types/topics/ghost.py` 那条协议
  **尚未定案**，为一个没有生产者的 schema 做设计是它明说该避免的事。第一版只靠
  `role` + `rms_db` 区分在说话 / 被听到。
- **MOSS 文字"活过来"**：文字是遮罩，动的是它背后的东西（canvas 内
  `destination-in` 文字遮罩，或 `background-clip: text`）。
- **字符雨**：头部下落 + 定长尾迹，不是无限加长。canvas 2D + 预渲染字形图集，尾迹用
  半透明黑覆盖层；只在说话时活动 → 守空闲冻结。底噪不发（`rms_db` 阈值）。
- **输入框两个位置**：空屏时是主角（屏中下），有物料时退成细带。层属 chrome（background
  不接收指针）。topbar 是人发起，与 QA（系统发起）别合流。
- **常驻微指示器**：全幅视效在 background，但「人在说话被听到了」必须始终可感知 —— 否则
  物料上屏时人以为自己没被听到。

### WD6 group rail（左侧）

**左侧是 group，不是 item** —— item 上屏 = 布局写权 = 模型专属，人点 item 上屏是主权倒挂。

tile = name 卡片 + 数量角标 n + notify 位 + activate 效果。**不做缩略图骨架**：它站在
中间态，既不如实时缩略图有信息，又不如文本可扫，且对切组决策无增量。

activate 效果是必需品：模型切组时人可能在看别处，绿点/边框轮转 n 秒是"谁干的"的可读信号。
人自己点 = 立即生效，不需要提示。

## 待验假设与验证页（按依赖排序）

| 页 | 假设 | 哑掉它的判据 |
|---|---|---|
| **`stage.html`** | 不 reparent + 只改样式 → 换格/换组**不重载** | 计时器 + 输入框状态丢失 = 假设死（这是地基，先验） |
| `veil.html` | canvas 上 block / idle 两种生命周期能干净共存 | block 进场出现 idle 残影/擦除 = 抢占规则没定对 |
| `background.html` | 文字遮罩够廉价；字符雨 canvas 2D 能 60fps | 掉到 30fps 以下 = 该换 WebGL backend |

`stage.html` 同时验 rail。页面是**验证假设的载体**，不是一次性 demo —— 被验证的结论进
本文，代码将来搬进 node，不重写。

### 验证方法论（dogfood，也是本 workstream 的判据）

不写 mock 单测去模拟浏览器；**用 playwright node 驱动同一个验证页**，headed 窗口 + `say`
语音解说，人机共同观察。这本身就是 screen-manager 要交付的能力（模型驱动屏幕 + 轨道协同）
—— 用产品验证产品。bash 只能跑 headless 脚本再读输出，MCP 要手写 plumbing；这里 playwright
/say/nodes 已经是 first-class channel，模型在 turn 内 emit→observe→调整。

- **保活探针**：srcdoc iframe 内跑计时器 + 输入框，操作后看值是否保留。
- **序位探针**：页面 `readout` 同时打印 DOM 顺序与按 grid 坐标排序的视觉顺序，两者分叉 = 重排没动 DOM。
- **轨道协同**：每个 block 动作带声明时长，`say` 与之同轴 —— 语音和动作的相位差肉眼可判。

### stage.html 已验证（2026-09-19）

- ✅ 6 item → `cols=ceil(sqrt(6))` 的 3×2 大分屏
- ✅ swap 重排不动 DOM（`视觉:` 变、`DOM:` 不变）
- ✅ 全屏 = scrim 插层 + z-index，不改结构
- ❌→✅ **全屏退出入口必须在 scrim 之上**（WD3）—— 首版把退出按钮放控制条，被 scrim
  盖住点不到，dogfood 当场暴露

### veil.html 已验证（2026-09-19）

- ✅ 单 canvas 上 block / idle 两级生命周期干净共存（单 rAF 循环 + 每帧 `clearRect` → 无残影）
- ✅ 抢占规则：block 进场 idle 让位、结束恢复；恢复不重放（idle 是时间函数，block 期间不推进）
- ✅ await 边界：`block()` 按时长 resolve（mark→arrow→text 顺序解析 1200/1200/1500ms）
- ✅ 语义坐标桥：`geometry()` 返回 item rect，`mark(item, region)` / `arrow(from, to)`
  用语义坐标 —— 模型不算像素，这条桥是将来 vision 截图 → veil 绘制的地基

### background.html 已验证（2026-09-19）

- ✅ 字符雨 canvas 2D + 字形图集：ghost 说话时 **75fps**（判据：30fps 换 WebGL，目标 60fps）
- ✅ MOSS 文字遮罩（`background-clip: text`）：单 DOM + GPU 合成，成本可忽略
- ✅ 底噪不发：`rms_db` 低于阈值不产生字符 → 静默时冻结（KD6）
- ✅ 数据面 = mock `AudioSampleTopic`（5Hz，`role`/`rms_db`/`spectrum_bins`/`waveform`）；
  真实数据由 node 订阅 `types/topics/audio.py` 的 topic，页面零改动

## 并行两条线

- **页面线**（浏览器）验躯体；**Python 线**（`ScreenModel` + channel + notice/回执的纯单测）
  不依赖浏览器，可并行。
- 两线在 **ws 协议**处汇合，协议**复用 terminal / file_editor 的帧协议**（不发明），要早钉。
- 「零依赖」不在页面里验：单独一步最小 `main.py` + HTTP 服务，确认父进程
  `sys.executable` 下可跑。
