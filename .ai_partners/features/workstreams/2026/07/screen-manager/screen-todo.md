# Screen Manager TODO

> screen-manager 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修, 带 commit) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug) / `superseded`(被重写取代)。
> 由 `screen-manager` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。
> 编号在本 workstream 内唯一；跨 workstream 引用时写 `screen D3`。

> **2026-09-21 全量重写**：store / projection / channel / surface / index.html 按新语义
> （桌面为一级公民、至多一组、per-group layout+fullscreen、自动物化、dismiss/destroy 分家）
> 重写。D1–D4 是旧结构的缺陷，随重写**消失**，标 `superseded`。新结构本身尚未 dogfood，
> 其行为是**判据候选**，不是已复现的事实 —— 见下方「待 dogfood」。
>
> **2026-10-02**：background 的音频数据面接通（**W1 关闭**，`e8467e05`）—— 雨改为
> **逐列由真实频谱驱动**。新增 W6–W8：这两天要把它推成正式产品，这三件是收尾项。

## 缺陷（旧结构，已随重写消失）

| # | 状态 | Pri | 问题 |
|---|------|-----|------|
| D1 | superseded | P0 | 切组不撤全屏 —— 旧 `activate` 清全局 `_fullscreen` 但帧不带；新结构 fullscreen 是 per-group，切组天然切换 |
| D2 | superseded | P0 | 关掉全屏 item 不撤全屏 —— 旧 close 不补帧；新结构 dismiss/destroy 走 `_detach` 清 fullscreen + `state` 帧全量下发 |
| D3 | superseded | P1 | snapshot 不去重 → 重复 slot —— 旧 makeSlot 无幂等；新 snapshot 带全量 `items` + reconcile |
| D4 | superseded | P1 | family/dir 模型级 —— 新结构 layout 随 group 存（`Group` 对象），组删随组忘 |

## 待 dogfood（新结构的判据，尚未在浏览器复现）

| # | 判据 |
|---|------|
| V1 | `open` 落桌面 → 代理圆盘出现，iframe 隐藏；`arrange` 拉入组 → 圆盘消失、iframe 分格 |
| V2 | 切组 / 桌面↔组 有过渡，不闪屏（当前 `.slot` 有 CSS `grid-row/column` transition，是否够用待验） |
| V3 | 全屏某 item → 切组 → 回来仍全屏（per-group fullscreen 记忆） |
| V4 | 桌面代理圆盘物理（花束/布朗/环绕）不抖不漂出，点击命中准确 |
| V5 | `mock_scene` 连调两次 → 不叠 iframe（snapshot reconcile） |
| V6 | 字符雨字符集：**2026-10-02 已按需求换掉** —— 现在是拉丁字母 + 希腊字母（原来那版 CJK+假名）。留此条只为记住"字符集是一处常量、随时可换" |
| V7 | 桌面代理点击 → aside 到达模型（`tap` uplink） |
| V8 | `open` 返回 index 句柄；`navigate` 改自建 item url 且 iframe 原地重载；对 webview item navigate 越权中断 |

## 未接能力

| # | 状态 | 能力 | 依赖 |
|---|------|------|------|
| W1 | **fixed** | **background 真实音频数据面** — `TopicAudioSource` 订阅 `AudioSampleTopic`；`main.py` 默认走总线，`--mock-audio` 退为调试（`e8467e05`）。 | audio topic 发布方 ✅ |
| W2 | open | **veil 视觉坐标桥** — 页面有 `geometry()`，模型侧无取几何/截图通道 | vision 通道 |
| W3 | superseded | **chrome 交互面** — 侧边 `#chrome` 已拆；输入框、通知、历史整体移交 `ghost-in-web` | ghost-in-web |
| W4 | open | **被嵌入页面的 resize 自适应契约** — 合成器只改 iframe viewport，不负责内部重排；avatar/terminal/file_editor 不监听 resize | 各节点自查 |
| W5 | open | **playwright 高阶面** — 真外部 web 内容（绕 X-Frame-Options）与接管浏览器 | playwright node |
| W6 | open | **调试面退役** — 打磨期加的东西必须在产品化前收口：`probe_audio` 命令、每 10 帧一行 stdout 频谱、`?mode=hybrid\|bars\|edge` 对照画法、`mock_scene`/`mock_audio`。目标：默认面只剩产品本身，调试面挂在 `--debug`（或一个显式开关）下 | 本 feature |
| W7 | **fixed** | **听侧（用户音频）的呈现与接线** — 见下方「W7 听侧」。 | listener / 本 feature |
| W8 | open | **多屏机制未验证** — 一台机器上多个 screen node、或多显示面之间的分工与主权（谁排布、谁全屏、人怎么看）今日未测 | 本 feature |

## W7 听侧（用户音频）

**三个 bug 叠在一起**（D04 夜发现，前两个是断的、第三个才轮到画法）：

1. **总线断的**：`role=user` 的 `AudioSampleTopic` 从来没被发布过。说侧桥在
   `host/voice/interleaved.py` 里装了，听侧音频桥 `with_audio_sample_service` 却只有
   `host/nodes/listener_node.py` 那条路调 —— 幽灵走的 interleaved 路径漏了。
   修法：装线收进 `ListenerController.__aenter__`，跟 clause 桥并排（**这条约定归
   voice 模块自己履行**，谁组装 controller 都不必记得补一句）；`with_audio_sample_service`
   补幂等，`listener_node` 那条显式路径不双发。
2. **判据错的**：页面把整段 `rms_db` 当门限，而远场 / 骨传导下 rms 轻易掉到 −60，
   频谱里却有 −17dB 的低频。改用 bins 的最强值判"活"。
3. **画法不搭**：第一版是底部频谱柱，横向满格扫一条，跟上面的雨是两个世界。

**最终画法（他定方向、我定实现）**：把频谱柱换成**从底部长上来的字** —— 同一套字形、
同一套配色，方向相反。我的声音从上往下掉，人的声音从下往上长。一层 = 一个 `GH`，
`RISE_ROWS` 层封顶。关键几笔：

- **钟形包络**（`RISE_BELL`）：字不该挤在某一侧，乘一条 σ≈屏宽 32% 的正态曲线，
  中段最盛、两侧渐稀。这是**画法不是数据** —— 频谱本身长什么样仍由 bins 决定。
- **横向平滑**（`RISE_SMOOTH`）：16 个 bin 抽到 28 列后做两遍扩散，把频带跳变的锯齿
  抹成一道连绵的浪。
- **与雨错开半格**（`RISE_OFFSET = GH/2`）：两条浪不落在同一行上，视觉上不打架。
- **dB 窗口固定**（`BAR_DB_FLOOR = -52`、`BAR_DB_CEIL = -20`）：不用自适应 —— 自适应会把
  安静时的底噪放大成满格。地板定在 −52 是因为静音底噪落在 −55 上下，低了会留一排矮字。
- **两侧分家存**（`feed` / `userFeed`）：共用一条总线，谁说话谁发帧；按 role 分家，
  另一侧说话时不会把这一侧抹掉。

**未验**：`RISE_*` 与 `BAR_DB_*` 的数值是照他现场反馈调的，尚未做跨设备的独立回归。
听侧链路（ASR）弱网下的故障恢复/传导缺失仍是另一条账（见 D02 journal、`REGRESSION.md`）。

## 设计问题

| # | 状态 | 问题 |
|---|------|------|
| O1 | superseded | **输入框未接线** — `#input` 随侧边栏一并拆除；输入/通知/历史归 `ghost-in-web` |
| O2 | open | **三档命名** — node 已定 `webview_screen`（档位进名字），但 qt/os 档命名、channel 命名是否跟档位绑定仍未定 |
| O3 | open | **qt 档 node 落点** — `qt_compositor.md` 说另起目录，新目录名待定 |
| O4 | open | **人切组 / 顶部导航栏的归属** — FEATURE 写"输入·通知·历史整体移交 `ghost-in-web`"，但按第三轮讨论，人"管看哪"（切组、全屏开关）的入口要留在屏幕页顶栏。两者边界未定，需裁决：切组 UI 住哪、通知住哪、扩展还剩下什么 |

## 2026-10-10 新登记（现场）

| # | 状态 | 问题 |
|---|------|------|
| T1 | open | **CTML 参数被隐式转换就炸** — `matrix.nodes:read_output limit:str="25"` 直接报 `'>' not supported between instances of 'str' and 'int'`；同一族里 `arrange(ids)` 传 int / list 也炸（D03 撞 int、D08 撞 list）。签名即契约：要么容错，要么错得说清"这里要什么类型"。跨 node 工具链的共性问题，暂记此处（归属待裁） |
| T2 | open | **平台侧几何自报已在线上** — 屏幕页每 2s 落一行 `[client]`（viewW/H、dpr、baseY、rms、riseMax），模型可自查，不必再让人念屏幕。Retina 画布那处修复（`f9504cc2`→`7723cc86`）在 dpr=2 下复现通过。待做：把它写成 V 列的判据，别留在口头 |
