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
| V3 | superseded — per-group fullscreen 已删，换成**视图** `focus`；**切组即撤销焦点**（判据：选一个基础视图 = 撤销覆盖） |
| V4 | superseded — 桌面圆盘那一套退役，home 改成**池子**（螺旋铺位，见 V9） |
| V5 | `mock_scene` 连调两次 → 不叠 iframe（snapshot reconcile） |
| V6 | 字符雨字符集：**2026-10-02 已按需求换掉** —— 现在是拉丁字母 + 希腊字母（原来那版 CJK+假名）。留此条只为记住"字符集是一处常量、随时可换" |
| V7 | superseded（**并附一条诊断**）—— 圆盘点击那条路随圆盘一起消失。附带结论：`aside` 是旁路、**忙的时候会丢且不报错**，不能承载"人做了什么"；人的视图动作现在写**状态**（`view_by`），见 FEATURE「呈现模型」 |

| # | 状态 | 判据 |
|---|------|------|
| V9 | **verified 2026-10-10** | **池子**：螺旋铺位不重叠；指针进舞台 → 自转缓动到停；hover **原地**放大（207→281px，z 198→599，停 3 秒不抖）；点瓷砖 → 整屏（`elementFromPoint` 命中 IFRAME）；点侧边栏 / Esc → 退出焦点、池子回来、栏还原 188 |
| V10 | open | 池子在 **14 张以上**时的密度与标签可读性（当前 cap = 14，远端只留缩略） |
| V11 | **verified 2026-10-10** | **带图的瓷砖与"运动场"**：瓷砖上的图是节点给的地址（camera 的 live tile 实机验过）；**漂移按视口算**（同屏 1280→2560 观感一致；实测 12 秒内 x 峰谷 53px / y 32px，改前只有 16px）；**指针在场上时整个运动场缓动停住**（不只自转）；只有一件时落在**旋心**；一帧中位 13.3ms、最差 14.5ms（一张 live 图不构成压力） |
| V8 | `open` 返回 index 句柄；`navigate` 改自建 item url 且 iframe 原地重载；对 webview item navigate 越权中断 |

## 未接能力

| # | 状态 | 能力 | 依赖 |
|---|------|------|------|
| W1 | **fixed** | **background 真实音频数据面** — `TopicAudioSource` 订阅 `AudioSampleTopic`；`main.py` 默认走总线，`--mock-audio` 退为调试（`e8467e05`）。 | audio topic 发布方 ✅ |
| W2 | open | **veil 视觉坐标桥** — 页面有 `geometry()`，模型侧无取几何/截图通道 | vision 通道 |
| W3 | superseded | **chrome 交互面** — 侧边 `#chrome` 已拆；输入框、通知、历史整体移交 `ghost-in-web` | ghost-in-web |
| W4 | open | **被嵌入页面的 resize 自适应契约** — 合成器只改 iframe viewport，不负责内部重排；avatar/terminal/file_editor 不监听 resize | 各节点自查 |
| W5 | open | **playwright 高阶面** — 真外部 web 内容（绕 X-Frame-Options）与接管浏览器 | playwright node |
| W6 | **fixed** | **调试面退役** — 2026-10-10 落地：node 加 `--debug`（自开窗带 `?debug=1`），页面按它决定；`#st`（组·role·dB）、帧率与几何、活动日志一律**收进侧栏底部**，默认全隐藏（隐藏走 CSS，绘制逻辑不变）。`?mode=` 画法对照同开关。**一处例外**：断线常驻可见（侧栏顶上的点 + 收起时边条变红；水位标同步灰掉）。**另一处例外**：stdout 的 `[client]` 自查**不进**这个开关——它不到屏幕上，是模型的探针 | 本 feature |
| W7 | **fixed** | **听侧（用户音频）的呈现与接线** — 见下方「W7 听侧」。 | listener / 本 feature |
| W9 | **fixed** | **face：节点自报"脸"** — 2026-10-10 落地（协议 `b377c7d8` + 两个节点 `7543a751`）：`face_line` / `face_art` 进 **condition**（不进 declaration——它在 provide 时就被冻结）；`set_face()` 与 touch/notify 同形；**图是一个可选的地址**，"它会自己变"是生产者的事（camera 用自己那路流当图）；瓷砖自下而上画 图→线→名；脸变更走**自己的 delta 帧**（不让闲聊重画舞台）| — |
| W10 | open | **别名累加** — 重启即涨（`screen` → `screen_2` → `screen_3`），今天第三次撞上；人机都在用别名索引，涨到 `_9` 会难读 | 本 feature |
| W11 | open | **池子的 show 模式** — 今天只做了 home 的静止场；"我要讲一件事"时把场**推近**（电影参考）留作第二步 | 本 feature |
| W12 | uncertain | `arrange` 现在也会**撤销焦点**（按"选一个基础视图 = 撤销覆盖"推出，不是特例）。若判定过强，一处回调 | 本 feature |
| W14 | open | **侧栏第二层**：组里的 item（点条目 = 满幅）＋ **node 生死开关**——主权里"人管 node 生死"至今没落地 | 本 feature |
| W15 | open | **多页面覆盖上报**：同一地址开了多个页面时，谁在看哪一个（人侧 W8 多屏的前置） | 本 feature |
| W13 | **fixed** | **页面版本指纹** — `/__ver` 返回 index.html 的 sha1 前 10 位；页面每 3 秒问一次，**版本一变自己刷新**。动机：开着的旧窗口会把"旧世界"的数据混进同一份 stdout（2026-10-10 白查过一轮） | 本 feature |
| W8 | open | **多屏机制未验证** — 一台机器上多个 screen node、或多显示面之间的分工与主权（谁排布、谁全屏、人怎么看）今日未测 | 本 feature |
| W16 | **fixed** | **窗口标题栏** — 每窗一条、由屏画（label 来自 node 声明，右端是这一块的 maximize / restore）；图标走 SVG 方框不赌字形；窗口感来自"栏比内容亮一档"而不是加框；accent 只标 focused。`0cde24bb` + `257de158`。判据：任意两个 node 并排时标题栏结构与位置完全一致，且 node 页面内不再出现自己的名字 | 本 feature |
| W17 | open | **局域网网关（`--lan`）** — screen 内组合的可拆模块：唯一对外 listener、子域或端口路由、HTTP+WS 代理、静态拒绝页（不连上游）、Host/Origin 白名单、TLS。路由表来自 `WebViewClient`，slug 由 title 派生并持久化；策略文件 0600。上游保持 127.0.0.1 | 本 feature |
| W18 | open | **URL 治理三处** — `WebViewDeclaration` 继续只报 loopback 事实；channel 的 `url` 面向人时给 public URL；`add_local()` 返回 `/local/<token>` 相对地址；screen 拆 `--host`（绑定）与 `--public-url`（宣告）。判据：局域网浏览器打开任意 item 与本地一致 | W17 |
| W19 | open | **认证与设备凭证** — 全模式登录；Argon2id（不加密）；长期 device token + 逐台吊销；一次性配对码；本地 CLI/TUI 做注册与重置（不依赖 screen 存活）；限速；跨子域 cookie；WS 握手校验 | W17 |

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
| O4 | **decided** | **人切组 / 顶部导航栏的归属（2026-10-20 裁决）** — 拆成两家：**切组留在侧栏**（人的"看哪"入口不动，标题栏只管"这一块"）；**每个窗口的标题栏由屏画**——左端 = node 声明里的 label，右端 = 这一块自己的 maximize / restore（走已有的 `focus`，不造第二套语义）。node 不再在自己页面里写名字。**全局顶栏被否**（它回答的是"窗口怎么占满屏"，不是"砖怎么放大"）。notify 抽屉归屏，尚未落地。详见 FEATURE「窗口 chrome 的归属」 |
| O5 | open | **LAN 路由与 TLS 选型（2026-10-24）** — 子域（需泛解析，WebAuthn 可用）vs 独立端口（零 DNS）；TLS 用 Caddy internal CA / mkcert / Tailscale。默认值与 DNS 供给方式待定。论证见 `discuss/2026-10-24_lan_access_and_gateway.md` |

## 2026-10-10 新登记（现场）

| # | 状态 | 问题 |
|---|------|------|
| T1 | open | **CTML 参数被隐式转换就炸** — `matrix.nodes:read_output limit:str="25"` 直接报 `'>' not supported between instances of 'str' and 'int'`；同一族里 `arrange(ids)` 传 int / list 也炸（D03 撞 int、D08 撞 list）。签名即契约：要么容错，要么错得说清"这里要什么类型"。跨 node 工具链的共性问题，暂记此处（归属待裁） |
| T2 | open | **平台侧几何自报已在线上** — 屏幕页每 2s 落一行 `[client]`（viewW/H、dpr、baseY、rms、riseMax），模型可自查，不必再让人念屏幕。Retina 画布那处修复（`f9504cc2`→`7723cc86`）在 dpr=2 下复现通过。**它不是"调试面"**（W6）：那条线上不了屏幕，是模型的眼睛，所以默认开着、另给开关。待做：把它写成 V 列的判据，别留在口头 |
| T3 | open | **co_browser 的 `exec` 会无声丢弃（2026-10-11 实测 6 次）** — exec 体里用 `page.mouse.move` 把光标移到某块瓷砖上、之后再 `page.evaluate`，这条调用**直接消失**：无报错、无回执、不进 `history`；而页面本身是活的（后续 exec 正常、rAF 在跑、`peek` 状态可读）。改成 `aexec` 则正常返回。怀疑在 exec 的完成/回传路径，不在页面侧。**共同特征还没锁定**（同形状的短脚本有时能过），要单独查 `co_browser` 的 exec 生命周期 |
