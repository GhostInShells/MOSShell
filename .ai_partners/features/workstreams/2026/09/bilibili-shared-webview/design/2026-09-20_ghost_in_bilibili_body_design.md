# ghost-in-bilibili 躯体设计:共享观看的身份、通讯与授权模型

> 2026-09-20 定稿。本文是**声明式结论** —— 只写设计是什么,不写怎么讨论出来的。
> 碰撞过程见 feature 的 `discuss/`(待补)。
>
> 前身:`nodes/browsers/ghost-in-bilibili/` 下已提交的代码(2381127a / c176b352)
> **是探针,不是实现方案** —— 它验证了"插件 → 本地 node → channel"这条路走得通,
> 也暴露了 CSP 禁 eval 与 url 当页面 key 两个坑。本文不继承它的结构。

## 背景与动机

ghost 陪人类一起看 bilibili。这是"人机共享协作"的开箱做法之一 —— 不是把视频当任务
处理,而是两个意识在同一段时间轴上共存。

核心不是控制播放,是**字幕**:有了字幕 ghost 才有时间戳认知,才能说出"刚才 3:42 那句"
并与人讨论。没有字幕,它就只是个遥控器。

页面侧的动机是**非打扰**:人正在看片,ghost 的界面不能抢屏幕。所以球 + 悬停面板;
球本身承载授权语义 —— 主球(灰=停 / 绿=运行)+ 环绕卫星(每颗 = 一项能力,绿=授权)。

## 设计约束

限定的验证场景:

> **单机、单人类、一人一 ghost,通过 Chrome 扩展 + 本地 node 共享观看 B 站视频。**

明确不考虑(预留为 future work):多人类共享同一页面、跨机远程、非 B 站的通用 webview
页面语义、登录态与个人账号数据的使用。

一条**不追求**的事:**不做可靠的授权边界**。node 侧是软边界(与 `matrix.warrant` 的
定位一致 —— 模型能自我迭代就会自我授权)。真正的物理门在浏览器里:授权状态由扩展持有,
node 只是镜像。设计上 node **不提供任何授予授权的命令**。

## 核心设计

### 1. 身份:label 是唯一持久身份,bvid 是易变属性

| 层 | 定名 | 载体 | 生命周期 | 可靠? |
|---|---|---|---|---|
| 浏览器实例 | **session** | 扩展生成,存 `chrome.storage.local`,WS `hello` 上报 | 跨 SW 重启、跨浏览器重启稳定 | 可靠 |
| 窗口 | **tab** | **SW 从 `sender.tab.id` 盖戳**,内容脚本不自行发明身份 | tab 存续期 | 可靠 |
| 页面身份 | **label**(`p1`/`p2`) | node 给 `(session, tab)` 分配 | 同 tab | **唯一持久身份** |
| 内容 | bvid / title / url | 扩展随内容变化上报 | 一个视频 | **易变,不可作 key** |

**不变量:label 是唯一持久身份;bvid 是页面的易变属性。** B 站会**自动播放** —— 一个
视频播完自动跳下一个,bvid 在 label 不变的情况下连续变化。所以任何"用 bvid 当身份/key"
的做法都是错的:页面身份只能由 (session, tab) 派生,bvid 只是"当前在播什么"的一个
随时会变的字段,由扩展在它变化时上报(见 §6 字幕、未验证声明)。

探针的错有两层:用带 `trackid` 的一次性 url 当队列 key(队列孤儿化),以及把 bvid 当
内容身份。本文明确废止,内容变化只表现为 label 的**属性更新**,不产生新身份。

同一 bvid 开两个 tab = 两个 label,是两个独立页面身份;字幕文件按 bvid 存只是**内容缓存**
按内容 key,与页面身份无关。

### 2. 通讯:一个 SW 一条 WS

**为什么只能是 WS。** MV3 下 content script 不能直接发跨域请求,网络边界只能是
background service worker。而 SW 是**一个扩展一个**,不是一 tab 一个 —— 所以一条连接
天然复用给所有 tab,不需要自造 multiplex。

对 SSE 的否决:EventSource 同样受 CORS 管辖,内容脚本里开不出来;只能开在 SW 里,而
长连接在 SW 的生命周期里是最脆的一环(见「未验证声明」)。

帧协议(JSON,type 驱动):

| 方向 | type | 载荷 | 语义 |
|---|---|---|---|
| 下行 | `cmd` | `{cid, tab, action, value}` | 模型要页面上做的事(阻塞,按 cid 回执) |
| 下行 | `say` | `{tab, text}` | 往该页面板回话(不等待回复) |
| 上行 | `hello` | `{session, boot, ua}` | 连上即登记 session(SW 被回收后重连是常态) |
| 上行 | `content` | `{tab, bvid, title, url}` | tab 开 / 导航 / 自动播放换视频(内容变化) |
| 上行 | `auth` | `{tab, group\|null, on}` | group=null 是主球(presence),否则是卫星(group) |
| 上行 | `state` | `{tab, t, paused, rate, duration}` | 1~2Hz 实时播放状态 |
| 上行 | `input` | `{tab, text}` | 人在该页面板里说的话 |
| 上行 | `result` | `{cid, ok, result\|error}` | `cmd` 的完成回执 |
| 上行 | `bye` | `{tab}` | tab 关闭 |

**线上只谈 `tab`,不谈 `label`。** label 是 node 内部抽象(由 (session, tab) 派生);
SW 在每帧上盖 `sender.tab.id`,node 再把 tab 映射回 label。字幕行 `line` 不下发 ——
node 有全文轨道后,只靠 currentTime 自己切窗口,扩展只报时间。

**命令的两条路径**都落在 MOSS 原生原语上,不发明新东西:

| | 命令侧 | 结果侧 |
|---|---|---|
| 阻塞 | `blocking=True` + `timeout`,await 一个 cid future | 返回值直接给模型 |
| 后台 | `blocking=False`,发完即返回 | 完成后 `CommandUtil.send_signal`(`aside`)推回 |

人的面板输入走 `CommandUtil.send_input_signal`,`input` signal 的语义正是"来自外部的、
期待回答的消息"。

### 3. channel 面:一个 channel,page 是参数

命令面(全部以一个页面 label 为第一参数):

| 命令 | 语义 |
|---|---|
| `play` / `pause` / `seek` / `speed` | 视频控制(需 `control` 组授权) |
| `subtitle(page, start, end)` | 字幕**时间区间查询**(需 `subtitle` 组授权) |
| `say(page, text)` | 往该页面的面板回话 |

**否决 root → page → group 的父子树。** 树每一层只能表达一个维度,而授权是
(页面 × 能力组) 的二维格 —— 无论把哪一维放进树,另一维仍要落到命令层解决,那套机制
反正要写。树唯一的结构性收益是跨页并行,而这里所有命令都是几十毫秒的往返(唯一的慢
命令是人等待,用 `blocking=False` 即可覆盖)。代价却是每个 tab 开合都要抖动 facade。

**触发重新考虑的条件**:出现一个必须阻塞人类延迟(秒级以上)且不能被 `blocking=False`
化解的命令。届时再评估。

`say` 是**对话**不是请求-响应:页面的输入输出是"单页面对话轨迹治理"的便利面
(语音才是主对话通道)。`say` 发完即返回,人的回答作为下一个 `input` signal 异步到达。

### 4. 授权:状态进 notice,靠模型推理

授权格 = **(页面 × 能力组)**。能力组即卫星:

| 组 | 卫星 | 覆盖 |
|---|---|---|
| (主球) | 灰 / 绿 | **presence** —— 该页面存在,ghost 知道人类在看什么 |
| `sense` | 视频状态 | 实时状态流(currentTime / 播放态 / 倍速)+ 实时字幕行 |
| `control` | 视频控制 | play / pause / seek / speed |
| `subtitle` | 字幕 | 全文轨道的抓取与持有 |
| `interact` | 弹幕-评论 | 弹幕与评论的读(写待定) |

**否决三种替代:**

- **`gate` / `mount_child` 机制。** 它是**披露**不是授权 —— `mount_child` 是模型可调的
  命令,模型能自己挂载,等于没有门。
- **`available` 谓词。** 它不是 per-page 的:授权粒度是命令级的,表达不了"p1 已授权、
  p2 未授权",只能退化成"任一页授权则命令可见"。
- **"未授权 = 命令不可见"。** 不可见会剥夺 ghost 的**协商能力** —— 命令整个消失时它连
  "能给 p2 授权吗"都说不出来。看得见 + 明确拒绝更好。

**落地形态**:授权状态进 per-page named notice,配合 instruction 说明语义,模型自己推理;
用错就拿到明确的 observe("p2 的 `control` 未授权,请人类点对应卫星")。这是 soft boundary
的自然表达 —— 诚实,且给模型下一步。

**逐条审批按钮:砍掉。** 弹窗审批与卫星球是两套控制,并存必然打架(已绿的组里再逐条
弹窗,人会觉得幽灵没在听自己)。卫星球是**人类意志的唯一入口**,ghost 想要未授权的能力
就在 panel 里说话请求,人的手去点卫星。

### 5. 感知分层

| 层 | 载体 | 内容 | 变化率 |
|---|---|---|---|
| 冷 | `instruction` | 授权语义、label 与窗口的对应关系 | 几乎不变 |
| 温 | `named_notices` | **每页一条**:存在 + 标题 + 各能力组的授权状态 | 慢 |
| 热 | `context_messages` | **每页一条**:当前秒 + 滚动字幕窗口(2~4 句) | 热 |

分工的硬理由:热数据**必须**留在 `context_messages`。`named_notices` 的语义是"文本一变
就整片重发",秒级变化会烧 token。反之 `named_notices` 的 `None`=移除 / `""`=未变 语义
正好承载页面的生灭 —— tab 关掉时模型收到 `<p1 removed/>`,这是 notice 层独有的能力。

滚动窗口(而非单句)是刻意的:ghost 要能引用"你刚才那句",需要连续性。

### 6. 字幕:两条线,读取是时间区间查询

| 线 | 来源 | 去向 |
|---|---|---|
| **实时行** | 扩展侧取当前句 | `state` 流 → `context_messages` |
| **全文** | 卫星授权的那一瞬间,扩展在页面内走完字幕逻辑拿到轨道 | 剥隐私后 POST 给 node → 存 `matrix.home/subtitles/<bvid>.json` |

授权瞬间抓取是刻意的:那一刻人类已经点了绿球,是唯一无需额外打扰的采样时机。此后
**文件存在性 = available**。

字幕文件按 bvid 存,但 bvid 是**当前内容的缓存 key**,不是身份:自动播放把内容切到
下一个视频时,`available` 针对新 bvid 重新判定 —— 旧文件不删(缓存),新内容没有文件
就退回实时行。

**读取不是"读文件"**:channel 暴露的是时间区间查询,不是文件路径。把整个轨道倒进上下文
不可用 —— 30 分钟视频约 400 句。探针 instruction 里那句"返回值是文件路径,读文件取内容"
与本节相反,已废止。

一个推论:**文件存在后,扩展只需上报 currentTime**,node 自己能从轨道切窗口;文件不存在
时退回扩展 DOM 抓的当前行。两个源,文件优先。

隐私边界在扩展侧剥离,node 只收视频自身的内容:轨道只留 `{from, to, content}`,
uid / 昵称 / 账号 / cookie 一律不落盘、不过线。

### 7. 安全

1. **WS 必须校验 Origin 白名单。** 绑在 127.0.0.1 的 WS,**浏览器里任何一个网页都能连** ——
   不校验的话,人类随手打开的一个页面就能驱动他的扩展。允许 `chrome-extension://<id>`;
   启动时打出实际观测到的 Origin,让人类 pin 进配置(unpacked 扩展的 id 由路径派生,
   manifest 加 `key` 字段才能跨机稳定)。
2. **模型的权限 = 扩展里枚举的动作表。** 不下发任意 JS。这不只是 CSP 的结论,是安全性质:
   ghost 对页面的权力**恰好**是枚举出来的那几个动作。
3. **`say` 往页面写文本必须 `textContent`,永不 `innerHTML`。** 模型产出的文本进 DOM 是
   XSS 面。(探针 `index.html` 的事件日志就是未转义拼 `innerHTML` 的 —— 那页重写,模式不继承。)
4. 绑 `127.0.0.1`,永不 `0.0.0.0`。

## 与现有范式的关系

- **`nodes/screens/screen_manager`** 是最近的先例:一个进程两张脸(channel 给 ghost,
  surface 给人类),人的动作经 `matrix.send_signal_to_ghost` 回流。本设计沿用该结构,
  把 HTTP + 轮询换成 WS,并引入 session 身份(screen_manager 只有一个人类 surface,
  不需要跨浏览器实例的身份)。
- **signals 体系**:人的面板输入 → `input`;后台命令完成 → `aside`。两者都不打断当前思考。
- **feishu / 语音等既有对话通道**:本设计不替代它们。语音是主对话通道,页面 panel 是
  单页对话轨迹治理的便利面。

## 未验证声明

以下全部待探针 #2 验证 —— 它们错了,通讯设计要重来:

1. **MV3 service worker 能否稳定持有 WS。** 依赖心跳续命避开 ~30s 空闲回收;心跳间隔与
   回收的边界需要实测。
2. **content script 跨域的真实边界。** 已知它在 MV3 下不能直接跨域;但页面的 CSP
   (`connect-src`) 对它到底有没有约束,需要实测确认 —— 已知同类事实:页面 CSP 禁
   `unsafe-eval` 时,内容脚本里的 `new Function` 同样被禁(与"隔离世界不受页面 CSP 约束"
   的直觉相反)。
3. **B 站页面内字幕轨道的可得性。** 卫星授权瞬间要"走完字幕逻辑拿到轨道",具体走播放器
   对象、接口请求还是资源嗅探,未定。DOM 只能拿到当前行。
4. **弹幕与评论的读取接口稳定性。**
5. **unpacked 扩展 id 的跨机稳定性。**
6. **自动播放 / 站内跳转时,内容脚本如何可靠检测"内容变了"。** url 可能不变(播放器内
   切源)或变,不能只盯 `location.href`;需要看 video 元素的 src 或播放器上报的 bvid。
   这是"bvid 是易变属性"在扩展侧的落地 —— 检测漏了,ghost 会拿旧 bvid 的认知对应当前
   内容。

## 关联文档

- `../FEATURE.md` — 本 feature 的索引与决策摘要
- `nodes/screens/screen_manager/src/ghoshell_screen_manager/surface.py` — 人类 surface 范式
- `src/ghoshell_moss/core/blueprint/channel_builder.py` — channel 三层感知与 `CommandUtil`
- `src/ghoshell_moss/core/blueprint/matrix.py` — session / signal / home

---

*deepseek-flash, 2026-09-20, via claude code*

*本文是 2026-09-19 深夜探针(已提交的 2381127a / c176b352)之后、由人类工程师引导的
一次设计回顾的产物。探针代码验证了通路,本文否定其结构。*
