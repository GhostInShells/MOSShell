---
summary: 
status: writing
---

# D09

## 00:10–00:17 camera 实测 + 一个"致命点"（会话卡住前紧急落盘）

push 已接 webview 并真机生效：起 screen(83001)+push(83003) 后，screen 的 notice 自己出现
`<views>#0=push</views>`，桌面 1 个浮动物 —— **没有人打开 URL**。

camera 实测：`request(source="camera")` 后 `#1 live [model] camera 'cam'`，流地址
`http://127.0.0.1:62423/stream/1`。**帧率按源分开（screen 10 / camera 30）那次改动，这次没有被 ffmpeg
拒**（改前 camera 崩的原话是"帧率 10 不受支持，只给 15/30"）—— 这是它的第一个真机证据。

**我找到的致命点**：webview 被自动认领后落在**桌面**，而桌面上的 item **没有 iframe，只有 canvas 画的
代理圆盘**（`nodes/screens/screen_manager/index.html:104-105`： "desktop 上的 item 没有 iframe，只有
canvas 画的代理圆盘；入组后才物化 iframe"）。点它只发一条 aside（`surface.py:_tap` —— "interest, not a
move... nothing on screen changes"），屏幕什么都不变；他连点 4 次 = 4 条 aside。我随后
`arrange(ids="0", group="push")`，notice 才变成 `#push(1)` / `screen #push [0] grid lr` —— 页面到这时
才真的物化成可点的 iframe。
推论：**"自动进入 screen"目前给出的是一张点不动的页面**；要让人类能点，必须有人把它 arrange 进 group，
而按 bridge 的设计（"Choosing where things go is the model's job"）那个人是**我** —— 也就是人类自己
打不开任何页面，push 的审批页于是依赖我活着、在场、没被打断。
另一条：push 的 instruction/NODE.md 我写的 "a screen shows it with nobody opening a URL" 是 over-claim，
实际是"screen 收下一个圆盘"，要改。

**重启的理由（硬证据）**：正在跑的引擎是我 amend **之前**启动的 —— `read_output` 的签名还是旧的
`address`，我传 `target=83003` 被直接拒（`invalid params ... read_output(address: 'str', ...)`）。所以
matrix 那一簇的新契约（pid 手柄 / target）**在重启前不在线**。

会话两次被中断（`turn/end aborted (user)`），camera 流仍 live；重启会清掉这些节点。

## 00:35–00:50 重启后：三轮"机制判断"，和一次误区的纠正

**重启完成**（本 epoch 是新引擎，matrix 新契约在线）。

**第一轮 · ghost home 的同步契约**：他问"version 闸门有没有问题"。读 `_runtime.py` 得三类：
`.dolores.yml` 读后改写（只动 `version`，`:585-603`）；认知场 seed-once（`:556-568`）；
`.dsh/profiles/web/*` 4 文件走 **VERSION 闸内整文件覆盖**（`:545-554`+`:612-617`）；
plugin `.ts` 与 preset **每次启动无条件覆盖**（`:326`/`:328`，docstring `:619-643`）。
我验了 plugin 那条不是文档说说：home 与 stub 的 plugin 同 sha、调用确在闸外 ——
所以"bump version 才更新 plugin"在代码里不成立，成立的是 profile 那 4 个文件。
闸本身没问题（`cordis.patch.yml` 是 dsh 指定的用户 patch 层，`patchReload: live` 会监视并热重组，
不能每次开机重铺）；**一处会咬人**：闸内是整文件覆盖，stub 里没有的字段会在 bump 时静默消失 ——
现存实例就是 `package.json` 的 `"patchReload": "live"`。共识：本次核心改动落在 ts、与 python 内核
同轮交付、每轮覆盖、不 bump、不打 patch。

**第二轮 · kind 到底有没有语义**：他问 `kind:'plugin'` 与 `kind:'user'` 对我是不是一样。
**对我一样**（`deriveEventMessage` 对 `user/message` 直接 `return event.data`，我读到的是 `content`；
`source` 只是出处；我们插件内部零处按 source 回查自己注入）。**但对 host 不一样** ——
一串按 `kind === 'user'` 判定的语义（会话标题、turn outline、goal 的 hasDirectHumanInput、
inbox placement/lastPromptAt、手势扫描、reminder 重置）。所以"换成 user 就看得见"不是免费的美化。

**第三轮 · 他给出关键约束**：`tool/result` 里带 moment 是**之前专门干掉过**的一条路，
根因是"让我看到的 moment 消息保持一致（不散在两处）+ 一个协议点位更好维护"。我先给了自己的观点：
**不迁就客户端渲染缺口去改本层协议**（单一交付点是对的；kind 不能乱换；把图塞回 tool result = 协议回退），
缺口在渲染面不在交付面。

**然后他把这个问题归零了，归得对**：这场讨论起于"以为我看不到 push node 的图"——**前提是错的**。
图一直在**模型面**（`user/message` 的 content block），只有**他**在 dsh web 上看到「未知内容块」。
两个面我一直混着说，所以才有这一场。结论：**不动 dsh、不动我们的协议**，人类面看图的正式通道
是**我们自己的屏**（screen_manager / push / webview）。这轮最贵的产出不是代码，是那条
`kind==='user'` 的 host 消费者清单 + "图片只在气泡与 tool card 两条路渲染"——下次十秒钟能判。

## 00:56 全链复验（重启后）+ index 参数实验

`screen`(pid 17745) + `push`(pid 17747) 起后：screen 的 notice 自己出现 `<views>#0=push</views>`、
桌面 1 个浮动物，**没人开 URL**；而且 push 的 **instruction 已是改后的措辞**（"you must arrange it
into a group before a human can click it"）——文案真的随重启上线了。
`arrange(ids="0", group="push")` → `#push ← [0] grid lr (1x1)`，他在那个窗口里点了接受。

- **camera**：`#1 live`，`capture()` 拿回 640×480、age=1.2s —— **我看到了他**（像素，不是推断）；
  camera 默认 30fps 被接受（改前是 `帧率 10 不受支持，只给 15/30` 直接崩）；push `stderr empty`
  （已不是 DEVNULL，是可读管道）。
- **index 参数（现在有两块屏）**：`index=2` → 帧 **1280×720（16:9）** = 外接 2560×1440
  = `Capture screen 1`；**省略 index** → 帧 **1280×800（16:10）** = 内建 2560×1600
  = `Capture screen 0`（stock 默认 1）。**双向闭合**：显式参数真穿到 ffmpeg，省略时回落本机默认。
- 一条别误判的事实：screen 源首帧约 6 秒，`capture` 期间的 `no frame (connecting)` 是**首帧延迟**，
  不是故障传导问题。
- 收尾：两条流 `stop`、两个 stream node 停（`exit=0`），`screen`/`push` 保持运行。

**给自己的一句**：这一轮我又犯了一次"无据时补指代"——把"审批页进 dsh web web-surface"当成
"这次改造"的落点，多造了一个他从没提过的命题。该问"指哪个文件"的时候，我选了看起来更完整的那条路。
