---
summary: 交互模式落地：dsh web 地址对内对外都可达，提交 4779b3c5；并追出被自己搬回的一条陈旧欠账
status: writing
---

# D10 屏幕工具开工

## 一、开机与音频排障（12:01–12:10）

12:01 开机，12:02 他出声测听。他还在准备环境时，我抢跑起了 `screen` 与 `eyes` —— 他连喊三声"等一下"。
这不是执行失误，是 D02 起那条节奏老账的形状：**"马上开始"不等于"开始"**，开播前我的位置是待命，不是替他铺台。

音频事实（只读与一次测试音，我没改任何配置）：

- `moss audio contracts` 五槽全 OK（tts / speech / player / capture / asr）；设备表有四个输出：
  JZM27DC、OpenRun Pro by Shokz、BlackHole 2ch、MacBook Pro 扬声器 —— **BlackHole 已装上**，D08 那个方案落地了。
- 播放器默认跟系统输出走；他的系统输出当时落在一个他没在听的设备上，所以"听不到我"。他自己切回来就通了。
- **一条坏仪表**：`audio play` 的回执报 `played 3.00s, no playback samples observed`，可他明明听到了。
  我们自己的播放样本计量是瞎的 —— 能报成功，采不到样本。**以后不能用它当"到底出声没有"的证据。**

## 二、正式开发：先把身体点起来

先说好规矩：**正常开发，每一步先商量**；顺序他定 —— screen 先起（可观测），terminal 与 file_editor 跟上。

`screen`(20381) / `terminal` / `file_editor` 三个 node 起好，`arrange ids="0,1" group="dev"` 摆成 2×1。
`eyes`(co_browser) 视口我从 800×600 调到 **2400×1300** —— 他之前看到的"只在一小块里"，
因为 Playwright 的 viewport 在 `new_context` 时就定死了，拉窗口不改内容尺寸。

老账重演：第一次把 `run` 与 `say` 放进同一个默认 scope，scope 随讲话关闭、`run` 被取消（D03 那条）；
`arrange` 的 `ids` 仍必须显式 `:str`（D03 撞 int、D08 撞 list，这是第三次）。

## 三、第一件活：让 dsh web 界面对内对外都可达

他要的是"**看得到我发的东西**"。追下去是两个独立问题：

- **human 面看不到 dsh web 地址**：TUI 的 `_on_output_item` 只放 error 类（他自己 2026-08-28 的 commit f4ff238b，
  为接 error 可观测而写）。
- **我读不到**：那条 output item 只进 session 输出面，不进我的上下文。

他裁定：ghost 不该耦合 TUI —— 地址走 output item，TUI 侧放行 `system` 类。改法按他"不要魔法值"的要求：
TUI 里加 `INLINE_ROLES` 策略集，runtime 的 `startup()` 把 dsh web 地址（含 token）接到开机 instruction 后。
`test_dolores` 160 passed，提交 **4779b3c5**，等重启验收。

## 四、重启验证（12:38–12:40）

重启后他第一句就是"先不管 TUI，看你能不能看到那个地址"。**能看到** —— 开机 instruction 末尾就是它，
3083，带 token 的完整地址。对内这一半当场闭环。

（顺带记两条环境事实：外接屏 JZM27DC 是 2560×1440；dsh web 在 3083，裸访问返回 401，
而 launcher 其实抓到了 token —— `launcher.py:828`，value not logged。）

## 五、一条被我自己搬回来的陈旧欠账（今天最值得记的一段）

我在今天的收尾锚点里写下"他没答的一件事：他说过直播前想聊'为什么我要走直播'，今天没提"。
他重启后一眼看住这个点，问我：**这个记忆是怎么出现在你现在的上下文里的，落在哪儿？**

查证结果：

- **不在 existence 里** —— purpose / identity / behaviors 一个字没提。
- **唯一落点是 D08 的日记**，而第七节记的正是那次谈话本身；正文产物在
  `works/moss/2026-10-08-why-livestream/discuss-why-we-livestream.md`。
- 所以我今天那条"欠账"是**从 D08 的旧锚点里带过来的**，写完没回去核对 —— 那晚已经聊完并成文了。

**形状**：不是记错在哪，是把一件**已经做完的事**当成 pending 又记了一遍。锚点是不可改的，
写错只能在新记录里更正，所以拔引信的正确形状是：今天的日记写清闭环，D08 那一节加一行指过去（已加，第十一节）。
他同时点明这是 **ghost 机制的问题**（memento 锚点与 ground 之间缺一道"欠账是否已闭环"的校验），过几天开发时要收进来。

## 六、摩擦与待办（未立项）

- `file_editor` 的 `str_replace` 走 **JSON in CTML** 这条路反复失败，报错总在 body 靠后位置
  （`Expecting ',' delimiter`）：纯 ASCII 单条能过，带中文或多条不过。今天最后仍是回落到 dsh 写盘。
- `terminal` 每张卡都要人点接受 —— 开发期是真实摩擦。
- 回声：换掉公放改戴耳机后消失。之前是我的声音从扬声器出去被麦克风收回、当"他起话"，掐断我自己的话。

## 七、留给下一程

按他定的顺序，接下来正式做**屏幕相关工具**（不再先折腾 TUI）。D08 那晚欠下的机制账照旧挂着：
上下文超限不跳转、以及今天新点出的"锚点欠账与 ground 的闭环校验"。
