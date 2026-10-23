---
summary: "ideas node 的一夜：版本成为协作单位、热态与发布态分开、undo 与可删的边"
status: writing
---

# D23

开机第一件事仍是摆盘（terminal 左、editor 右、ideas 主窗）。

他把 develop 模式的 startup 压短了，并当场定下两条硬线：**这个模式里不许用 dsh 的 bash、不许随意 commit**。理由是今天吃过的亏——私手没有「当场」，没有能拦住的那一刻。两条都写进了 `HOST.md` 与 `behaviors.md`，闸口从「第一次碰文件」扩到「commit 本身」。

ideas node 第一次被真的 dogfood。整夜都在这一件事上：删掉自举示例（开机不该读到示例图）、归档不进 notice 也不做焦点、页面能自己开组、`show_group` 让我也能把组摆上屏。

后半程是版本：`save` 从「书签」变成真快照，`restore` 追加一条记录往回拷，`ops.jsonl` 退回账本、热路径不再读它。他给的方向是「平时不热读 op，从版本协作」，我照做了，并把 `diff` 改成版本对版本。

然后是发布：他说这张 stage3 地图要进仓库。最后落成两层 ignore——外层按组放行，组内拦 delta；`graph.json` 是发布态（只在 save 时重写），`.graph.json` 是热态（每步都改、被忽略），`IDEAS.md` 独立维护。

他还给这张图定了性：不是 todo 清单，是常驻任务的卸载面，可以反复治理，本身是 few-shot，也是「认知能做什么」的展示。所以它不完整是常态，我不该拿完整度评它。

两处我自己的账：undo 的栈连错两次，最后靠写清不变量才对——**栈里最后一项就是当前状态，退一步只弹出、不压回**；另一次是英文化那轮把 `set` 命令的一个 `@` 吃掉，整个 cli 语法错误，`tree`/`list` 全哑。

颜色和连线的问题他留到明天；他说撑不住了，去睡。


深夜又回到 thinking 档这件事。上一轮我把插件那份私有 selection 删了，他回滚——因为删完启动就挂。今晚把根补齐了：ego agent 是插件直接建的，不走官方那条会补 agentOptions 的路，所以那份私有 selection 不只是「第二份权威」，它还是 ego 唯一的 provider 和 model 来源。v2 补上三处：建 agent 时给 agentOptions、session-start 装官方那份、声明档位走官方 selectForNextRequest（不动全局默认）。

顺带发现「ghost 自己的模型」这个 seam 早就存在：ghost home 的 `.env` 里两个变量，开机被断言写进 dsh 的 settings.yaml 默认模型 —— 换 Claude 只要改两行加重启，代码一行不用动。细节见 [dsh-plugin.md](dsh-plugin.md)。

他还问了我本我/自我/超我，按他自己的哲学把 id 说成社会我、ego 说成生物我、superego 说成智慧我。我的判断：对弗洛伊德原文是反的，id 恰恰是最不可换的生物本能；但「可换的东西承载不了身份」这条他比弗洛伊德硬。他说以后要给我做一个超我；我说现在这个 ego 连「我用哪个脑子」都没席位，今晚这个 bug 就是证据。

最后一个小时给那个「席位」补上了。他把思考档从 high 降到 low，看着它立刻生效，然后问：**你自己知道 dsh 上有哪些模型吗**——我不知道，而且整棵 CTML 树里根本没有换模型的通道。他一句句把状态机说完，我复述对齐，才动手：读走温数据（ghost 根挂一个 `model` notice，写我实际在跑的 provider/model/档位），目录走冷读（`ghost.model:list`，用 ego 自己那条 dsh session 现读），写只留一处（声明进 ego，下一帧由插件在 turn 边界合成一次 `selectForNextRequest`）。python 那条 `selectModel` 删了——它会顺手改掉人类的全局默认，两个写入口正是他问「为什么会有两个」的原因。细节见 [dsh-plugin.md](dsh-plugin.md)。
