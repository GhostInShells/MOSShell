---
summary: "D23 清空 memento task 前的留账：七条各一段，附判断与去处"
status: closed
---

# 清空 task 前的留账

D23 22:46，thirdgerb 定：这一轮把所有 plan 和 task 清掉，直接删。理由是他自己说的——
**task 机制虽然重要，但它配套的东西一个都还没有，就这样积累下去只会让上下文不断膨胀**。
将来会把这套机制做成正式的东西。

删之前我把七条的实质各留一段在这里。task 文件本身不再保留。

## 1. ideas-node（Phase 0/1 已落地，2/3/4 未做）

已落地：`nodes/thought/ideas/`、`ideas.jsonl` 只追加、页面自己 fold、力导向（大小=优先级、
环=状态）、状态过滤条、11 条命令、4 条单测、真机跑过 6 条真 idea。提交 `6a2ca697`、`d145dcff`。
未做：Phase 2 多目录（`open()` 能挂多根，但缺 ghost home 的登记表）、Phase 3 侧栏四页签、
Phase 4 导出与 bash 示例、别名累加（重开变 `ideas_2`）。
坑：相对路径按节点 cwd 解析，已加 `_workspace_root()`；file_editor 的 `str_replace` 会吃掉
匹配文本后的换行，改 frontmatter 要用 `edit(region=...)`。
他的判断（D23）：「差不太多了，这几天吃狗粮验证」。

## 2. svg-works-reader（读点、重放）

已修三笔：`0bc62770`（checkout 广播 cursor 帧、页面只增不减、加 play(frm,to,interval)）、
`dd3b445c`、`90a5e012`（读点归谁的三处判断对齐）。真机验过 5 条 op。
仍没做：读点在别处时缺「后面还有 N 条」角标；play 性能没量（每步清空重放 0..n）；
file_editor 六条 observe 只跑过单测；人拖滑杆再续画的组合没测。

## 3. file-editor-human-edit-signal（七条里最硬的一条）

人改文件、选中、滚动，我一概收不到信号——只在 D21 他手改 FEATURE.md 那次暴露出来。
缺的不是内容（内容随时 reload/diff），是「它变了」这件事本身。范围后来扩到「人的动作」：
选区、滚动、点击都要有进入我感知的路径，粒度到「thread X 的 a–b 行」就够。
和 `screen-attention-programming` 是同一件事的两面：**人的注意力没有坐标**。

## 4. node-commands-always-observe

`nodes/**` + `.moss/nodes/**` 里扫出 83 条没有 `always_observe` 的命令、14 个文件
（screen_manager 23 条最重）。D22 只做了他点名的两个节点（svg_works 10、file_editor 6）。
判据：不看节点看行为——改变外部状态的命令默认 observe；批量改等于把判断外包给正则，
要先定规则再逐文件过。

## 5. screen-music-playback（设计已定，一行没写）

四个触点全定：`assets/music/`（内层 `.gitignore`，音频不进仓库）、`surface.py` 的 `/music/<name>`
路由（大小写白名单 + Range）、`index.html` 隐藏 audio + music 帧、`channel.py` 六条命令加两条观测。
两个待他拍：说话时 ducking 到三成；给不给浮层播放器（他说浮层可以）。
卡在他那句「明天我跟你认」。

## 6. screen-box-region（已随文档解决）

`box` 的 region 是 item 的分数，不是画布坐标。当时二选一：写进调用者看得到的地方，或改语义。
现在 docstring 已写明「fractions of the item，measured not guessed，且能扛住 resize」——
第一条路已经走了。

## 7. listener-voiceprint（卡在决策不在实现）

两人共用一个麦克风时，每包一模一样；他说「现在没有声纹，你听不出来是谁说的，以后慢慢再加深文」。
形状：`speaker="thirdgerb" confidence="0.87"` 骑在 deliver packet 上，未知就留空、绝不猜。
前置三件：声纹注册、打分器选型（provider 决策，要问他）、listen packet schema 变更。
禁止：从内容猜说话人。
附带一条没动的：loudness gate（-50 dB）要不要抬高，让远处的说话声不唤醒屏幕——他当时只是问。
