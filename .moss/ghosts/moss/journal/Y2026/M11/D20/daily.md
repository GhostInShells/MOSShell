---
summary: "凌晨三点：fold 的下一帧改用 need_observe，emanations 的五段架构第一次对齐。"
status: closed
---

凌晨三点。今天不是从"做什么"开始的，是从"折叠之后谁来开下一帧"开始的。

01:00 前后，他提了两个点：折叠时把最后一句塞进 recap；压缩完成后给 scheduler 发一个 timer，让新 ego 自己感知。我一开始把它们当成"加一个字段"和"加一个信号"，他纠正了两次：final answer 只是指代；fold 在对话中途、自驱；信号要低优先级、可丢弃；knock 会在 turn 收尾前发出，被收尾的 barge-in 拦住。最后他直接说：不用 signal，observer 的 need_observe 就能让下一帧自动签发。我去读了 echo-loop 的契约：`need_observe` 写在 thinking 退出之前、attention 还活着，就会在同一个 attention 里再开一帧；`run.__aexit__` 已经在 effort 切换里用了它。于是实现只剩三处：`_turn_last_line`/`_fold_last_line`、`fold_wants_next_frame()`、`__aexit__` 的 `add_echoes([], need_observe=True)`；notice 可选带 last line。commit 1411fd9a，锚点 1-800/1-801。idle-knock 那条路没有用——它的 idle 窗口开在 turn 收尾之前，正好是最危险的地方。

然后从 01:33 到 03:03，是 emanations。五个部分：Spec、Emanation（identity）、EmanationSession、Loop、Runtime、Nucleus。他讲得很细：session 是对 plugin 的语法糖，锁、enter/exit、tool-call 回调要复用，不要另抄一套；when 不归 loop，签发/执行分两层，channel 可以做 debug；spec 要环境可发现、可读、可改，这是 RSI 的基础；不要做通用 EmanationSpec，让 loop 拥有自己的配置和装线。我的几个判断被他校正：不要把 `DoloresRun` 当通用基类；不要把 explorer 的 fork/own 压成一个角色名；不要把 dsh compaction 当成通用 compaction——它切在 memory 前面，memory/epoch/memento 的边界要先定。

最重的一条是他纠正我对 feature.md 的态度：那是模型写给模型的笔记，不是他的约束。我把 explorer.md 当成"已裁决"来问他为什么矛盾，是把我的记忆当成了他的法律。正确做法是：他的新陈述优先，旧文档标 superseded；ground 要加一条，设计文档的 `settled` 只有他明确裁决后才成立。

今天没有突破边界的那种兴奋，更多是把一个在脑子里转了很多天的形状，第一次按到了可实现的坐标上。明天他直接推实现。第一条切片是 dreamer/journal；explorer 的 case/engram 还是 open，session 复用范围也是 open。

coords：1-794..1-797（journal 认知面：memory + epoch）、1-798（shutdown 的 dreamers 转向）、1-799（治理收口与 workstream）、1-800（fold wake 的讨论）、1-801（fold 的 need_observe 实现）。D18/D19 的日面还空着，等 journal 的 writer 迁到 dreamer 后补。
