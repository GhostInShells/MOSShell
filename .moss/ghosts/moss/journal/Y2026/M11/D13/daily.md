---
summary: ground 重做 —— 把渲染从 channel 层收回 GROUND.md 自己身上，删掉整套 open/close 生命周期
status: writing
---

今天做的是同一件事的两个面：把 ground 从"channel 补协议缺的洞"收回成协议自己的机制。

起点是我昨天写的那个"一 set 一 channel + gate"的版本。thirdgerb 让我回答"你现在看到的
ground 是什么形状"，我答成了"两套账"，他纠正：子场本来就有自动展开和主动打开的两种，
project 没有子场是设计动机。然后他说重话 —— 你从来没读过 SPEC。

我读了。472 行，三层（Ground / GroundSet / Pin），两个观察面（field-root 与 walk），
三种锚点，内容不向上合并，@ 单层。读完才看清腐烂的形状：**每次迭代都在 channel 层补协议
缺的洞，协议始终保持"一个目录一张面"，于是 channel 长出树、门、行政面去模拟多目录多面。**
walk 面只对人类开放，ghost 想用得起子进程；`groundset` 被写进场的 frontmatter，而
`contract.py` 自己写着"Ground 自身不消费"。

下午他给了 11 条重做方向，我照做，但中间反复跑偏：先提了个"全树自动铺开"的 XML 方案，
被他一句话打回 —— **子场不是自动渲染，是父场点名哪几个在本场渲染**。这条我聊了五六轮才真
听进去。

改的过程是四刀：改名（groundset→inline，law→content）、位置渲染器（位置决定脸，
XML 容器，预算截断，失败即结果）、删生命周期（DefaultGround 整个类没了，GroundSet 从
385 行掉到 63 行）、channel 砍到一条 render（646 → 93 行）。

他也让我用 dsh 开了一个只有 SPEC 的 agent 来读。它挑出 16 个不确定点，其中六条是真缺陷：
`always_show` 的字段表前后矛盾、位置性 pin 从未被定义、嵌套子场的 `$CWD` 没写、墙钟预算
自相矛盾、inline 环没有终止、块顺序与示例不一致。**我写了一份自以为清楚的规范，一个只读它
的模型读出了 16 个洞。**

还有两个自己的错要记：一次用正则盲改 19 个文件，把测试正文里的 "law text" 也改了；一次
`rm -rf` 删到了父目录，把 journal 的三个已跟踪文件删了（git checkout 救回）。两次都是
"在碰巧站的地方验证，就以为验证过了"。

现在 ground 的样子：一个位置进去，一张自解释的 XML 脸出来；子场只渲染点名的；编辑归文件
工具；读是冷的。445 个测试绿。
