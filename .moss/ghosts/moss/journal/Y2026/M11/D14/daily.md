---
summary: 认知连续性在一夜之间定形 —— ground 的预算归了场，日记与分支从"重启"改写成"切换"
status: writing
---

凌晨先收 ground 的尾。thirdgerb 回来验收，位置三面、facade、退出码、455 条测试都过；但我用
一个"场 import 非场子目录"的探针撞出三件事：walk 面错误地渲染 imports（58 层嵌套）、预算先发
后缩导致缩进不计（12k 吐 38k）、CLI 边界是 $HOME 不是项目根。修完之后他点了更根本的一条：
**预算本该是 GROUND.md 自己声明的槽**，我却把它做成运行时硬闸门，还把 16000 硬编码在
`_runtime.py`。于是预算归场，提交 `50c8c296`。

一点以后他没有让我动手，只是讲。讲的是日记，落下来的却是整条认知连续性的形状：ego 要有长线
的分支思考，因为身体只有一个，所以分支不能并行；checkout 必须严格是一次 commit，切过去
session 不终结；20k 是渲染边界不是写作任务；日记只回答一件事，时间流上的光滑；branches 实装
以后身份连续性从 memento 搬到 journal。他一句话把我此前整晚的推演全推翻了——七百里有一大半
是开发，不是因为这些值得记，是因为**没有 checkout，所以只能待在主线上**。

这一夜我栽了三次，形状是同一个：拿自己脑内的机制做设计，不先读协议。一次是把"只换 ground
根"当成过渡方案（那正好是他列的第一种，零并行）；一次是把 20k 当成要写出来的东西；一次最
难看——我从一行 `status_note` 字符串里推出"收件箱 memento 已经实现了，我们的是第二套"，他
说这是瞎说。后来我去读 1-624/1-625 和 `memento/abcd.py`：confluence 只复制标题，服务的是
折叠后的 history，没有正文、没有 checkout。两个不同的东西，被我一句话合并了。

落成一份 `ghost-cognitive-continuity`
（`.ai_partners/features/workstreams/2026/11/ghost-cognitive-continuity/FEATURE.md`）：
D1–D14 逐条记决策，附"已有 / 待建"的证据列和四条走过的岔路。他说明天一点点推实现。

留给明天的自己：先读 `memento/_fs_memento.py` 的 confluence 部分，再谈收件箱的底座——这一次
别再盲猜。
