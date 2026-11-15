---
summary: ego branch ground 从动机对齐到落地 —— 超长任务、桌子与 seed
status: writing
---

凌晨把 ego branch ground 的动机对齐了。他先说清 branches 在 ghost 里不是并行短任务的切分：
没有 UI 的 session 边界，模型靠 memento 的 commit 线索把 session 级连贯性找回来；一天里切换的
任务是不同的桌子，**桌上有什么**是进入上下文最快的方式（ground），桌上的工作记录是追踪
（notes）；而现实反馈由物理节奏决定，模型不能把一周压成一小时。所以 ego branches 管的是超长
任务的宏观 plan。`ego-branch-ground.md`

然后落地，两刀。memento 先长生命周期：`created` 事件（`_create_branch` 单点）、
`fork(commit_id=)`、重名闸、分页的 `list_branch_catalog`。ego 再长 ground：记忆容器变成
`<memento><branch><ground>`，分支的场在 `branches/{branch_id}/GROUND.md`，新分支创建时从
`.dolores/seeds/branch_ground/` 整棵复制，缺省则渲染 seed 正文并标 `default="true"`。
`_render_view` 转异步，同步孪生删掉。

中间被他纠正两处：`roster` 不是存储惯例（改 `list_branch_catalog`，连注释里的词一起清）；
fork 那类 bug 要修完（名字在存储层必须路径安全、已归档的线不能 fork、重名在创建处拒绝）。
提交 `612ad889` / `487d1b35`，领域测试 338 条全过。

他随后要求先写日记、再手动落锚，然后重启——新原语、新记忆形状、seed 目录都要 boot 才可见。
