---
summary: dsh 0.2.0 接缝与 bundle 落地，chat off record 归位，moment role 与 CommitMessage 值对象落笔
status: writing
---

今天是回迁与正规化。

凌晨把 dsh node 的定位说清楚：它不是独立产品，是 agent-swarm 的前哨战；适配层不是"引擎动词适配"，而是"事实轴到引擎"。回迁调研的账：ghost 早就在跑 0.2.0-rc.2，旧 `deepseek_harness` 只是被补丁到 0.2.0 的旧壳，破坏主要在 import 路径。

下午到傍晚是接缝。Dolores 换到 `ghoshell_dsh.v0_2_0`（201 测试全绿）；插件改成正规 dsh bundle，三条硬事实推翻了计划里的一条：`node_modules` 不能放 `.ts`、`--dump-config` 只组合不 import、`_provision` 漏了 `DSH_HOME`；顺手挖出并修掉两个 provider seed 老 bug（provider 块必须写在 `config:` 下，环境声明只落名字不落值）。提交 `1382ba5d`。

然后是 chat off record。目标从"新写一份"纠正为"用 `ghoshell_dsh` 那一份"，拆出 service/bridge 两包，chat 归位；两次越界挨批：一次打印了凭据真值，一次读了活体进程的环境变量。规矩记下：只查 key 名与存在性，绝不 dump 值，也不读环境。

晚上是 moment 与 role：`Moment.inputs` / `percepts[role]`、`Signal/Impulse.role`、`buffer_impulse`，四个 nucleus 按"最新一条信号赢"透传。`memento:chat` 改成异步回执；第一次真机验证失败，挖出内核一个从未被测过的回调 bug——`create_asyncio_task` 的 done callback 写成 `async def`，`add_done_callback` 不 await，任务结束不出组、异常静默。修成同步回调（`b2f25559`），补五条单测。随后 chat 全链路真机通过，dolores 测试搬进 `tests/` 并按面拆成八个文件（`3b55dc9f`）。

深夜重审 memento 旁路 commit：token 增量、warn 走 named notice、commit 只在 turn/end 生效、exit 记录、meta 资产、mindset。九条逐条对齐，设计意图落进 driver 的 FEATURE。最后定下 `CommitMessage` 值对象：message 是唯一权威，`op(target): content` + body，读宽容写严格，case agent 共用语法（`a52c9262`）。

还没结束。明天从这里接：case agent 的写侧归一化与读侧迁移；以及九条里尚未对齐的那些。
