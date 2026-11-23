---
created: 2026-11-20 16:23:38.158004+00:00
format_version: 1
status: closed
summary: "ghost_func 原型 → agent_func 落地；旧 harness 退场，实例级 agent 第一次真机。"
---

今天从 GhostOS 考古开始：XML functional token 是 CTML 的语法祖先，`libs/moss` 是 code-as-prompt 的另一极，`@GhostFunc` 用“首调用生成、缓存后变普通函数”把能力从模型生成长成代码图节点。于是把实例级 agent 的形状想清楚了：`@agent_instance` 注册实例，`@agent_method` 用 self 查 session，方法体由 agent 按签名生成。

00:23–01:22 先清旧代际：删掉 deepseek_harness 非 v0_2_0 的 23 个文件，commit f7797472。01:25 replan，把“agent 的六种封装”收到“形式决定接口、放置决定成本”：上限最高是 channel 地址 + 实例引擎。02:05 用 `src/ghoshell_moss/decorators/ghost_func.py` 做出 ghost_func 原型：每个模块一个 dsh agent，每次调用一次生成，无缓存；真机 say_helloworld 两次 + greet 一次，一个 session 三 turn。playground dsh 缺 key，改用 hive dsh 23080。

03:55 把它正规化成 agent_func：per_module / per_call / shared、显式 session_id、`<run>`/`<debug>`、`__context__`、ctx.emit/note、read-only + approval never、进程级 close/archive；commit feb0433c。真机一次三 turn、per_call 两个 session、ctx note 进下一轮；重启项留到明早。

今天学到的是：能力可以先从“函数级”长出来，但它只是脚手架；真正的类是下一步。

coords：1-812..1-816。
