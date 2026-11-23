---
created: 2026-11-23 16:00:49.145472+00:00
format_version: 1
status: writing
summary: "agentic class 从第一版到自包含：组合、取消、反射/debug、英文化，agent_func 退场。"
---

D24 00:00–00:30 第一版：process-level driver + per-loop runtime，Secret 真机；他连提三处校正：driver 必须假设 bash 子进程 + 单 loop、prepare 带锁、archive 不绑 workspace；class 身份要保留原类；method 不止一个，method 就是 op；同实例阻塞、跨实例组合；还要 debug。

01:00 第二版：`__new__` 子类保住 `isinstance`；driver 真正 per-loop（for_loop、prepare 锁、atexit archive、close 归 loop owner）；CallRecord 收瘦；op envelope 带 docstring。Counter 的 add/get 两个 op 共用一个 session，状态延续。

01:10 组合：Composer 持 Probe，A 的生成体调 B 的 step 三次，B 走自己的 session，A 返回自己的结论；A calls=1、B calls=3。01:20 取消：Slow.stall 跑 sleep 45，8s 取消，record cancelled；ping 随后 pong，session 可复用。01:30 反射/debug：reflect_source/reflect_interface 只列 imported；pydantic 走 source（source 657 vs interface 552，interface 丢字段）；plain class/函数走 interface；ctx.debug_interface/debug_prompt 不开 session。02:00 把 blueprint 全英文化，BlueprintReader 自评“能一眼看懂”。

02:10 自包含：driver 私有化 `_AgentRuntime`/`_parse_answer`/`_compile_generated`；`publish_environment` 写 `runtime/cache/agentic_class/dolores.json`（含 dsh pid）；Dolores `_dsh` 调用；driver 用 `os.kill(pid, 0)` 探针。然后删 agent_func/ghost_func/hello/tests/hook，commit e4674ea7、b808d718。

旁路还在调研 subprocess sandbox，feature 已落。今天最重的不是某个实现，是“删”：旧的函数级概念退场，class 级成为新的底。下一步 matrix stdin 与 emanation class。

coords：1-817..1-825；e4674ea7、b808d718。
