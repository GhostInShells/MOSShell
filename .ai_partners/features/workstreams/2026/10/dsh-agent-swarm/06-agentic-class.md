---
name: 06-agentic-class
description: Agentic class — class 是 op 表、instance 是身体、method 是阻塞 op；从 agent_func 到类级 surface 的技术决策
---

# 06 — Agentic Class

> 本篇记技术决策：为什么做、解决了什么、边界在哪；不记实现方案。实现细节在
> `src/ghoshell_moss/agents/blueprint/agentic_class.py`、
> `src/ghoshell_moss/agents/blueprint/_agentic_class_helpers.py`、
> `src/ghoshell_moss/ghosts/dolores/agentic_class_driver.py`。
>
> 位置：dsh-agent-swarm 的底 primitives，不是 swarm 本身。它定义的是 swarm 里
> "一个可编程 agent" 的形状：class 是 surface，instance 是 body，session 是记忆。

## 问题：agent_func 没解决什么

agent_func 把 op 绑在函数上：一个函数一个 agent，session 按 module/call 生成。
它证明了"一个函数可以由 agent 实现"，但没有解决类级的问题：

- **身份**：函数没有 instance；同一实例的多个 op 无法共享身体与记忆。
- **组合**：没有"A 持有 B"的形状；A 的 op 无法调 B 的 op。
- **边界**：没有 class 级 context、workspace/writable/readonly、取消链。
- **调试**：没有不开 session 就能看接口与提示词的 dry-run。
- **返回**：`__run__` 被当成字符串；实际它应该遵循 method 的返回协议。
- **重入**：同实例 reentry 与跨实例嵌套没有清楚的分界。

## 决定

**KD1 — class 是 surface，instance 是 body，method 是 op。**
- class 源码就是 op 表；`@agent_method` 的签名与 docstring 是 op 契约。
- 一个 instance 一个 session；同一 instance 的多个 op 共用记忆与状态。
- 当前是哪个 op 由调用信封说明；method 本身不逐次重渲染。
- 这解决了 agent_func 的"函数级身份"问题，也让"一个类 = 一个领域 surface"成立。

**KD2 — 不 proxy；生成 module 级函数，编译进定义模块的 facsimile。**
- agent 返回 `async def __run__(self, ...)`（或 `__debug__`）；driver 把它编译进
  定义模块的仿制体，用活 instance 调；module imports 与 `self` 字段都可用。
- 不走 proxy、不走 CTML channel；单个 op 直接在 bash sandbox 里 `python -c` 跑。
- 这让生成代码保持普通 Python 语义，也让组合、单测、grep 都成立。

**KD3 — session 按 instance，进程级；不共享，归档后不恢复。**
- `session_id` 显式设置时也只属于一个 instance；重复绑定抛
  `AgentSessionConflictError`。
- `archive_after_close=True` 是默认；`close()`/`atexit` 按 session id archive；
  归档后不恢复。
- 实例 GC 不拥有 session；默认生命周期就是进程结束。

**KD4 — driver 按 event loop；method 阻塞；同实例 reentry 拒绝，跨实例组合允许。**
- `for_loop()` 拿当前 loop 的 driver；`prepare` 带锁；loop owner 负责 `close()`。
- 同实例再调 `@agent_method` 是 reentry，直接拒绝；需要另一个 op 的逻辑，就写进
  同一个 `__run__`，不要绕回去。
- 不同实例可以组合：A 持有 B，A 的生成体 `await self.b.op(...)`；真机里
  `Composer` 调 `Probe.step` 三次，B 走自己的 session，A 返回自己的结论。

**KD5 — answer 协议只有 `<run>` / `<debug>`；工具边界在 preset，不在 blueprint。**
- 一轮 answer 必须有且只有一个 `<run>` 或 `<debug>`；否则
  `AgentMethodProtocolError`。
- agent 可以用工具，但答案不是工具调用；blueprint 不设 "No tools"，也不设
  per-method 工具白名单。
- 这解决了之前把"工具边界"错写进 op docstring 的问题：工具是 preset 的事，
  blueprint 只管答案协议。

**KD6 — cancel 要上升到 driver，取消 session 的在途 turn。**
- `CancelledError` 从 method 冒到 driver；driver `cancel` 取消 session 的在途
  turn；session 本身可继续使用。
- wrapper 只在真正进入 `ctx.lock` 后 cancel；在锁外等待被取消，不会误杀另一个
  call 的 session。
- `_ensure_session` 在取消时清理半建的 raw session。

**KD7 — workspace / writable / readonly 是写边界，不是工具边界。**
- `workspace` 是入口；`None` → `Path.cwd()`；必须在 project home 内。
- `writable` 是写根；`None` → workspace；必须在 workspace 内。
- `readonly` 是声明式只读；driver 用 policy 落实；子目录级 writable 的精确强制
  留给 driver。
- 类级 `AgentConfig` 是默认；实例首次调用前用 `AgentCtx.configure(...)` 覆盖；
  `AgentConfig` 每实例 deep copy，不串。

**KD8 — 反射是显式选择，不是自动猜。**
- `reflect_source` / `reflect_interface` 只列 imported 依赖；class 自身源码总是打印。
- pydantic/dataclass 走 source：字段定义就是 API，模型天然知道怎么读；实测
  `ExplorerQuestion` source 657 chars vs interface 552 chars，interface 只有
  `__init__(**data)`，丢掉了字段。
- plain class / module / function 走 interface：签名 + docstring 就够；函数已验
  `expand_question`。
- renderer 不做 pydantic 特判；选择规则交给 decorator 数组。

**KD9 — debug 是 RSI 的 dry-run。**
- `AgentCtx.debug_interface()` 出 op 表 + 反射块；`debug_prompt(op, ...)` 出首条
  instruction + call envelope；都不建 session。
- 公共实现收在 `_agentic_class_helpers.py`；blueprint 只惰性 import，不背 codex。
- 迭代变成：改 class/decorator → `python -c` 看 interface/prompt → 再跑 agent。

**KD10 — 返回值原样透传；不做通用强类型校验。**
- `__run__` 的返回就是 method 的返回注解，不是 str；`__debug__` 才是 str。
- 需要额外校验，就在 plain method 里包一层：`foo` 调 `_foo`，validate 后返回。
- blueprint 不替 Python 做通用类型系统。

**KD11 — 重试与副作用：给模型一个 abort 信号。**
- 协议/执行错误重试到 `max_attempts`；已知不可逆副作用，
  `raise AgentMethodAbortError`，driver 不重试。
- docker/worktree 级边界暂时不做；`metadata` 是 driver-specific 逃生仓，默认空，
  容易腐烂，不定义键。

**KD12 — 可观测性靠手动持有，不建 children 链。**
- 先建 A，再建 B 并让 B 持有 A；调用方同时持有双方，两个 ctx 都能查。
- 不在 `CallRecord` 里链 B 的 calls；B 的 trace 在 B 的 session。

## 真机证据

- `DOLORES/agentic_demo.py` 的 `Secret`：第一个 class 级 op 真机通过。
- `Counter`：`add`/`get` 两个 op 共用一个 session，状态延续；`isinstance` 对原类与
  子类都成立。
- `MossExplorer`：`__context__` 读项目根 `GROUND.md`，`quick_answer` 不用工具、
  `explore` 用 read-only 工具跑 `moss features list`；plain `describe` 调
  `quick_answer` 通过。
- `Composer`/`Probe`：A 持有 B，A 的生成体调 B 的 `step` 三次，B 走自己的
  session，A 返回自己的结论；A calls=1，B calls=3。
- `Slow`：`stall` 跑 `sleep 45`，8s 取消，record `cancelled`；`ping` 随后
  `pong`，session 可复用。
- `BlueprintReader`：把 blueprint 自身作为 `reflect_source` 起一个 agent 自评，
  结论是"能，一眼能看懂"。
- 反射对照：`PydanticSourceDemo` / `PydanticInterfaceDemo`；debug 无 session。

## 未决

- `ctx.active` 由 driver 置位，blueprint 只读；契约里还没写明谁在何时置位。
- `reflect_source` 的示例用了整个 blueprint module，和"小源码"的措辞容易混淆。
- `set_session_id()` 与 `configure()` 的分工、`AgentSessionConflictError` 的校验
  时机，文档还没讲透。
- 重试幂等、timeout 是否算一次 attempt、`cancel()` 的完整语义。
- `debug_result_cap` 溢出文件的落点/命名/清理。
- `agent_func` / `ghost_func` 已删除；class driver 自带 runtime 与环境声明
  （`runtime/cache/agentic_class/dolores.json`，含 dsh pid 探针）。

## 非目标

- 不是通用 agent swarm；不是 dsh-agent-swarm 原型本身。
- 不是 per-op CTML channel；不是 node/channel 级机制。
- 不是容器/sandbox 边界；那是未来 channel 拉容器时的事。
