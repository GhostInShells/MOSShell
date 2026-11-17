---
summary: emanation task —— 三处部署期错误、405 根因与 turn-only ref
status: writing
---

# Emanation task —— 首跑与自伤

## 它是什么

一次 off-record 的结构化自问：对某个 commit 的 session 发一次 `chat_off_record`，
请求容器 `<emanation-task-request id="et-N">` 带 prompt 与 response schema；模型必须以
`moss_emanation_task_response(id, resp)` 工具调用回答；`resp` 过 pydantic 校验，失败重问。
工具在插件里声明，主路 execute 是常量 `{ok:false}`，只为让工具表在会话轮与旁路轮完全一致
（前缀缓存）。实现：`src/ghoshell_moss/ghosts/dolores/mindset/emanation_task.py`；
调试面：`ghost.beside.debug:emanation_task`（`mindset/beside/debug.py`）。

## 为什么会把 ego 弄死

1. **插件 schema**：`resp: {type:'object'}` 缺 `additionalProperties`。dsh-tools 的 author
   校验（`node_modules/@deepseek-ai/dsh-tools/lib/index.js:699`）要求 object 显式布尔；抛出
   发生在插件模块求值期 → 整个 bundle 不挂 → 所有路由缺失 → `ego/create` 405。修：
   `additionalProperties: true`（`dsh_plugin/index.ts:687`）。`resp` 是插件里唯一的 object 参数。
2. **契约声明位置**：`EmanationTasks` 原来声明在 `DoloresMeta.contracts()`。宿主
   `host/ghost_runtime.py:200` 在 `ghost.__aenter__` 之前 force_fetch meta contracts，而
   `EmanationTasksProvider` 要 `DshConnection` —— 它到 `_dsh()` 才发布。修：声明移到
   `Dolores.runtime_contracts()`，在 `__aenter__` 末尾校验（`_runtime.py:649/795`）。
3. **配置类型**：`EmanationTaskConfig` 是裸 `BaseModel`，没有 `conf_name`，默认值无处落盘。
   修：继承 `ConfigType`，物化到 `.dolores/configs/emanation_task_config.yml`。

三处单独都可能侥幸过去，叠在一起就是下一次启动黑屏。

## Review 发现（commit fbbc7f87）

- `_meta.py` 的 `Contracts` import 变成 unused。
- `DshLauncher` 发布了但无人 fetch；`EmanationTasksProvider` 仍 fetch `DshConnection`。
- `runtime_contracts().validate()` 只查 bound；宿主原先 step 3.5 的 force_fetch 循环对
  Dolores 失效了 —— boot 期实例化保证没了。要恢复：同处对每个 contract `force_fetch`。
- `__init__` 签名重排是无关 whitespace。

## 真机验收撞到的第二个 bug

`ghost.beside.debug:emanation_task(coord="1-767")` →
`UNKNOWN_ERROR: ref has no end_seq: a turn-only ref cannot become an anchor without guessing`。

- `_ego_memento.py:425` 写 ref 只记 turn；实测 1-767 的 `start_seq`/`end_seq` 都是 null。
- `DshSessionRef.to_anchor`（`packages/ghoshell-dsh/src/ghoshell_dsh/blueprint/models/refs.py:97`）
  故意拒绝 turn-only：`seq=None` 在 anchor 里意为"会话起点"，那是另一个意思。
- 生产 note 路径 `_ego_memento._note_call`（`_ego_memento.py:587`）直接构造
  `DshSessionAnchor(session_id, seq=end_seq, turn=end_turn)` —— turn 回退就是为这种 ref 设计的。
- `debug.py:131` 用错了 API。修：`DshSessionAnchor.from_ref(ref, at="end")`。

## 验收结果（重启后真机，16:56–16:57）

七项全过：

1. 工具进 ego 工具表 —— `moss_emanation_task_response` 出现在本轮的 function 列表里。
2. `ghost.beside.debug` 存在且唯一 —— facade 下只有一个 child。
3. coord → ref → anchor —— `1-769`（turns 4–5，`start_seq`/`end_seq` 全 null）通过；
   turn-only 修复是真的走通了。
4. 前缀继承 —— `et-1 attempt=0 in=2679 cache_read=136448 out=2215`；重试那次
   138,624 → 138,880，工具表稳定的设计意图兑现，不是纸面推论。
5. 回答形状 —— 一次 tool call，`resp` 过校验，返回 JSON（score 6.0 / 8 条 issues）。
6. 负路径 —— `json:JSONEncoder` → `error: TypeError: ...`，同批的下一条命令照跑。
7. 重试与超时 —— 用「不要调用任何工具」的 prompt，两次都答散文 →
   `EmanationTaskError` 带最后那条 reason；`timeout=0.01` →
   `DshTransportException: ReadTimeout`（超时在 dsh 传输层，不是 asyncio）。

探针 `/tmp/moss_emanation_probe/ground_friction.py:GroundFriction`，验完已删。

## 异步化（commit cddd771c）

thirdgerb 判它不该是阻塞命令：debug 的代码形状会被真机制继承，所以表面必须是回执 + 投递。
**不是 `blocking=False`** —— 那只放开 channel 队列，任务仍绑在 scope 上、scope 一关就被
cancel；这条记成了 kernel issue #141，今天不解决。改成 `memento.chat` 那套：命令立刻回执，
工作放进 `CommandUtil.create_signal_task`，结果以 `mind/delivery` 信号回来；失败是内容，
不是没人读的异常；模型引用错了仍在回执里拒绝，不启动任务。

四条新测试（`test_beside_debug.py`）：回执 + 投递、失败作为内容、坏模型拒绝、坏坐标拒绝。
336 passed。

**运行中的进程仍吃旧模块**，这条表面的真机复验要等下一次重启。

## 异步面真机复验（重启后，19:16）

重启后第一次跑，`coord="1-770"`（上一条 commit，跨进程的 session）：

- 回执 19:16:31，呼叫同一秒返回——命令真的没占 channel。
- 答案 19:16:44 作为 `mind/delivery` 到达，任务耗时 12.5 秒；日志
  `[emanation] task at 1-770 ... finished: ok in 12.5s`。
- 缓存：`in=131270 / cache_read=11136`。跨进程目标的 session 前缀是冷的，
  跟同会话那次 `cache_read=136448` 对照——"旁路继承前缀"只在同一会话里兑现，
  跨进程重放继承不到。不是缺陷，是边界。
- 这一轮期间 dsh 有一次模型调用卡了 10 分钟（thirdgerb 判断是官方 API 侧），
  与本机制无关，记在这里免得日后误读成回执路径的问题。
