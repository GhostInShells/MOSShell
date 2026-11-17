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

## 验收状态（1-766 的计划）

1. 插件重装后工具进 ego 工具表 —— 通过。
2. `ghost.beside.debug` 存在且唯一 —— 通过。
3. coord → ref → anchor —— 首跑失败；修复后需重启复验（运行中的进程吃的是旧模块）。
4. off-record `follow=False`、cacheRead 是否继承 —— 待验。
5. `resp` 校验通过、返回 JSON —— 待验。
6. 非 BaseModel 负路径 —— 待验。
7. 超时/重试路径 —— 待验。
