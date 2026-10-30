# 03 — 并发单位是调用，不是对象

## 动机

要"session 任意一个位置可以高并发"。先得弄清 dsh 里哪个对象能被并发复用——
直觉上 `Request` / prepared call 都不像并发单位。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source

grep -n "a prepared LLM call can only be dispatched once" packages/llm/llm/src/index.ts
sed -n '965,1015p' packages/llm/llm/src/index.ts   # prepareCall 的 stream 闭包
sed -n '270,300p'  packages/llm/llm/src/index.ts   # 适配器侧 prepareCall 基类实现
sed -n '1160,1185p' packages/llm/llm/src/index.ts  # LlmRuntime.stream
sed -n '1,10p'     packages/core/agent-loop/src/constants.ts
grep -rn "semaphore\|maxInFlight\|concurren" --include=*.ts packages/llm | grep -v tests
```

## 结论

- `[读码]` **`PreparedLlmCall` 是一次性的。** 第二次 `stream()` 抛
  `INVALID_PREPARED_CALL`（*a prepared LLM call can only be dispatched once*，
  `packages/llm/llm/src/index.ts:1001-1010`）。它还要求 `options` 与 prepared 的 config
  完全相等。→ 它**不是**并发单位。
- `[读码]` **`LlmRuntime.stream(options)` 无状态**：`stream → streamWithRegistration →
  adapterStream`（`index.ts:1170-1184`），没有 dispatch 标记，**可以并发调 N 次**。
  未 prepared 时它在内部按需 `adapter.prepareCall`（`index.ts:1080`），代价是每次重新解析模型。
- `[读码]` **`PreparedAdapterCall.stream` 可复用**：基类实现是
  `{ model, stream: options => this.stream(options) }`（`index.ts:290-300`）。
  想省掉每次的模型解析，用它可以 N 次复用；它不带 dispatch 状态。
- `[读码]` **LLM 层没有并发闸门**：`semaphore / maxInFlight / concurrency` 在 `packages/llm`
  里零命中。全仓唯一的并发上限在 agent loop：
  `DEFAULT_MAX_PARALLEL_TOOL_CALLS = 10`（`packages/core/agent-loop/src/constants.ts:5`），
  而且那是**工具调用**的，不是模型请求的。
- `[推断]` 因此"任意位置高并发"= `buildForkSeed(log, k)` 取前缀 + `llm.stream()` 并发 N 次；
  闸门由调用方自己设，provider 的 429 不算 dsh 的错。

## 观点

并发单位是**调用**，不是对象。dsh 把"一次请求需要的所有状态"都压进了 `options`，
所以只要不带状态，什么都能并发。

代价两条：① 每次 `stream` 都重新解析模型（复用 `PreparedAdapterCall` 可省）；
② 没有背压——N 开太大就是 429，而 429 由 provider 决定，不由 dsh 兜。

判据：并发数应该由一个**显式常量**控制（形状抄 `DEFAULT_MAX_PARALLEL_TOOL_CALLS`），
而不是靠 provider 报错来反向发现。否则"高并发"会变成"高失败率"。
