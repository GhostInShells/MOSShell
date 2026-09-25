---
created: 2026-08-06
depends:
- llms-cli
description: 函数化大模型单轮请求——单一字符串 prompt 协议 + 结构化响应。激活 LLMConfig 死配置， 把 moss llms call
  协议化，作为 ghost 运行时可 scale 的模型调用资产。
milestone: null
priority: P1
status: completed
status_note: 引擎/锚/@ 文件协议/MossLLMFuncs 分层/run_benchmark 参数范围全部落地。讨论结论见 FEATURE 脚注。
title: Model Func
updated: '2026-08-11'
---

# Model Func

> Use `moss features set-status model-func <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Model func 是函数化的大模型单轮请求：结构化入参 → 结构化出参。核心不是让模型调复杂工具，
而是把模型预训练能力挖掘成函数——翻译、多分类、识别、判断。本质是拿模型一个结构化输出，
用来做特殊处理。

**为什么现在**：

1. **激活 LLMConfig 死配置**。`llms-cli` workstream 自己记录：`contracts/llms.py` 契约写完、
   43 个测试覆盖，但没有任何运行时消费者——atom ghost / memento 直接读 env 建模型，完全绕过
   LLMConfig。model func 是第一个真正消费 LLMConfig 的运行时抽象，补完 llms-cli 未竟目标。
2. **把 call 协议化**。`moss llms call` 现在能查 model 是否可用（对 project debug）。协议化后
   变成一个通用单轮结构化调用命令：`instruction + prompt -> response model`。
3. **CLI 即运行时**。moss CLI 是 ghost 的运行时工具链（`moss-self-channel` 已验收：ghost 通过
   CTML `<moss:exec>` 或 bash 调 moss 命令）。任何加进 CLI 的能力自动成为 ghost 的可调用资产。
   model func 是可 scale 的资产——定义一次，CLI 化后用杠杆放大到 channel、脚本、benchmark。
4. **准确率与性能测试**。moss 有大量小任务要做结构化输出实验（如 flash 单 token 多分类判断
   "话是否说完"输出 0-9 打分）。model func + benchmark 让这变成可脚本化、可复用、可评估的闭环。
5. **为 dolores（并行脑）铺路**。并行脑的思维单元最小形态就是一个 model func。

## Design Index

- 前置依赖: `llms-cli` workstream — ConfigStore 收口 + `build_agent()`
- 契约路径: `src/ghoshell_moss/contracts/llms.py` — 既定依赖，实现无关
- 沙箱编译: `src/ghoshell_moss/core/codex/compiler.py`
- 模块发现参考: `src/ghoshell_moss/channels/module_channel.py`

## Key Decisions

### 1. 单一字符串 prompt 协议

Request 不做 `to_prompt()` ——该技术约定无法框架无关。统一为**单一字符串 prompt 协议**：
请求就是字符串（prompt + 可选 instruction），响应是 BaseModel。这是模型调用的最小公约数，
pydantic-ai / 裸 API / 未来消息引擎都符合。有上下文协议就是 agent —— model func 明确
单轮、无状态、无工具。

### 2. 最简形态：instruction + prompt -> response model

全部上下文由字符串定义，字符串怎么生产不管。最多是 `instruction + prompt -> response model`。
未来要做消息协议时，走 `@ 文件路径` 拆上下文，用文件约定协议做 content block（第一版不碰）。

### 3. instruction 三种形态，从文件约定起步

instruction 在项目目录可能有三种：1. 字符串 2. 单个文件 3. 可执行脚本的执行结果。
3 → 2 → 1 几乎等价。第一版从文件约定开始，不展开。

### 4. 引擎抽象，pydantic-ai 为默认实现

model func 定义"模型调用最小协议"层，pydantic-ai 的 Agent 是实现该协议的引擎之一。
model func 引擎依赖 `contracts/llms.py` 的 `LLMConfig.get_model()` / `ResolvedModel` /
`llms/client.py build_agent()`。不同引擎可换，协议不变。

### 5. 返回结构

```python
class ModelFuncResult(BaseModel):
    content: str | None = None      # 原始字符串（有则给）
    response: BaseModel | None = None  # 建模后的结构化输出（无则 None）
    usage: ... | None               # token 开销
    elapsed: float                  # 秒
    retries: int
```

真实返回值是 json；CLI 不加 `-j` 时打印成字符串。

### 6. CLI 化 = 协议化（先 cli，回头内存化）

`moss llms call` 加参数：
- `-i --instruction`：纯文本或文本文件，作为 instruction
- `-j --json`：输出 json
- `-v --verbose`：更多讯息
- `-r --response-model`：`module:attr` 指向 BaseModel 类（文件则沙箱编译，走 Compiler）
- `-n`：内存中连续跑 n 次

CLI 是 ghost 的接口，不是开发者便利——CLI 化等于给 ghost 上架能力。内存化是同一能力的优化形态。

### 7. Benchmark：bench.md + xxx.case.jsonl

- `bench.md`（YAML frontmatter + markdown）：设计动机、验证目标、产物结构（response_model）、
  缺省 instruction（或 instruction_ref 指针）、scorer 指针、expect 语义。**元信息模型无关**——
  模型由运行时 `--provider/--model/--tag` 选择，同一 benchmark 可换模型重跑对比。
- `xxx.case.jsonl`：每行 `{id, prompt, expect?, instruction?}`，可脚本注入。
- 产物分层：`result.jsonl`（逐 case 合并 prompt/expect/response/usage/elapsed/retries）+
  summary（平均延迟/准确率/错误分布）。
- 打分闭环：scorer 本身是一个 model func（`score(prompt, expect, response) -> Score`），
  loop 把测试结果发给打分器——体系自举。

### 8. 克制边界

不做 `__instruction__() -> str` / `__prompt__(prompt) -> list[str]`，以后加。
不做消息协议。第一版：协议 + 引擎 + CLI + 一个样例。

## Implementation Notes

- 依赖门控：pydantic-ai 是 `[ghost]` extra。仿 `llms_cli.py` 的
  `try: depend_ghost() except ImportError` 模式，无依赖不展示命令。
- env var pre-flight：`_resolve_for_call` 检查 `$ENV_VAR` 未 set 时报错，不静默 fallback。
- 契约路径已定：`contracts/llms.py`（363 行后 model func 区块），具体实现无关。
- 模块发现复用 `channels/module_channel.py` 的 `_iter_public_callables` 模式（尊重 `__all__`）。

### 契约最终形态（2026-08-06，review 后定稿）

- `LLMFuncs`(ABC)：`call()` / `run_benchmark()`。引擎无关，pydantic-ai 首个实现。
- `LLMFuncResult[R]`(BaseModel+Generic) 强类型 + `LLMFuncResultRecord` 弱数据，
  `to_record()` 强转弱（result 展平 dict）。pydantic v2 原生支持泛型 BaseModel，
  故不用 ABC 子类化结果。
- `ModelRef`：`ResolvedModel` 的不泄密同构投影（排除 api_key/base_url），
  list 展示与 benchmark 溯源共用；`resolve()` 从 `LLMConfig` 反查复活密钥。
- benchmark 四层：`BenchmarkMeta`(bench.md frontmatter, 模型无关) / `BenchmarkCase`
  (label/prompt/instruction/expected/times) / `BenchmarkRun`(meta+model=ModelRef) /
  `BenchmarkRecord`(run+results)。markdown/jsonl io 标 NotImplementedError。
- 决策：`BenchmarkRun.model` 必须无密钥（持久化安全）；`run_benchmark` 带 `cwd`
  解析 case 文件路径。
- 2026-08-08 依赖升级：`pydantic-ai` → `pydantic-ai-slim[anthropic,openai]`，
  去除 fastmcp-slim → mcp<2.0 传递依赖锁，mcp 2.0.0 独立升级不冲突。
  59 tests 全过（43 contract + 15 funcs + 1），回归基线在
  `.ai_partners/regressions/llms/REGRESSION.md`。

## TODO

- [x] contract 设计（review 模式：AI 出方案，人类 review）— 2026-08-06 定稿
- [x] 引擎实现（pydantic-ai）— `llms/funcs.py` PydanticAIFuncs: call + run_benchmark
- [x] CLI `moss llms call` 参数扩展 — `-i/-j/-v/-r/-n`
- [x] IoC 定义（引擎不需要容器；导出点见 `architecture.py`）
- [x] 引擎测试 — 15 mock + CLI smoke + 7 cases 真实 benchmark 全链路闭环
- [x] 样例：话说完检测（单 token 多分类打分）— example/ 目录, 7 cases 全链路闭环通过
- [x] benchmarks 体系 — `.ai_partners/benchmarks/` 约定, utterance-end-detection 首发
- [x] 依赖升级 — pydantic-ai-slim[anthropic,openai] + mcp 2.0.0, 消除 fastmcp-slim 传递冲突

## Next — done (2026-08-11)

**Prompt protocol** — 扩充 model func 的入参从纯字符串到结构化 content block:

- [x] ``@ 文件路径`` — 行首 `@` 文件引用 → `message.prompt.message_from_prompt` → `list[Message]`
- [x] 二进制理解 — Base64Image content, 经 `llms/pydantic_ai_adapter/conversion` 转 pydantic-ai parts
- [x] 描述文件 — file meta 弱容器 (tag="file" + path/type/size), `expose_file_meta` flag 开关

契约分层落地: `LLMFuncs.call(prompt: str)` moss-free 抽象, `MossLLMFuncs` 开始 moss
耦合 (`call_prompt` / `call_messages`), `PydanticAIFuncs(MossLLMFuncs)` 引擎共享私有
`_call_impl`。run_benchmark 支持 effort/thinking + per-case 配置 (BenchmarkCase.thinking/
effort)。见 cognitive-anchor FEATURE v4.3 与 benchmark Findings。

## 讨论脚注 (2026-08-11)

utterance-end-detection 单 token 改造 + 策略 A/B 的讨论结论 — llmsfuncs 体系与
mindflow 架构的关联:

- **flash 分句级单 token 判断足够快** — ~1s (含手机热点网络) 在
  "ASR 尾包 + VAD 1.5s + 链路开销 + 模型首包" 的反应预算内可接受。粒度是
  clause 级, 不是逐词 — 决策按分句边界摊销。
- **分层判断**: benchmark 证明 0/9 两端 (硬切/完整句) 可靠, 真正吃不准的是
  歧义中间带 (u5)。所以端侧廉价手段 (尾句 embedding 匹配 / 规则) 抓清晰端,
  模型只在歧义句上跑 — 把 ~1s 从每个 utterance 摊到只有歧义才付。
- **缓存阴性**: 相同 case 无缓存提速 — 1s 是推理 + 网络本征, 不是可缓存前缀。
  优化杠杆在分层与粒度, 不在缓存。
- **mindflow 进模型推理内部 = 内观的自然延伸**: `--thinking` 已让模型"持有
  既有立场"; mindflow-in-inference 是把立场结构化 — 感知→仲裁→行动 压缩进
  生成本身。clause 级单 token 判断正是"冲动→决策"压成一个 token 的形态,
  是 mindflow 推理内部化的最小样例。
- **云端全双工不可控, 根因是智力没解耦** — 工程层在模型外做编排, 模型内部
  是黑盒。MOSS 的 channels (主动传感器) + mindflow (仲裁) + CTML (控制) 是
  把解耦显式化的尝试, benchmark 给这个解耦供可测结论。