---
created: 2026-08-03
depends: []
description: Make llms config genuinely usable — Project.configs() single-source ConfigStore
  construction, then a `moss llms` CLI backed by pydantic-ai. Prep for dolores.
milestone: null
priority: P1
status: in-progress
status_note: 'Round 3 (2026-08-11): LLMFuncs.call produces cognitive anchors
  (export_anchor + LLMFuncResult.anchor, CallAnchor payload, CLI --export-anchor).
  Collision with cognitive-anchor v4.'
title: Llms Cli
updated: '2026-08-11'
---

# Llms Cli

> Use `moss features set-status llms-cli <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`contracts/llms.py` 的 LLMConfig 契约已写完、测试覆盖 43 个,但**没有任何运行时消费者**。
现有模型调用路径(`ghosts/atom/_meta.py`、`memento factory.py`)直接读 `ANTHROPIC_MODEL` 环境变量
建 `AnthropicModel`,完全绕过 LLMConfig。契约是"死配置"。

这是为 **dolores**(并行脑、思维状态切换)做的铺垫:llms config 必须真的可用,后面上手做 ghost
才有样例可依。目标是 `moss llms` CLI:配置检查(list/resolve)+ 集成验证可用性(call/test),
底层用 pydantic-ai(已是 `[ghost]` extra 依赖)。

**第一步(本次 scope)**:configs 调整 — ConfigStore 构造逻辑收口到 `Project.configs()`,
让 CLI 只依赖 project 发现就能拉起 mode 专属 ConfigStore,不需要 Host/Matrix。

**Round 2 (2026-08-09, in-progress)**:llms CLI 定位 = 环境调试工具 (与 CLI audio 同级) + 日常开发调试能力。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### 1. ConfigStore 构造逻辑收口到 `project.configs`(懒加载单例)

原来 ConfigStore 构造散落两处:`EnvConfigStoreProvider`(matrix,活跃)与
`WorkspaceYamlConfigStoreProvider`(host 风格,当前未注册),两者都是
`YamlConfigStore(workspace.configs(), mode_name=...)` 的手写副本。再加 CLI 第三处必然漂移。

`Project.configs`(property,懒加载单例)成为唯一构造出口,内部委托 `_configs()` 工厂方法;
两个 provider 改为薄委托 — `EnvConfigStoreProvider` 取 `project.configs`,
`WorkspaceYamlConfigStoreProvider` 调 `project._configs(on_save=, mode_name=, configs=)` 保留自身参数。
懒加载:project 级只构造一次,CLI 与 matrix 共享同一 store 视图,避免每次重新初始化。
matrix_impl 已在 `_prepare_container` 里 `container.set(Environment, ...)` / `container.set(Project, ...)`
(672-674 行),provider 从 IoC 拿 Project 走 set 单例,不走发现 — 委托链路可靠。
不搬 matrix 的整个 IoC 构建:那 95% 是运行时服务(adapter/presence/subprocesses/session/topic),不归 project。

### 2. mode 语义:None=从 env 推导,显式值=照用

`project.configs(mode_name=None)` → `'' if env.no_mode else env.mode_name`(匹配 EnvConfigStoreProvider)。
显式传入(含 `''`)原样传给 YamlConfigStore。
`WorkspaceYamlConfigStoreProvider` 用 `mode_name=self._mode or ''` 保持原 base 视图语义
(LocalConfigStore 里 None 与 '' 行为等价)。

### 3. CLI 依赖分类(depends gating)

`moss llms` 组常注册 — `list`/`resolve` 是纯配置检查,只需 project 发现(typer/rich/dotenv)。
`call`/`test` 用 `try: depend_ghost() except ImportError: pass else: register` 按需挂载
(仿 main.py 48-60 的 matrix pattern),满足"没有依赖不展示"。

### 4. pydantic-ai client 构造:惰性 import 的独立模块

不把 pydantic-ai import 进核心包。client 构造放独立可复用位置,惰性 import pydantic-ai,
CLI 与未来 Dolores/Ghost 共用。(决策细化后追加)

### 5. TokenCount + count_tokens — 同步、结构化返回、tiktoken 按协议估算

`LLMFuncs` 增加同步纯函数 `count_tokens(text, *, model=None, include_tokens=False) -> TokenCount`。
`TokenCount` 是 frozen dataclass:`service` / `model` / `encoding` / `count` / `estimate` / `tokens`。

- **同步 + docstring 声明性能风险**: CPU-bound BPE 分词, O(n), `include_tokens=True` 物化
  token id 列表 (长文本有内存开销)。协程调用者必须卸载到线程池 (asyncio.to_thread /
  run_in_executor)。风险写进 ABC docstring — 调用者是引擎无关的, 只看契约。
- tiktoken 0.13.0 无 `count_tokens` 捷径 (已验证), 计数 = `len(encode(text))`。
- **编码选择**: openai 协议用 `tiktoken.encoding_for_model` (gpt-4o 系 → o200k_base,
  旧模型 → cl100k_base), 未知模型回退 o200k_base; 非 openai 协议 (anthropic/deepseek)
  一律估算 — tiktoken 是 OpenAI 分词器, `estimate=True` 由调用方决定怎么标注。
- 依赖: tiktoken 由 ghost extra 的 `pydantic-ai-slim[openai]` 引入, 无需新增声明。

### 6. think effort 作为 call 参数, 不进 config — protocol adapter 落在 build_agent

mindflow 的 effort 刻度 (no/default...max) 映射为 pydantic-ai 的 **effort 字段**。已核实
pydantic-ai 2.5.0 同时暴露: 通用 `thinking` (bool | minimal..xhigh)、`anthropic_effort`
(low..xhigh/max)、`openai_reasoning_effort` (none..xhigh)。这是新式统一抽象; 旧式
`anthropic_thinking` (enabled+budget_tokens / adaptive / disabled) 仍在但不作主路径。

- effort 是 `LLMFuncs.call` 的参数 (CLI `--effort`), **不进 LLMConfig**。
- `build_agent` 按协议映射: anthropic → `anthropic_effort`; openai → `openai_reasoning_effort`。
- **冲突处理**: client.py 硬编码 `anthropic_thinking=disabled` 作默认基线; 传了 effort 就走
  effort 字段, 不再设 disabled thinking (两者互斥)。
- deepseek/doubao 等 kwargs 不一致时, 在 `build_agent` 的 protocol 分派里加分支 (protocol adapter)。

### 7. LLMFuncs 注册为 project 级 provider (懒加载, 无 ghost 不爆炸)

`LocalProject._default_providers()` 增补 `ProjectLLMFuncsProvider` (contract=LLMFuncs,
factory=PydanticAIFuncs, 惰性 import)。与 Subprocesses/JobSupervisor/ConfigStore/
ResourceRegistry 并列; workspace 用户在 ProjectManifest.providers 显式覆写即可覆盖。
fetch 时才 import pydantic-ai; 无 ghost extra 时 fetch 报干净错误, 不拖垮项目容器。
matrix 的 contracts() 校验不含 LLMFuncs, 不 fail-fast。

### 8. CLI 读配置走 project 容器, 不走 matrix; 防御边界收口

list 从配置读取开始就走 `Project.discover()` + `project.bootstrap()` (幂等: 载入 env +
container.bootstrap 触发 ConfigInstanceRegisterBootstrapper), `project.container.force_fetch(LLMConfig)`
取配置; call/count/test 额外 `force_fetch(LLMFuncs)` 取引擎。

- 比 audio CLI 轻: 无 `Matrix.new`, 无 cell 身份/网络/topic。调试工具在环境坏时也要能跑。
- 与运行时消费者共享同一容器 source of truth。
- **防御边界 = api_key + base_url 两个字段**。base_url 可能嵌凭据 (query/fragment),
  保持不显示, 不引入"消毒后展示"的复杂度。protocol/model/tags 已在 ModelRef 投影里。

## Implementation Notes

- 循环 import 规避:`project.configs()` 方法体内惰性 import `contracts.configs`;
  `contracts/configs.py` 的 provider factory 内惰性 import `Project`。
- `WorkspaceYamlConfigStoreProvider` 活跃代码未注册(仅 contracts 导出),改动低风险。
- 既有不一致(顺带处理):两个 stub 树对 LLMConfig 注册语义不同 —
  主 `stubs/workspace` 用实例(仅内存),`host/stubs/workspace` 用类型(文件持久化)。影响开箱体验。
- 仓库无 `.env.example`(workspace 自带 `.env.example` 机制),LLM env 变量需补 DEEPSEEK 家族。