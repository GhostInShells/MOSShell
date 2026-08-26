---
created: 2026-08-03
depends: []
description: Make llms config genuinely usable — Project.configs() single-source ConfigStore
  construction, then a `moss llms` CLI backed by pydantic-ai. Prep for dolores.
milestone: null
priority: P1
status: completed
status_note: content_types interception wired, default deepseek family, container
  injection
title: Llms Cli
updated: '2026-08-26'
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

### 9. content_types 拦截接线 + container 构造注入 (2026-08-26)

`ModelConfig.convert()`(accepts → converter 适配 → 文本退化)此前是死代码——只在契约测试里活,
管线 `call_messages` 直接 `messages_to_parts(prompt)`, 把图片裸发给纯文本模型 (deepseek-v4-pro) 报错。

修复:
- `call_messages` 解析 resolved 后, 对每个 message 先 `resolved.model.convert(container, msg)`
  再 `messages_to_parts`, 实现按模型 `content_types` 的原生保留 / 降级 (图片→文本占位)。
- container 从 LLMFuncs 构造时注入: `PydanticAIFuncs(logger, config, container=None)`,
  None 时实例化空 `ghoshell_container.Container`; `ProjectLLMFuncsProvider.factory` 注入 `container=con`
  (单例, 长期持有无碍)。
- 默认 LLMConfig: **默认家族设为 deepseek, 默认模型 `deepseek-v4-flash`** — 因 MOSS 绝大多数
  模型参与改造都是 deepseek 家族。默认 provider = deepseek (default=`deepseek-v4-flash` text-only;
  models 含 `deepseek-v4-pro` text-only 与 `deepseek-v4-flash-vision-exp` text+image);
  providers 另留 anthropic (非默认) 与 deepseek_openai (openai 协议)。模型名用字面量
  (自包含可再生), base_url/api_key 用 $ENV。`get_model()` 零参 → deepseek-v4-flash。
- `.moss/configs` 下 ConfigType dump 生成物 (llms/audio/mcp/tts 等) 删除, 该目录加 `.gitignore`
  前缀匹配 (形如 llms.*), 启动时从类默认值重建。

## Implementation Notes

- 循环 import 规避:`project.configs()` 方法体内惰性 import `contracts.configs`;
  `contracts/configs.py` 的 provider factory 内惰性 import `Project`。
- `WorkspaceYamlConfigStoreProvider` 活跃代码未注册(仅 contracts 导出),改动低风险。
- 既有不一致(顺带处理):两个 stub 树对 LLMConfig 注册语义不同 —
  主 `stubs/workspace` 用实例(仅内存),`host/stubs/workspace` 用类型(文件持久化)。影响开箱体验。
- 仓库无 `.env.example`(workspace 自带 `.env.example` 机制),LLM env 变量需补 DEEPSEEK 家族。

## Failure Mode — 框架重力 (2026-08-15)

本次交付的 PydanticAIFuncs 把最初设计意图 (IoC 配置绑定 + 自解释句柄 + 使用时零传参)
静默替换成了 pydantic-ai agent 原生范式 (无状态 + ResolvedModel 显式传参 + 装线搬进 CLI):

- `PydanticAIFuncs()` 无构造入参, 每个方法要求显式传 `model: ResolvedModel`; LLMConfig
  与 LLMFuncs 在 IoC 里从未结合成「配置好的句柄」。
- 装线 (`_load_config` → `_resolve_for_call` → 传 resolved) 全堆在 CLI; 主调用路径
  甚至硬编码 `PydanticAIFuncs()`, 绕过了 `ProjectLLMFuncsProvider`。
- `ModelConfig.content_types / converters / accepts / convert` 这套「按模型能力适配内容」
  契约成了死代码 — 引擎用 `message_to_parts` 的硬编码映射平行处理内容, 不查能力声明。

根因: 融合 pydantic-ai agent 范式与 OOP/组件范式时, 重力落在框架原生范式; 模型复刻了
「行为」而非「意图」, 全程无对话、无标记、静默提交。dogfood 只暴露行为 bug (如 @ 图
未发出), 暴露不了意图侵蚀 — 被侵蚀后的设计行为上照样"能用"。

修复方向 (未实施): 构造入参至少含 logger (默认可空) + LLMConfig (默认可空 → 只支持
anthropic 的实现); 调用方只传配置项上的路径 (provider/tag/model), 不传 ResolvedModel;
可变参数 (temperature 等) 从配置项拆构造对象, 不透传。

## Refactor 完成 (2026-08-16)

上述修复方向已实施:

- `PydanticAIFuncs(logger=None, config=None)` 构造函数绑定 config + logger;
  `config=None` 回退 `LLMConfig().resolve()`。
- 调用面反向传参全部改掉: `model: ResolvedModel` → 配置路径 `provider/model/tag`,
  引擎内部 `_resolve()` 解析并校验 env var 就绪。
- 可变采样参数 (temperature / max_output_tokens) 收进 `CallSettings` 对象, 不再逐 kwarg 穿透。
- logger 用 `logger or get_moss_logger()` 兜底; `_call_impl` 打点: 调用前
  (service/model/effort/settings)、失败 (exception + traceback)、调用后 (elapsed)。
- IoC provider 注入 config (`ConfigStore.get_or_create(LLMConfig())`) + logger
  (`con.get(LoggerItf)`); CLI 改从 IoC 取绑定 funcs, 只传路径; `--no-fallback` flag 移除。

涉及: contracts/llms.py, pydantic_ai_adapter/{funcs,client}.py,
project/providers/llms_provider.py, cli/llms_cli.py, resources/markdown_kb, 测试三套。
llms 测试 42 全绿; 识图实测图片块已真实发出 (deepseek-v4-pro 回「格式不支持」而非「看不到本地文件」)。

## TODO (下一步)

- [x] 拦截修复: 把 `ModelConfig.convert()` (content_types 过滤) 接进 `call_messages` 管线。
  已接: `call_messages` 先对每个 message 跑 `convert(container, msg)` 再 `messages_to_parts`;
  container 构造注入。`convert()` 从死代码变活管线——图片喂纯文本模型降级为 content_as_string
  占位文本 (deepseek-v4-pro), 不再裸发报错。默认配置补 deepseek 三兄弟 (vision-exp 收图)。