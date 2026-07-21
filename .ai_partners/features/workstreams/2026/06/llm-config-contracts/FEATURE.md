---
created: 2026-06-10
depends: []
description: Pure-config LLM service/model declaration (ConfigType), env var resolution
  for secrets, content-based model capability filtering with text degradation fallback,
  with Matrix exposure for cross-process discovery.
milestone: null
priority: P0
status: completed
title: LLM Config Contracts
updated: '2026-06-25'
---

# LLM Config Contracts

## Motivation

旧 GhostOS 有完整的 `LLMs / LLMApi / LLMDriver` 五层抽象。模型迭代速度远超代码迭代速度，
adapter 层写了就是技术债。YAML 里加一行模型名，比写 class + 注册 + 测试快两个数量级。

当前 `.env.example` 只有一种模型，用起来不方便。需要一个纯配置驱动的 LLM 声明系统：
Service（连接）+ Model（能力），通过现有的 ConfigStore + manifests 发现路径暴露到 Matrix。

## Key Decisions

### 1. 纯 ConfigType，不做 adapter 层

模型迭代太快。纯配置方案，加模型只需改 YAML。
api_key 走现有 `$ENV_VAR` → `resolve()` 机制，敏感值不落盘。

### 2. Provider = Service + Model 阵容

`Provider` 封装一个服务 + 它的模型阵容：`service: ServiceConfig` 管连接
（base_url, api_key, protocol），`default: ModelConfig` 管默认模型，
`models: dict[str, ModelConfig]` 管其他模型。`LLMConfig` 持有一个
`default: Provider` 和 `providers: dict[str, Provider]`，default 也是
可被按名检索的 provider。

### 3. content_types 声明 + 三层降级

`ModelConfig.content_types: list[str]` 声明模型原生支持的 `Content.type`。
`converters: dict[str, str]` 按 content type 映射 converter import path。
发 prompt 前依次走：**原生支持 → converter 适配 → 文本退化**。
最后一层用 `Message.content_as_string()` 确保模型至少能"读到"所有内容，
不再静默丢弃。

### 4. tag 做模型标签，provider+model 做查找键

`get_model(provider="", model="", tag=None, *, no_fallback=False)` —
零参数返回默认 provider 的默认模型；provider 按名匹配（同时查 default
和 providers）；model 按名在所有 provider 的 models 字典中精确搜索；
tag 对解析后的 ModelConfig 做 unwrap（如 `small_fast_model` → 实际模型名）。
`no_fallback=True` 无匹配抛 KeyError。
预定义标签常量：`DefaultModelTag = Literal['small_fast_model', 'flash', 'pro']`。

### 5. model settings 不管

temperature / top_p 等是调用时业务参数，放 config 会频繁变更污染 git log。
Config 只存"谁在哪能干什么"。

### 6. 修复 _environ dead code

`LocalConfigStore.__init__` 接受 `environ` 但从未使用 — `_resolve_config_data_from_env`
直接读 `os.environ`。现已穿透：`environ` 参数从 `LocalConfigStore` → `ConfigType.resolve()`
→ `_resolve_config_data_from_env()`，默认为 None 时 fallback 到 `os.environ`。

## Implementation Notes

- [x] contracts 层抽象 (`ghoshell_moss.contracts.llms`)
  - `ServiceConfig` — 连接配置 (name, base_url, api_key, protocol)
  - `ModelConfig` — 模型配置 (model, tags, content_types, converters, context_window, max_output_tokens)
  - `Provider` — 供应商 = Service + 模型阵容 (service, default, models dict)
  - `ResolvedModel` — 查找结果 = model + service 对，带 `client_protocol` property
  - `LLMConfig` — 配置中心 (default Provider + providers dict)，提供 `get_model` / `list_models` / `get_service` / `services`
  - `MessageContentConverter` — converter 抽象基类
  - `register_converter()` / `clear_converters()` — 公开的 converter 注入/清空 API
- [x] 三层内容降级 — `ModelConfig.convert()` 原生 → converter → 文本退化 (`Message.content_as_string`)
- [x] `_environ` 修复 — `LocalConfigStore` → `resolve()` 透传，增加 list 内 `$ENV_VAR` 递归解析
- [x] manifests 注册 — `.moss_ws/src/MOSS/manifests/configs.py` + stub
- [x] tests — `tests/ghoshell_moss/default/contracts/test_llms.py` (43 tests)
  - 单元: ServiceConfig, ModelConfig (accepts / unwrap_tag / convert 三层降级 / converter 适配), Provider, ResolvedModel, LLMConfig
  - 集成: YamlConfigStore + LocalStorage (save/load roundtrip, env var resolution, mode-specific config, cache invalidation)
- [x] Matrix 查询/订阅接口 — `Matrix.query_config()` + `Matrix.on_config_change()` + `LocalConfigStore` 的 `on_save` 钩子，MatrixImpl 串联