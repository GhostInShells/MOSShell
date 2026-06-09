---
title: LLM Config Contracts
status: in-progress
priority: P0
created: 2026-06-10
updated: 2026-06-10
depends: []
milestone:
description: >-
  Pure-config LLM service/model declaration (ConfigType), env var resolution for secrets,
  content-based model capability filtering, with Matrix exposure for cross-process discovery.
---

# LLM Config Contracts

## Motivation

旧 GhostOS 有完整的 `LLMs / LLMApi / LLMDriver` 五层抽象。模型迭代速度远超代码迭代速度，
adapter 层写了就是技术债。YAML 里加一行模型名，比写 class + 注册 + 测试快两个数量级。

当前 `.env.example` 只有一种模型，用起来不方便。需要一个纯配置驱动的 LLM 声明系统：
Service（连接）+ Model（能力），通过现有的 ConfigStore + manifests 发现路径暴露到 Matrix。

## Key Decisions

### 1. 纯 ConfigType，不做 adapter 层

模型迭代太快。ServiceConf + ModelConf 的纯配置方案，加模型只需改 YAML。
api_key 走现有 `$ENV_VAR` → `resolve()` 机制，敏感值不落盘。

### 2. Service 和 Model 分离

Service 管连接（base_url, api_key, api_type），Model 管能力（model_name,
context_window, protocols）。一个 Service 可挂多个 Model，Model 通过
service name 引用 Service。

### 3. protocols 是 content 能力声明，不是抽象协议

`protocols: ["text", "image"]` 直接对应 MOSS 的 `Content.type`。发 prompt
前按此字段过滤 content。`converter` 字段（import path）是扩展点 — None 时
不支持的 content 丢弃，非 None 时走转换。

### 4. model_type 做降级键，default_model 做最终兜底

`get_model(service="", model_type="", *, no_fallback=False)` — 精确匹配优先，
无匹配降级到 `default_model`。`no_fallback=True` 抛 KeyError。
预定义常量：`default / pro / flash`，字段类型 `ModelType | str` 支持扩展。

### 5. model settings 不管

temperature / top_p 等是调用时业务参数，放 config 会频繁变更污染 git log。
Config 只存"谁在哪能干什么"。

### 6. 修复 _environ dead code

`LocalConfigStore.__init__` 接受 `environ` 但从未使用 — `_resolve_config_data_from_env`
直接读 `os.environ`。现已穿透：`environ` 参数从 `LocalConfigStore` → `ConfigType.resolve()`
→ `_resolve_config_data_from_env()`，默认为 None 时 fallback 到 `os.environ`。

## Implementation Notes

- 本 session 完成: contracts 层抽象 (`ghoshell_moss.contracts.llms`) + `_environ` 修复
- 待后续: manifests 注册（`MOSS.manifests.configs` 中暴露 LLMConfig）、默认 YAML 生成、
  Matrix 上的查询/订阅接口、converter 扩展点的具体调用约定
