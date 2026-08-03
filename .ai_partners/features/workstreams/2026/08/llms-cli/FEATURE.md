---
created: 2026-08-03
depends: []
description: Make llms config genuinely usable — Project.configs() single-source ConfigStore
  construction, then a `moss llms` CLI backed by pydantic-ai. Prep for dolores.
milestone: null
priority: P1
status: completed
status_note: 'CLI core done: list/call/test, no-fallback, three protocols (anthropic+openai).
  .env.example and stub llms.yml deferred.'
title: Llms Cli
updated: '2026-08-03'
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

## Implementation Notes

- 循环 import 规避:`project.configs()` 方法体内惰性 import `contracts.configs`;
  `contracts/configs.py` 的 provider factory 内惰性 import `Project`。
- `WorkspaceYamlConfigStoreProvider` 活跃代码未注册(仅 contracts 导出),改动低风险。
- 既有不一致(顺带处理):两个 stub 树对 LLMConfig 注册语义不同 —
  主 `stubs/workspace` 用实例(仅内存),`host/stubs/workspace` 用类型(文件持久化)。影响开箱体验。
- 仓库无 `.env.example`(workspace 自带 `.env.example` 机制),LLM env 变量需补 DEEPSEEK 家族。