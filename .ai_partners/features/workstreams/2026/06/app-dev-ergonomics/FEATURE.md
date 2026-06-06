---
title: App Dev Ergonomics — app 模板增强、误用警告、测试与依赖治理
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-04
depends: [web-fetch-apps]
milestone:
description: >-
  修复 web-fetch-apps 开发中暴露的 4 个摩擦点：app 模板过于 minimal、uv sync 污染主 venv、
  测试目录约定缺失、channel 测试套件缺失。入体系：stubs/app、start.md、howtos。
---

# App Dev Ergonomics

## Motivation

`web-fetch-apps` 开发过程中暴露了 app 体系和主项目之间的 6 个认知摩擦点。其中 2 个（F5/F6）已在代码中修复，剩余 4 个需要入体系——模板、文档、howto。这些摩擦点会让每个新入场的 AI 实例或人类开发者重复踩坑。

**阻塞关系**：app 体系是 MOSS 的核心能力交付机制。摩擦点不修，后续每个 app 开发都会浪费一轮踩坑。

## 待修摩擦点

### F1: `moss apps create` 模板过于 minimal

**根因**：`stubs/app/` 是最早的占位实现，从未随 app 体系成熟而更新。

**修复**：
- `stubs/app/main.py`：升级为 Matrix.discover 入口模式，含 Channel 构建示例注释
- `stubs/app/CLAUDE.md`（新增）：独立 project 思维、测试放自身目录、依赖用 uv run 管理
- `stubs/app/APP.md`：补充 description 字段

### F2: `uv sync --active` 在 app 目录污染主 venv

**根因**：app 的 pyproject.toml 中 ghoshell-moss 通过 editable path 引用主项目，uv workspace 机制下 `--active` 修改共享 venv。

**修复**：`start.md` 增加 "Common Command Misuses" 区块，每条列出错误命令、后果、正确做法。

### F3: App 测试目录约定缺失

**根因**：无明确约定，直觉会把测试放到主项目 `tests/` 下。但 app 是独立项目。

**修复**：
- `stubs/app/CLAUDE.md` 明确测试约定
- 新增 `howtos/app-dev/` 目录

### F4: 缺少内建 channel 测试套件

**根因**：测试 channel 需要了解 `ctml_shell_test()` API 和 import 路径，入门成本高。

**修复**：
- `channel_builder` 增加 `test_channel()` 便捷函数
- 新增 `howtos/app-dev/test-an-app.md`

## Design

### 改动文件

| 文件 | 变更 | 解决 |
|------|------|------|
| `stubs/app/main.py` | 升级为 Matrix.discover 入口 | F1 |
| `stubs/app/CLAUDE.md` | **新增** — app 开发上下文 | F1, F3 |
| `stubs/app/APP.md` | 补充 description | F1 |
| `cli/start.md` | 新增 "Common Command Misuses" 区块 | F2 |
| `howtos/app-dev/README.md` | **新增** — app-dev 领域概述 | F3, F4 |
| `howtos/app-dev/test-an-app.md` | **新增** — channel 测试方法 | F3, F4 |
| `core/blueprint/channel_builder.py` | 新增 `test_channel()` 函数 | F4 |

### test_channel() 签名

```python
async def test_channel(
    *channels: Channel,
    ctml: str,
    timeout: float | None = None,
) -> list[CommandTask]:
    """Convenience wrapper around ctml_shell_test for app channel testing.

    Usage in app tests:
        from ghoshell_moss.core.blueprint.channel_builder import new_channel, test_channel

        chan = new_channel(name="my_app")
        @chan.build.command()
        async def greet(name: str) -> str:
            return f"Hello, {name}"

        tasks = await test_channel(chan, ctml='<apps.my_app:greet name="world" />')
        assert await tasks[0] == "Hello, world"
    """
```

**注意**：`test_channel` 不在 `__all__` 中声明——这是开发者便利函数，不是给运行时模型用的 API。用法记录在 howto 中。

### start.md "Common Command Misuses" 条目

1. `uv sync --active` in app dir → pollutes main venv → use `uv run` or `moss apps test`
2. `mkdir apps/xxx` → skips template → use `moss apps create`
3. Writing tests under `tests/ghoshell_moss/apps/` → violates app independence → put in app's own `tests/`

## Key Decisions

### KD1: app-dev howto 独立目录

**决策**：在 `howtos/` 下新建 `app-dev/` 子目录，与 `host-dev/` 平级。

**理由**：app 开发者是独立的用户画像——不关心 manifests、IoC、provider。分开路由，AI 按需进入。

### KD2: test_channel 不在 __all__ 中

**决策**：加入 `channel_builder.py` 但不进 `__all__`。

**理由**：便捷函数，非核心抽象。进 `__all__` 会让 `get-interface` 暴露给模型，增加认知噪音。

### KD3: start.md 误用区块只列操作级错误

**决策**：只列具体命令的对比（错误 vs 正确），不超过 5 条。

**理由**：start.md 是认知入口，过长警告淹没核心信息。

## Implementation Plan

1. stubs/app/ 增强：main.py + CLAUDE.md + APP.md
2. start.md：新增 "Common Command Misuses" 区块
3. howtos/app-dev/：新建 README + test-an-app.md
4. channel_builder：新增 test_channel()
5. 提交

## 验收标准

1. `moss apps create test/myapp` 生成的模板含 Matrix.discover 入口 + CLAUDE.md
2. `moss howtos list` 能看到 `app-dev/test-an-app.md`
3. `moss --ai start` 输出含 "Common Command Misuses" 区块
4. `test_channel()` 可通过 `from channel_builder import test_channel` 使用

---

*设计: DeepSeek V4 Pro 与人类工程师, 2026-06-04*
