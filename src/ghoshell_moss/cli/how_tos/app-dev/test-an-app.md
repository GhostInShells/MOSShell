---
title: Test a MOSS App
description: 三层递进测试 app：本地验证 channel 逻辑、MCP 路径验证闭环、运行时 Ghost 集成。面向 app 开发者和 AI 协作者。
---

# How to Test a MOSS App

## 背景

App 的形态决定了测试策略。不是所有 app 都需要 CTML 测试——先确认你的 app 是否暴露 channel。

- 暴露 channel → 三层递进：本地 CTML → MCP → 运行时
- 纯进程 / GUI / Sensor → 测试你的业务逻辑，MOSS 只管理启停

## 第一层：本地测试

### 测试如何拿到 channel 定义

`moss apps create` 生成的 `main.py` 将 channel 定义内联在 `if __name__ == "__main__"` 保护块中——测试文件无法 `from main import channel`。两种解法：

**解法 A（推荐）：提取到 src/ 模块**

将 channel 构建逻辑抽到 `src/` 下的模块中。`main.py` 和 `tests/` 都从它 import。

```
src/
└── my_app/
    └── channel.py      # 定义 build_channel() → 返回 channel 对象
main.py                  # from my_app.channel import build_channel
tests/
└── test_commands.py     # from my_app.channel import build_channel
```

适用场景：app 有 `pyproject.toml`，代码超过一个文件。

**解法 B：在测试中重建 channel**

直接在测试文件里用 `new_channel()` 重建相同的 channel 结构。`test_channel()` 是这个模式的最简封装：

```python
import pytest
from ghoshell_moss.core.blueprint.channel_builder import new_channel, test_channel

@pytest.mark.asyncio
async def test_extract():
    chan = new_channel(name="web_trafilatura")

    @chan.build.command(always_observe=True)
    async def extract(url: str) -> str:
        return f"content of {url}"

    tasks = await test_channel(chan, ctml='<apps.web_trafilatura:extract url="https://example.com" />')
    assert len(tasks) == 1
    assert "example.com" in await tasks[0]
```

适用场景：单文件 app，channel 定义简单，不引入 `pyproject.toml`。

**通用原则**：测试只要能 import 到你的代码即可——文件放在哪里不是硬约束。app 目录下的 `tests/` 是约定，不是要求。

### Channel 测试的边界

`test_channel()` 是基线——单命令调用和简单结果验证。作用域（until=all/any）、observe、cancel、嵌套 channel 等复杂场景需要 `ctml_shell_test` 的完整 API。读它的源码，搜索项目中已有的测试用法，理解后再用。

运行：`uv run pytest tests/ -v`（如果有 `pyproject.toml`）。没有则直接 `pytest` 或 `python -m pytest`。

## 第二层：MCP 开发验证

这是当前最主要的验证路径。启动 `moss-as-mcp`，通过 AI 编码工具的 MCP 连接执行 CTML，验证 app 在真实 Matrix 环境中正常工作。

```
1. moss-as-mcp                      # 启动 MCP server
2. AI 编码工具连接 MCP
3. 在 MCP 对话中执行 CTML 调用你的 app
4. 确认返回值正确
```

MCP 路径验证的是完整链路：app 启动 → Matrix 注册 → channel 发现 → CTML 执行 → 结果返回。这是今天的端到端验收标准。

未来 Ghost 运行时具备自迭代能力后，这个验证步骤将由 Ghost 自主完成——它自己 create、start、CTML 调用、observe 结果、stop。MCP 路径是这一能力的前身。

## 第三层：运行时集成

将 app 加入 Mode 的 `bringup_apps`，启动 Ghost 运行时，确认 app 随 Mode 自动拉起并正常工作。

```yaml
# MODE.md
bringup_apps:
  - 'your_group/your_app'
```

这一层验证的是 app 在完整 MOSS 环境中的行为——自动启动、与其他 channel 的协作、Ghost 的自主调用。

## 深入路径

- CTML 语法：`moss ctml read`
- `ctml_shell_test` 源码：搜索项目中该函数的定义和使用
- Channel 测试的完整 API：读 `channel_builder` 中 `test_channel()` 和 `ctml_shell_test()` 的源码
- MCP 连接配置：`moss-as-mcp` 的使用方式
