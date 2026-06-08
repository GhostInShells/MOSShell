# L1. Hello World App — 从零创建你的第一个 MOSS App

> Written by deepseek-v4-pro, 2026-06-02

**15 分钟，从头创建你的第一个 MOSS App，完成 create → start → call → stop 全闭环。**

## 你需要知道什么

这些是必要最小化知识。每条都能通过命令拿到：

- `moss apps create --help` — 了解 App 创建脚手架
- `moss ctml read` — 理解 CTML 调用语法
- `moss codex blueprint channel_builder` — 理解 Channel 构建 API

## 你要做什么

创建一个叫 `examples/hello` 的 App — 独立的子进程，向 MOSS 注册一个 Channel，让 AI 可以通过 CTML 调用你的命令。

## 你需要什么

- MOSS 已安装 (`.venv/bin/moss` 可用)
- MOSS 运行时在跑 (REPL，或 MCP——可选的开发时调试方案)
  - MCP 用法：`moss howtos list` 中找 MCP 相关文档
- 当前 workspace 是 `.moss_ws`

## 第一步：创建 App

```bash
.venv/bin/moss apps create examples/hello -d "A hello-world app"
```

这会在 App 目录下生成三个文件：
- `APP.md` — 启动配置 (executable, script, workers...)
- `main.py` — 入口脚本 (现在还是 helloworld 模板)
- `.gitignore` — 忽略本 App 的运行时数据

用 `moss apps show examples/hello` 可以看到完整信息。

## 第二步：写 Channel 代码

打开 `main.py`，改成这样：

```python
"""Hello World — the minimal Channel App."""

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(
    name="examples_hello",
    description="A hello-world channel. Greet someone or do math.",
)

@channel.build.command()
async def greet(name: str = "World") -> str:
    """Say hello."""
    return f"Hello, {name}! I'm running inside a MOSS App."

@channel.build.command()
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@channel.build.context_messages
async def context() -> list[str]:
    return ["[Hello App] Ready."]

async def main(matrix: Matrix):
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

几个要点：
- `new_channel(name=...)` — 这个名字决定 CTML 调用路径 `apps.examples_hello:greet`
- `@channel.build.command()` — 把 async 函数反射成 CTML 可调用的 Command
- `@channel.build.context_messages` — 动态信息，每次刷新时 AI 可见
- `Matrix.discover().run(main)` — 标准入口，自动发现当前 Cell 身份并管理生命周期

## 第三步：刷新 AppStore 并启动

如果你在 MCP 里，先用 CTML 刷新运行时的 App 列表（新创建的 App 运行中可能还没缓存）：

```ctml
<apps:list_apps />
```

看到 `examples/hello: [STOPPED]` 之后，启动：

```ctml
<apps:start fullname="examples/hello" timeout="5.0" />
```

等它显示 `[OK] App channel connected and ready`。

如果你用 CLI 调试，也可以前台跑：

```bash
.venv/bin/moss apps test examples/hello
```

## 第四步：调用你的命令

检查 `moss_dynamic`，你会在 Channel 树里看到 `apps.examples_hello`：

```
<channel name="apps.examples_hello">
A hello-world channel. Greet someone or do math.
<context>
[Hello App] Ready.
</context>
async def greet(name: str = 'World') -> str
async def add(a: float, b: float) -> float
</channel>
```

现在用 CTML 调用：

```ctml
<apps.examples_hello:greet name="MOSS" />
```

返回：`Hello, MOSS! I'm running inside a MOSS App.`

```ctml
<apps.examples_hello:add a="3.0" b="7.0" />
```

返回：`10.0`

## 第五步：停止

```ctml
<apps:stop fullname="examples/hello" />
```

## 你刚做了什么

1. `moss apps create` — 用脚手架生成 App 目录
2. 写了 20 行 Channel 代码 — 两个命令 + 一段 context
3. 刷新 AppStore — 让运行时看到新 App
4. `apps:start` — Circus 拉起子进程，Matrix 注册 Channel
5. CTML 调用 `greet` 和 `add` — 跨进程命令执行，结果返回
6. `apps:stop` — 优雅关闭

这就是 MOSS App 开发的基本循环。下一步：把 `greet` 换成控制真实设备的命令，或者给 App 加 `pyproject.toml` 做独立依赖隔离。

## 相关文档

用 CLI 命令自行探索：

- `moss codex blueprint channel_builder` — Channel 构建 API
- `moss docs read app-system` — App 体系论述
- `moss howtos read app-dev/build-an-app` — App 开发决策路径
- `moss howtos read app-dev/test-an-app` — 三层递进测试
- `moss apps list` — 发现已有 App 作为参考

---

## 验证记录

| 时间 | 模型 | 备注 |
|------|------|------|
| 2026-06-02 03:18 CST | deepseek-v4-pro | 全闭环走通：create → write → list_apps → start → greet → add → stop |
