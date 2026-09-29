# Dependency Governance — Execution Plan

2026-07-27. 调研完成，逐项可执行。

## 1. pyproject.toml 重组

当前 `[project.optional-dependencies]`：

```
zmq:    zmq, aiozmq
redis:  fakeredis, redis
ghost:  pydantic-ai, anthropic
matrix: prompt-toolkit, typer, eclipse-zenoh, python-dotenv
host:   eclipse-zenoh, pexpect, prompt-toolkit, typer, miniaudio, uvloop, websockets, httpx, fastmcp, loadenv
web:    playwright
```

问题：
- 没有 `cli` 组 → 新建
- `matrix` 混入了 CLI 依赖（typer, prompt-toolkit, python-dotenv）→ 拆分
- `host` 是"全抄"桶 → 按层重组
- `web` 无人用（playwright 只在注释里出现）→ 删除
- `rich` 隐式依赖（靠 typer 传递）→ 显式进 cli
- `python-dotenv` 在 matrix 但被 core 用 → 移入 cli

改为：

```toml
[project.optional-dependencies]
# Core layers — each includes all deps of the layer below
cli = [
    "typer>=0.24.1",
    "rich",                       # was implicit via typer
    "python-dotenv>=1.0.0",       # was in [matrix], needed by Project.bootstrap()
]
matrix = [
    "typer>=0.24.1",
    "rich",
    "python-dotenv>=1.0.0",
    "eclipse-zenoh>=1.8.0",       # matrix 唯一新增
]
host = [
    "typer>=0.24.1",
    "rich",
    "python-dotenv>=1.0.0",
    "eclipse-zenoh>=1.8.0",
    "prompt-toolkit>=3.0.52",
    "pexpect>=4.9.0",
    "miniaudio>=0.67",
    "uvloop>=0.22.1",
    "websockets>=15.0.1",
    "httpx>=0.28.0",
    "fastmcp>=3.1.1",
    "loadenv>=0.1.1",
]
ghost = [
    "pydantic-ai>=1.90.0",
    "anthropic>=0.84.0",
]

# Bridges — optional transport backends
zmq = ["zmq>=0.0.0", "aiozmq>=1.0.0"]
redis = ["fakeredis>=2.32.1", "redis>=7.0.1"]
```

删除 `web`。

## 2. depends.py 重写

现状：depend_cli 重复定义两次（都未调用），depend_circus 未调用且不在依赖表，depend_pydantic_ai 未调用，只有 depend_matrix 被 11 处调用。

重写为：

```python
def depend_cli():
    try: import typer, rich, dotenv
    except ImportError:
        raise ImportError("install ghoshell_moss[cli]")

def depend_matrix():
    depend_cli()
    try: import zenoh
    except ImportError:
        raise ImportError("install ghoshell_moss[matrix]")

def depend_host():
    depend_matrix()
    try:
        import prompt_toolkit
        import pexpect
    except ImportError:
        raise ImportError("install ghoshell_moss[host]")

def depend_ghost():
    try: import pydantic_ai, anthropic
    except ImportError:
        raise ImportError("install ghoshell_moss[ghost]")
```

调用点（替换现有 `depend_matrix` 为 `depend_matrix`）：
- bridges/zenoh_bridge/ (4 文件)
- matrix/providers/topic_provider.py
- matrix/providers/moss_session_provider.py
- matrix/networks/zenoh_network.py
- matrix/session/ (2 文件)
- matrix/topics/zenoh_topics.py
- tools/zenoh_helper.py

新增 `depend_host()` 调用点：
- host/tui.py（MossHostTUI 的 prompt_toolkit import）
- host/repl/repl_state.py (prompt_toolkit)
- host/speech/capture/miniaudio_capture.py (miniaudio)
- host/speech/player/miniaudio_player.py (miniaudio)
- core/terminal/pexpect_session.py (pexpect)

新增 `depend_ghost()` 调用点：
- agents/memento_pydantic_agent/impl.py (pydantic_ai)
- ghosts/atom/_runtime.py (anthropic)
- core/concepts/tools.py (pydantic_ai)

## 3. python-dotenv 惰性导入

`core/blueprint/project.py:26` 的 `import dotenv` 改为在 `HostMode.bootstrap()`(L429) 和 `Project.bootstrap()`(L581) 内部惰性 `import dotenv`。

三层防御：
1. 顶层不再 import dotenv
2. bootstrap 内惰性 import，失败抛清晰的 ImportError
3. CLI main.py 启动时先调 `depend_cli()` 确保 dotenv 可用

## 4. CLI main.py depends 检查

调研结论：所有 CLI 子命令模块的顶层 import 都不依赖 zenoh。zenoh 的依赖在 matrix 实现层（matrix/session/ 等），不在 CLI 层。

两种策略：

A) **全注册 + 运行时报错**（简单）：main.py 始终注册所有子命令。`moss nodes run` 执行时才调 `depend_matrix()`，无 zenoh 时报清晰错误。实现成本最低。

B) **条件注册**（用户要求）：main.py 启动时尝试 `depend_matrix()`，失败则跳过 nodes/networks/manifests 的注册。用户看到更干净的帮助输出。

建议 A — 所有 CLI 模块都能安全 import，运行时 depends 检查已经在更深层存在（depend_matrix 11 处调用）。条件注册增加了 CLI 启动时的副作用且收益有限。

## 5. uv.lock + 回归

- 改完 pyproject.toml 后 `uv lock`
- 在 `.ai_partners/regressions/dependency-install/` 建 REGRESSION.md
- 回归步骤：Python 3.10 venv → pip install .[cli] → moss --help / moss ground 可用 → pip install .[matrix] → moss nodes list 可运行 → pip install .[host] → moss-shell 可启动 → pip install .[ghost] → import pydantic_ai 可用

## 执行顺序

```
1. pyproject.toml 重组
2. python-dotenv 惰性导入 (project.py)
3. depends.py 重写 + 调用点替换
4. CLI main.py depends 检查 (§4-A 或 §4-B)
5. uv lock + 回归文档
```
