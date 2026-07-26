# Dependency Governance

2026-07-26. 依赖分组的重设计 + depends.py 对齐 + 回归方案。

## 分组语义

```
base (no extras):
  纯内核。外部引用 from ghoshell_moss import new_ctml_shell 即可用。
  CTML shell, Channel, 核心概念。不涉及 Project/Matrix/Host。

cli ← base + CLI:
  moss 命令行体系可用。轻量第三方可以用 moss ground 等命令。
  依赖: typer, rich, python-dotenv

matrix ← cli + zenoh:
  Cell 最小依赖。能入网、发现 node、加入 matrix 网络。
  不含 mode 层重依赖（mindflow/nuclei/audio/ml）。
  新增依赖: eclipse-zenoh (唯一新增)

host ← matrix + 运行时:
  主节点最小运行。完整产品形态 — TUI, audio, subprocess, MCP。
  新增依赖: prompt-toolkit, pexpect, miniaudio, uvloop, websockets, httpx, fastmcp, loadenv

ghost (正交):
  AI 模型支持。独立于上述链条。
  依赖: pydantic-ai, anthropic

contrib (非核心):
  zmq: zmq, aiozmq
  redis: fakeredis, redis
  web: playwright
```

## pyproject.toml 变更

每个 extra 自描述，列出该层需要的全部依赖。python-dotenv 从 [matrix] 移入 [cli]。

## depends.py 重写

四个函数，各自检查对应层的 imports：

- `depend_cli()` — typer, rich, dotenv
- `depend_matrix()` — depend_cli() + zenoh
- `depend_host()` — depend_matrix() + prompt_toolkit, pexpect, miniaudio, uvloop, websockets, httpx, fastmcp, loadenv
- `depend_ghost()` — pydantic_ai, anthropic

删除旧函数: depend_zenoh (合并到 depend_matrix), depend_circus (从未使用), depend_pydantic_ai (重命名为 depend_ghost), 重复的 depend_cli。

## python-dotenv 惰性导入

`core/blueprint/project.py` 顶层 `import dotenv` → 改为在 `bootstrap()` 方法内惰性导入。避免 base 层被迫依赖 dotenv。

## CLI main.py depends 检查

`moss` 命令启动时按可用 extras 决定暴露哪些子命令组：
- 仅 [cli]: start, codex, project, howtos, features, docs, ground, memento
- [matrix]: + nodes, networks, manifests
- [host]: + modes, ghosts

优雅降级，安装什么层就看见什么命令。

## 回归方案

新 regression set: `dependency-install`。

验证内容:
1. Python 3.10 空 venv 中 `pip install ghoshell-moss` (base) → CTML shell 基本功能
2. `pip install ghoshell-moss[cli]` → moss 命令可运行
3. `pip install ghoshell-moss[matrix]` → moss nodes 可运行
4. `pip install ghoshell-moss[host]` → moss-shell / moss-ghost 可启动（即便立刻退出）
5. `pip install ghoshell-moss[ghost]` → pydantic_ai 可 import

回归文件放在 `.ai_partners/regressions/dependency-install/REGRESSION.md`。
