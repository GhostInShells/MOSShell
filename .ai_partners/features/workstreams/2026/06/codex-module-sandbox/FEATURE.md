---
created: 2026-06-03
depends: []
description: 最小通用 Module 沙盒 — builtins 可控、有状态生命周期、父子沙盒副作用传导。 app + sqlite 验收。
milestone: null
priority: P1
status: draft
status_note: design completed, awaiting implementation
title: Codex Module Sandbox
updated: '2026-06-04'
---

# Codex Module Sandbox

## Motivation

当前 codex 下的 Compiler 提供 ModuleType 容器，但不控制 builtins，无生命周期钩子，无父子变量共享。
需要一个通用的、安全的、可贡献出去的沙盒模块。

三个核心特性：
1. **builtins 可控** — 默认屏蔽危险函数，可自定义
2. **状态持有 + init/destroy 生命周期** — 创建时初始化，销毁前保持状态
3. **父子沙盒共享变量** — child 在 parent 基础上构建，副作用传导到 parent

## 与 module_channel 的定位区分

`module_channel` 是快速反射路径：Python 函数 → CTML command，模型逐个调用。
函数签名即 prompt，适合"我有一组操作，模型选择并调用"的场景。

`Sandbox` 是 REPL 路径：整个命名空间暴露给模型，模型写任意 Python 代码执行。
适合"多步逻辑、循环、条件、变量绑定"无法被单个 command 覆盖的场景
——当执行代码的收益超过逐个 command 反射时，走这条路。

两条路径互补，不互相替代。

## API Contract

```python
SANDBOX_BUILTINS: dict[str, Any]
# 默认安全 builtins — 屏蔽 __import__ / open / eval / exec / compile / input / breakpoint

class Sandbox:
    def __init__(
        self,
        name: str = "__sandbox__",
        *,
        parent: "Sandbox | None" = None,
        builtins: dict[str, Any] | None = SANDBOX_BUILTINS,
        on_init: Callable[["Sandbox"], None] | None = None,
        on_destroy: Callable[["Sandbox"], None] | None = None,
    ): ...

    def exec(self, code: str) -> ExecutionResult:
        """执行代码，捕获 stdout，返回 (returns, std_output)"""
        ...

    def get(self, name: str) -> Any: ...
    def set(self, name: str, value: Any) -> None: ...
    def close(self) -> None: ...
    def __enter__ / __exit__ ...
```

## Key Decisions

### KD1: 不预设等级

不做 minimal/restricted/default 预设。给一个 sensible default（屏蔽危险函数），
更严格或更宽松时传自定义 dict 或 None。极端安全走 safe mode 审核（另一个会话的草案）。

### KD2: 删掉 injections，保留 on_init

不需要两个入口。on_init 内部用 `sandbox.set()` 注入。简单场景直接手动 set，不需要 on_init。

### KD3: exec 捕获 stdout

exec() 返回 `ExecutionResult(returns, std_output)`。模型 print 被捕获到 std_output，
返回值通过 `__result__` 获取。

### KD4: 父子共享 `__dict__` 引用

child 的 `module.__dict__` 复用 parent 的。副作用直接写入 parent。child close() 不销毁 parent。

## Implementation Notes

- 路径: `src/ghoshell_moss/core/codex/sandbox.py`
- 基于 Compiler 的 ModuleType 构建模式，不另起炉灶
- 目标 < 200 行
- 单元测试：用简单 Python 函数验证沙盒核心能力，不依赖 sqlite
- 验收：app + sqlite — 注入 sqlite3 封装，模型操作数据库，child 写 → parent 读