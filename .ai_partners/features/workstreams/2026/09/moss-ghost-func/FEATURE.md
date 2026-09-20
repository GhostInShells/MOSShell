---
title: Moss Ghost Func
status: parked
priority: P2
created: 2026-09-21
updated: 2026-09-21
depends: []
milestone:
description: >-
  幽灵函数 —— 把普通函数 decorator 成"声明 + 运行时物化实现"的原语：有 IoC 绑定就返回实现，
  没有则反射 interface、让模型在 workflow 里生成实现代码、eval 进缓存、沙箱实例化。
status_note: >-
  方向与机制已在 toy 里闭环（src/ghoshell_moss/types/ghost_func.py），但不在当前迭代优先级。
  先落提案 + 伪代码 + 预言，机制正式立项时再排期。
---

# Moss Ghost Func（幽灵函数）

> Use `moss features set-status moss-ghost-func <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

核心问题**不是**有上下文的分类器——那是另一个更窄的域。真正要问的是：

> 能否定义一种 Python 原语，让 moss project 里的函数可以被 decorator 快速"物化"出实现？

现状：`LLMFuncs` 是引擎（宽边界），但"函数怎么被声明、声明之后实现怎么自动出现"没有机制。每个消费者手写 instruction + result model + parser，于是腐烂——库里的三份手工 rubric 副本就是证据。

2024 年的 `ModelFunc` / `AIFunc` / `GhostFunc` 已经验证了"定义与实现分离"这条路是通的：`GhostFunc` 抓函数签名、让模型生成函数体、缓存进 `.ghost_func.yml`、可重生成。幽灵函数就是把这条路收成一种**比类更轻的 IoC 反射原语**——声明是函数，实现是运行时物化的，可替换。

## 核心机制（伪代码）

```python
# 声明：一个普通函数就是 interface
@ghost_func
def summarize(text: str, max_words: int = 100) -> str: ...

# 首次运行 summarize(...):
#   if IoC 有绑定:            → 返回绑定实现，直接调用
#   else:
#     interface = 反射签名 + return hint
#     ctx       = IoC 拿 contracts + 上下文引用的包树
#     source    = 模型在 workflow 里设计实现代码 (code as prompt)
#     candidate = eval(source, 沙箱 module)          # 隔离，不污染模块全局
#     旁路检查:   跑样例 / 静态检查 / 契约校验        # 生成 ≠ 直接上线
#     通过      → source 落盘缓存 (按环境 hash 寻址)
#                 沙箱 module type 实例化成函数
#     失败      → 迭代或抛未实现
#
# 环境 hash 变了 → 缓存失效，重新生成
```

## 关键点

1. **声明 = 函数**（名字 + 入参 + return hint）→ 边界，不替换；**实现 = 缓存里 eval 的源码** → 可替换。这就是"实现可替换、边界不替换"在函数粒度上的形态。
2. **环境 hash = 失效键**：依赖 / 包树 / 配置变了就重生成，和编译缓存同构。
3. **旁路检查是质量闸**：模型生成的代码必须先过检查再成为实现，不是"生成即上线"。
4. **沙箱 module type**：`eval` 隔离，不污染模块全局命名空间。
5. **比"类 + IoC 注册"轻**：不用为每个 func 声明类、注册 provider。函数签名本身就是 interface，`module:qualname` 本身就是身份。

   已有 toy 验证（`src/ghoshell_moss/types/ghost_func.py`）闭环了四条：`bind`/`GhostFuncNotImplementedError`（装线点 + 未实现哨兵）、`ParamSpec`（签名透传）、`return_type`（R 槽读成真类型）、impl 一次 build 后缓存。

## 完整实现要配套

- **CLI**：`moss ghost-func ...` —— 扫描 / 生成 / 检查 / 列出"已实现 vs 未实现"。
- **预扫描 / 预生成**：不等到运行时首次调用才物化，可批量化预热。
- **质量检查**：旁路检查的生产化（样例集、静态检查、契约校验）。
- **原语**：拿到某个 ghost func 的原始执行代码（introspection / 自解释）。
- **per-project**：函数级、模块级的自定义实现机制，不是全局。

## 预言

下一代高级编程语言会原生带这种机制：`compile` 动作由模型生成高层代码，"interface 声明 + 链接时物化"不再是动态语言打的补丁，而是语言的一等公民。moss ghost func 是这条路上的一个早期探针。

## 状态

parked —— 方向与机制已验证，但不在当前迭代优先级。本文件承载提案 + 伪代码 + 预言，机制正式立项时再排期。
